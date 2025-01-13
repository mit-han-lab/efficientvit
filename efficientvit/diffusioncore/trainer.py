import hashlib
import itertools
import json
import os
import shutil
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import torch
import wandb
from omegaconf import OmegaConf
from tqdm import tqdm

from efficientvit.apps.utils.dist import dist_barrier, get_dist_size, is_dist_initialized, is_master, sync_tensor
from efficientvit.apps.utils.ema import EMA
from efficientvit.apps.utils.lr import ConstantLRwithWarmup
from efficientvit.apps.utils.metric import AverageMeter
from efficientvit.diffusioncore.data_provider.latent_imagenet import (
    LatentImageNetDataProvider,
    LatentImageNetDataProviderConfig,
)
from efficientvit.diffusioncore.evaluator import Evaluator, EvaluatorConfig

__all__ = ["OptimizerConfig", "LRSchedulerConfig", "TrainerConfig", "Trainer"]


@dataclass
class OptimizerConfig:
    name: str = "adamw"
    lr: float = 9.375e-7
    warmup_lr: float = 0.0
    weight_decay: float = 0.06
    no_wd_keys: tuple[str, ...] = ()
    betas: tuple[float, float] = (0.99, 0.99)


@dataclass
class LRSchedulerConfig:
    name: Any = "cosine_annealing"
    warmup_steps: int = 1000


@dataclass
class TrainerConfig(EvaluatorConfig):
    train_dataset: str = "latent"
    latent_imagenet: LatentImageNetDataProviderConfig = field(default_factory=LatentImageNetDataProviderConfig)

    resume: bool = True
    resume_path: Optional[str] = None
    resume_schedule: bool = True
    num_epochs: Optional[int] = None
    max_steps: Optional[int] = None
    clip_grad: Optional[float] = None
    num_store_images: int = 64
    save_checkpoint_steps: int = 1000
    evaluate_steps: int = 20000

    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    lr_scheduler: LRSchedulerConfig = field(default_factory=LRSchedulerConfig)
    log: bool = True
    wandb_entity: Optional[str] = None
    wandb_project: Optional[str] = None

    ema_decay: float = 0.9998
    ema_warmup_steps: int = 2000
    evaluate_ema: bool = True


class Trainer(Evaluator):
    def __init__(self, cfg: TrainerConfig):
        super().__init__(cfg)
        self.cfg: TrainerConfig

        if cfg.train_dataset == "latent_imagenet":
            self.train_data_provider = LatentImageNetDataProvider(cfg.latent_imagenet)
        else:
            raise NotImplementedError

        self.setup_run_dir()
        self.setup_optimizer()
        self.setup_lr_scheduler()
        self.ema = EMA(self.network.get_trainable_modules(), cfg.ema_decay, cfg.ema_warmup_steps)
        self.setup_logger()

        if cfg.compute_fid:
            self.best_fid = torch.inf
        if is_dist_initialized():
            print(f"rank {self.rank}, train {len(self.train_data_provider.train)}")
        if self.enable_amp:
            self.scaler = torch.GradScaler()
        self.train_generator = torch.Generator(device=torch.device("cuda"))
        self.train_generator.manual_seed(cfg.seed + self.rank)
        self.global_step = 0
        self.start_epoch = 0
        if cfg.resume:
            self.try_resume_from_checkpoint()
        self.NaN_detected = False

    def setup_run_dir(self) -> None:
        self.checkpoint_dir = self.cfg.run_dir
        if is_master():
            if not self.cfg.resume or not os.path.exists(self.cfg.run_dir):
                os.makedirs(self.cfg.run_dir, exist_ok=False)
                OmegaConf.save(self.cfg, os.path.join(self.cfg.run_dir, "config.yaml"))
                with open(os.path.join(self.cfg.run_dir, "model.txt"), "w") as f:
                    f.write(f"{self.network}")
        if is_dist_initialized():
            dist_barrier()

    def setup_optimizer(self):
        param_dict = {}
        trainable_modules = self.network.get_trainable_modules()
        weight_decay, init_lr = self.cfg.optimizer.weight_decay, self.cfg.optimizer.lr
        no_wd_keys = self.cfg.optimizer.no_wd_keys
        for module in trainable_modules.values():
            for name, param in module.named_parameters():
                if not param.requires_grad:
                    continue
                opt_config = [weight_decay, init_lr]
                if any(key in name for key in no_wd_keys):
                    opt_config[0] = 0.0
                opt_key = json.dumps(opt_config)
                param_dict[opt_key] = param_dict.get(opt_key, []) + [param]

        net_params = []
        for opt_key, param_list in param_dict.items():
            wd, lr = json.loads(opt_key)
            net_params.append({"params": param_list, "weight_decay": wd, "lr": lr})

        if self.cfg.optimizer.name == "adamw":
            if len(no_wd_keys) > 0:
                self.optimizer = torch.optim.AdamW(net_params, lr=self.cfg.optimizer.lr, betas=self.cfg.optimizer.betas)
            else:
                self.optimizer = torch.optim.AdamW(
                    trainable_modules.parameters(),
                    lr=self.cfg.optimizer.lr,
                    betas=self.cfg.optimizer.betas,
                    weight_decay=self.cfg.optimizer.weight_decay,
                )
        else:
            raise ValueError(f"Optimizer {self.cfg.optimizer.name} is not supported")

    def setup_lr_scheduler(self):
        lr_scheduler_name = self.cfg.lr_scheduler.name
        if lr_scheduler_name == "cosine_annealing":
            num_iters = self.cfg.num_epochs * len(self.train_data_provider.train)
            self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=num_iters, eta_min=0.0)
        elif lr_scheduler_name == "constant":
            self.lr_scheduler = torch.optim.lr_scheduler.ConstantLR(self.optimizer, factor=1.0)
        elif lr_scheduler_name == "constant_with_warmup":
            self.lr_scheduler = ConstantLRwithWarmup(self.optimizer, self.cfg.lr_scheduler.warmup_steps, 0)
        else:
            raise NotImplementedError

    def setup_logger(self):
        if is_master():
            self.f_log = open(os.path.join(self.cfg.run_dir, "log.txt"), "a")
        else:
            self.f_log = sys.stdout
        self.print_and_f_log(f"run_dir: {self.cfg.run_dir}", flush=True)

        self.log_to_wandb = self.cfg.log and (self.cfg.wandb_entity is not None or self.cfg.wandb_project is not None)
        if not self.log_to_wandb or not is_master():
            return
        self.logger = wandb.init(
            entity=self.cfg.wandb_entity,
            project=self.cfg.wandb_project,
            config=vars(self.cfg),
            name=self.cfg.run_dir,
            id=hashlib.sha1(self.cfg.run_dir.encode("utf-8")).hexdigest(),
            resume="allow",
            tags=[self.cfg.model, str(self.cfg.resolution)],
        )
        self.log_dicts = []

    def print_and_f_log(self, message: str, flush: bool = False):
        if is_master():
            print(message, end="", flush=flush)
            self.f_log.write(message)
            if flush:
                self.f_log.flush()

    def cache_log(self, log_dict: dict[str, Any]):
        if not self.log_to_wandb or not is_master():
            return
        self.log_dicts.append((self.global_step, log_dict))

    def log(self):
        if not self.log_to_wandb or not is_master():
            return
        for step, log_dict in self.log_dicts:
            self.logger.log(log_dict, step=step, commit=True)
        self.log_dicts = []

    def save_model(self, model_name: str, epoch: int, only_state_dict: bool = False):
        if not only_state_dict:
            train_generator_state = self.train_generator.get_state()[None]
            torch_rng_state = torch.get_rng_state()[None]
            torch_cuda_rng_state = torch.cuda.get_rng_state()[None]
            if is_dist_initialized():
                train_generator_state = sync_tensor(train_generator_state.cuda(), reduce="cat").cpu()
                torch_rng_state = sync_tensor(torch_rng_state.cuda(), reduce="cat").cpu()
                torch_cuda_rng_state = sync_tensor(torch_cuda_rng_state.cuda(), reduce="cat").cpu()
        if not is_master():
            return
        model_to_save = self.network
        state_dict = model_to_save.get_trainable_modules().state_dict()
        if only_state_dict:
            checkpoint = {"state_dict": state_dict}
        else:
            checkpoint = {
                "state_dict": state_dict,
                "epoch": epoch,
                "global_step": self.global_step,
                "scaler": self.scaler.state_dict() if self.enable_amp else None,
                "optimizer": self.optimizer.state_dict(),
                "lr_scheduler": self.lr_scheduler.state_dict(),
                "train_generator_state": train_generator_state,
                "torch_rng_state": torch_rng_state,
                "torch_cuda_rng_state": torch_cuda_rng_state,
            }
            if self.ema is not None:
                checkpoint["ema"] = self.ema.state_dict()
            if self.cfg.compute_fid:
                checkpoint["best_fid"] = self.best_fid

        model_path = os.path.join(self.checkpoint_dir, model_name)
        if model_name == "checkpoint.pt":  # avoid partial saved checkpoints
            model_path_ = os.path.join(self.checkpoint_dir, "checkpoint_.pt")
            torch.save(checkpoint, model_path_)
            shutil.copy(model_path_, model_path)
        else:
            torch.save(checkpoint, model_path)
        self.print_and_f_log(f"save model to {model_path} at step {self.global_step}\n", flush=True)

    def resume_from_checkpoint(self, checkpoint_path: str):
        self.print_and_f_log(f"loading checkpoint {checkpoint_path}\n")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

        # load checkpoint
        self.network.get_trainable_modules().load_state_dict(checkpoint["state_dict"], strict=False)
        self.ema.load_state_dict(checkpoint[f"ema"])
        self.optimizer.load_state_dict(checkpoint[f"optimizer"])
        self.print_and_f_log(f"optimizer loaded\n")
        if self.enable_amp:
            self.scaler.load_state_dict(checkpoint["scaler"])
            self.print_and_f_log(f"scaler loaded\n")
        if self.cfg.resume_schedule:
            self.start_epoch = checkpoint["epoch"]
            self.print_and_f_log(f"epoch: {self.start_epoch}\n")
            self.global_step = checkpoint["global_step"]
            self.print_and_f_log(f"global_step={self.global_step}\n")
            self.lr_scheduler.load_state_dict(checkpoint[f"lr_scheduler"])
            self.print_and_f_log(f"lr scheduler loaded\n")
            if get_dist_size() == checkpoint["train_generator_state"].shape[0]:
                self.train_generator.set_state(checkpoint["train_generator_state"][self.rank])
                self.print_and_f_log(f"train generator state loaded\n")
                torch.set_rng_state(checkpoint["torch_rng_state"][self.rank])
                self.print_and_f_log(f"torch rng state loaded\n")
                torch.cuda.set_rng_state(checkpoint["torch_cuda_rng_state"][self.rank])
                self.print_and_f_log(f"torch cuda rng state loaded\n")
            else:
                self.print_and_f_log(f"warning: failed to load rng states\n")
            if self.cfg.compute_fid:
                self.best_fid = checkpoint["best_fid"]
                self.print_and_f_log(f"best_fid={self.best_fid:.2f}\n")
        self.print_and_f_log(f"checkpoint {checkpoint_path} loaded\n", flush=True)

    def try_resume_from_checkpoint(self):
        if os.path.exists(os.path.join(self.checkpoint_dir, "checkpoint.pt")) or os.path.exists(
            os.path.join(self.checkpoint_dir, "checkpoint_.pt")
        ):
            checkpoint_path = os.path.join(self.checkpoint_dir, "checkpoint.pt")
            try:
                self.resume_from_checkpoint(checkpoint_path)
            except Exception as e:
                self.print_and_f_log(f"got error {e} when loading from {checkpoint_path}")
                self.resume_from_checkpoint(os.path.join(self.checkpoint_dir, "checkpoint_.pt"))
        elif self.cfg.resume_path is not None:
            self.resume_from_checkpoint(self.cfg.resume_path)
        else:
            self.print_and_f_log("can not find a checkpoint, will train from scratch\n")

    def after_step(self, loss: torch.Tensor) -> dict[str, Any]:
        info: dict[str, Any] = {}
        self.optimizer.zero_grad()
        if self.enable_amp:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
        else:
            loss.backward()
        # gradient clip
        if self.cfg.clip_grad is not None:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.clip_grad)
        else:
            grad_norm = torch.norm(
                torch.tensor([torch.norm(p.grad) for p in self.model.parameters() if p.grad is not None])
            )
        info["grad_norm"] = grad_norm.item()
        # step
        if self.enable_amp:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()

        self.lr_scheduler.step()

        self.ema.step(self.network.get_trainable_modules(), self.global_step)

        return info

    def train_one_epoch(self, epoch: int, f_log=sys.stdout) -> dict[str, Any]:
        self.model.train()
        self.train_data_provider.set_epoch(epoch)
        train_loss_dict = dict()

        with tqdm(
            total=len(self.train_data_provider.train),
            desc="Train Epoch #{}".format(epoch + 1),
            disable=not is_master(),
            file=f_log,
            mininterval=10.0,
        ) as t:
            if len(self.train_data_provider.train) * epoch != self.global_step:
                self.print_and_f_log(
                    f"skipping first {self.global_step-len(self.train_data_provider.train)*epoch} steps", flush=True
                )
                self.train_data_provider.set_batch_idx(self.global_step - len(self.train_data_provider.train) * epoch)
                t.update(self.global_step - len(self.train_data_provider.train) * epoch)
            last_step_time = time.time()
            for _, (images, labels) in enumerate(self.train_data_provider.train):
                self.global_step += 1
                log_dict: dict[str, Any] = {}
                step_start_time = time.time()
                log_dict["load_data_time"] = step_start_time - last_step_time
                # preprocessing
                images = images.cuda()
                labels = labels.cuda()
                # forward
                with torch.autocast(device_type="cuda", dtype=self.amp_dtype, enabled=self.enable_amp):
                    loss, info = self.model(images, labels)
                # backward and update optimizer, lr_scheduler
                after_step_dict = self.after_step(loss)
                # update metrics
                for loss_key, loss_value in info["loss_dict"].items():
                    if loss_key not in train_loss_dict:
                        train_loss_dict[loss_key] = AverageMeter(is_distributed=is_dist_initialized())
                    train_loss_dict[loss_key].update(loss_value, images.shape[0])
                    log_dict[loss_key] = loss_value
                # tqdm
                postfix_dict = {
                    "shape": images.shape,
                    "global_step": self.global_step,
                    "grad_norm": after_step_dict["grad_norm"],
                }
                postfix_dict["lr"] = self.lr_scheduler.get_lr()[0]
                log_dict["lr"] = self.lr_scheduler.get_lr()[0]
                log_dict["grad_norm"] = after_step_dict["grad_norm"]
                for loss_key in train_loss_dict:
                    postfix_dict[loss_key] = train_loss_dict[loss_key].avg
                t.set_postfix(postfix_dict, refresh=False)
                t.update()

                step_end_time = time.time()
                log_dict["step_time"] = step_end_time - step_start_time

                # detect NaN
                mean_loss = sync_tensor(loss, reduce="mean").item()
                if np.isnan(mean_loss):
                    self.NaN_detected = True
                    self.print_and_f_log(f"NaN detected, break from training")
                    break

                # evaluate
                if self.global_step % self.cfg.evaluate_steps == 0:
                    if self.cfg.evaluate_ema:
                        if self.cfg.model in ["dit", "uvit"]:
                            network = self.ema.shadows[self.cfg.model]
                        else:
                            raise NotImplementedError(f"evaluate ema is not supported for {self.cfg.model}")
                    else:
                        network = None
                    valid_info_dict = self.evaluate(self.global_step, network=network, f_log=self.f_log)
                    self.print_and_f_log(f"valid info dict: {valid_info_dict}\n", flush=True)
                    if self.cfg.compute_fid:
                        self.best_fid = min(valid_info_dict["fid"], self.best_fid)
                    self.save_model(model_name=f"step_{self.global_step}.pt", epoch=epoch)
                    log_dict.update(valid_info_dict)
                    self.model.train()

                self.cache_log(log_dict)

                # save checkpoint
                if self.global_step % self.cfg.save_checkpoint_steps == 0:
                    self.save_model("checkpoint.pt", epoch)
                    self.log()

                if self.cfg.max_steps is not None and self.global_step >= self.cfg.max_steps:
                    self.print_and_f_log(f"max steps {self.cfg.max_steps} reached, breaking from train one epoch")
                    break
                last_step_time = time.time()
        train_info_dict: dict[str, Any] = dict()
        for loss_key in train_loss_dict:
            train_info_dict[loss_key] = train_loss_dict[loss_key].avg
        return train_info_dict

    def train(self) -> None:
        for epoch in itertools.count(start=self.start_epoch):
            if self.cfg.num_epochs is not None and epoch >= self.cfg.num_epochs:
                self.print_and_f_log(f"max epochs {self.cfg.num_epochs} reached, breaking from train")
                break
            train_info_dict = self.train_one_epoch(epoch, self.f_log)
            self.print_and_f_log(f"train info dict: {train_info_dict}\n", flush=True)
            if self.cfg.max_steps is not None and self.global_step >= self.cfg.max_steps:
                self.print_and_f_log(f"max steps {self.cfg.max_steps} reached, breaking from train")
                break
            if self.NaN_detected:
                self.print_and_f_log(f"NaN detected, breaking from train")
                break

        if is_master():
            self.f_log.close()
