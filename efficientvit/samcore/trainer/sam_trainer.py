import random
import sys
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from tqdm import tqdm

from efficientvit.apps.trainer import Trainer
from efficientvit.apps.utils import AverageMeter, get_dist_local_rank, get_dist_size, is_master, sync_tensor
from efficientvit.models.utils import list_join
from efficientvit.samcore.data_provider import SAMDataProvider
from efficientvit.samcore.trainer import SAMRunConfig
from efficientvit.samcore.trainer.utils import compute_boundary_iou, compute_iou, loss_masks, masks_sample_points

__all__ = ["SAMTrainer"]


class SAMTrainer(Trainer):
    def __init__(
        self,
        path: str,
        model: nn.Module,
        data_provider: SAMDataProvider,
    ) -> None:
        super().__init__(
            path=path,
            model=model,
            data_provider=data_provider,
        )

        if is_master():
            self.wandb_log = wandb.init(project="efficientvit-sam")

    def _validate(self, model, data_loader, epoch: int, sub_epoch: int) -> dict[str, Any]:
        val_loss = AverageMeter()
        val_iou = AverageMeter()
        val_iou_boundary = AverageMeter()

        with torch.no_grad():
            with tqdm(
                total=len(data_loader),
                desc=f"Validate Epoch #{epoch + 1}, Sub Epoch #{sub_epoch+1}",
                disable=not is_master(),
                file=sys.stdout,
            ) as t:
                for _, data in enumerate(data_loader):
                    image = data["image"].cuda()
                    masks = data["masks"].cuda()
                    bboxs = data["bboxs"].cuda() * 2 if image.shape[2] == 512 else data["bboxs"].cuda()
                    # points = data["points"].cuda() * 2 if image.shape[2] == 512 else data["points"].cuda()

                    bboxs[..., 2] = bboxs[..., 0] + bboxs[..., 2]
                    bboxs[..., 3] = bboxs[..., 1] + bboxs[..., 3]

                    batched_input = []
                    for b_i in range(len(image)):
                        dict_input = dict()

                        dict_input["image"] = image[b_i]
                        dict_input["boxes"] = bboxs[b_i]

                        batched_input.append(dict_input)

                    output, iou_predictions = model(batched_input, True)

                    _, M, _, _, _ = output.shape
                    output = torch.stack(
                        [
                            output[k][torch.arange(M), iou_predictions[k].argmax(-1).squeeze()]
                            for k in range(len(output))
                        ],
                        dim=0,
                    )
                    output = (
                        F.interpolate(output, size=(image.shape[2], image.shape[3]), mode="bilinear")
                        .reshape(-1, image.shape[2], image.shape[3])
                        .unsqueeze(1)
                    )
                    masks = masks.reshape(-1, image.shape[2], image.shape[3]).unsqueeze(1)

                    loss_mask, loss_dice = loss_masks(output, masks, len(output))
                    loss = loss_mask * 20 + loss_dice

                    iou = compute_iou(output, masks * 255)
                    boundary_iou = compute_boundary_iou(output, masks * 255)

                    loss = sync_tensor(loss)
                    iou = sync_tensor(iou)
                    boundary_iou = sync_tensor(boundary_iou)

                    val_loss.update(loss, image.shape[0] * get_dist_size())
                    val_iou.update(iou, image.shape[0] * get_dist_size())
                    val_iou_boundary.update(boundary_iou, image.shape[0] * get_dist_size())

                    t.set_postfix(
                        {
                            "loss": val_loss.avg,
                            "iou": val_iou.avg,
                            "boundary_iou": val_iou_boundary.avg,
                            "bs": image.shape[0] * get_dist_size(),
                        }
                    )
                    t.update()

        if is_master():
            self.wandb_log.log(
                {"val_loss": val_loss.avg, "val_iou": val_iou.avg, "val_boundary_iou": val_iou_boundary.avg}
            )

        return {
            "val_loss": val_loss.avg,
            "val_iou": val_iou.avg,
            "val_boundary_iou": val_iou_boundary.avg,
        }

    def validate(self, model=None, data_loader=None, epoch=0, sub_epoch=0) -> dict[str, Any]:
        model = self.eval_network if model is None else model
        if data_loader is None:
            data_loader = self.data_provider.valid

        model.eval()
        return self._validate(model, data_loader, epoch, sub_epoch)

    def before_step(self, feed_dict: dict[str, Any]) -> dict[str, Any]:
        image = feed_dict["image"].cuda()
        masks = feed_dict["masks"].cuda()
        bboxs = feed_dict["bboxs"].cuda() * 2 if image.shape[2] == 512 else feed_dict["bboxs"].cuda()
        points = feed_dict["points"].cuda() * 2 if image.shape[2] == 512 else feed_dict["points"].cuda()

        bboxs[..., 2] = bboxs[..., 0] + bboxs[..., 2]
        bboxs[..., 3] = bboxs[..., 1] + bboxs[..., 3]

        return {
            "image": image,
            "masks": masks,
            "points": points,
            "bboxs": bboxs,
        }

    def run_step(self, feed_dict: dict[str, Any]) -> dict[str, Any]:
        image = feed_dict["image"]
        masks = feed_dict["masks"]
        bboxs = feed_dict["bboxs"]
        points = feed_dict["points"]

        batched_input = []
        for b_i in range(len(image)):
            dict_input = dict()
            dict_input["image"] = image[b_i]

            if random.random() >= 0.5:
                dict_input["boxes"] = bboxs[b_i]
            else:
                try:
                    n_p = int(random.random() * 10 + 1)
                    dict_input["point_coords"] = masks_sample_points(masks[b_i], k=n_p)
                    if image.shape[2] == 512:
                        dict_input["point_coords"] = dict_input["point_coords"] * 2
                    dict_input["point_labels"] = torch.ones((points[b_i].shape[0], n_p), device=image.device)
                except:
                    dict_input["boxes"] = bboxs[b_i]

            batched_input.append(dict_input)

        with torch.autocast(device_type="cuda", dtype=self.amp_dtype, enabled=self.enable_amp):
            if random.random() >= 0.5:
                output, _ = self.model(batched_input, multimask_output=True)
            else:
                output, _ = self.model(batched_input, multimask_output=False)

            masks = masks.reshape(-1, image.shape[2], image.shape[3]).unsqueeze(1)

            loss_list = []
            for i in range(output.shape[2]):
                output_i = (
                    F.interpolate(output[:, :, i], size=(image.shape[2], image.shape[3]), mode="bilinear")
                    .reshape(-1, image.shape[2], image.shape[3])
                    .unsqueeze(1)
                )
                loss_mask_i, loss_dice_i = loss_masks(output_i, masks, len(output_i), mode="none")
                loss_i = loss_mask_i * 20 + loss_dice_i
                loss_list.append(loss_i)
            loss = torch.stack(loss_list, -1)

            min_indices = torch.argmin(loss, dim=1)
            mask = torch.zeros_like(loss, device=loss.device)
            mask.scatter_(1, min_indices.unsqueeze(1), 1)

            loss = (loss * mask).mean() * loss.shape[-1]

        self.scaler.scale(loss).backward()

        return {"loss": loss, "output": output}

    def _train_one_sub_epoch(self, epoch: int, sub_epoch: int) -> dict[str, Any]:
        train_loss = AverageMeter()

        with tqdm(
            total=len(self.data_provider.train),
            desc=f"Train Epoch #{epoch + 1}, Sub Epoch #{sub_epoch + 1}",
            disable=not is_master(),
            file=sys.stdout,
        ) as t:
            for _, data in enumerate(self.data_provider.train):
                feed_dict = data

                # preprocessing
                feed_dict = self.before_step(feed_dict)
                # clear gradient
                self.optimizer.zero_grad()
                # forward & backward
                output_dict = self.run_step(feed_dict)
                # update: optimizer, lr_scheduler
                self.after_step()

                loss = output_dict["loss"]
                loss = sync_tensor(loss)
                train_loss.update(loss, data["image"].shape[0] * get_dist_size())

                if is_master():
                    self.wandb_log.log(
                        {
                            "train_loss": train_loss.avg,
                            "epoch": epoch,
                            "sub_epoch": sub_epoch,
                            "learning_rate": sorted(set([group["lr"] for group in self.optimizer.param_groups]))[0],
                        }
                    )

                t.set_postfix(
                    {
                        "loss": train_loss.avg,
                        "bs": data["image"].shape[0] * get_dist_size(),
                        "res": data["image"].shape[2],
                        "lr": list_join(
                            sorted(set([group["lr"] for group in self.optimizer.param_groups])),
                            "#",
                            "%.1E",
                        ),
                        "progress": self.run_config.progress,
                    }
                )
                t.update()

        return {
            "train_loss": train_loss.avg,
        }

    def train_one_sub_epoch(self, epoch: int, sub_epoch: int) -> dict[str, Any]:
        self.model.train()

        self.data_provider.set_epoch_and_sub_epoch(epoch, sub_epoch)

        train_info_dict = self._train_one_sub_epoch(epoch, sub_epoch)

        return train_info_dict

    def train(self) -> None:
        for sub_epoch in range(self.start_epoch, self.run_config.n_epochs):
            epoch = sub_epoch // self.data_provider.sub_epochs_per_epoch

            self.train_one_sub_epoch(epoch, sub_epoch)

            val_info_dict = self.validate(epoch=epoch, sub_epoch=sub_epoch)

            val_iou = val_info_dict["val_iou"]
            self.best_val = max(val_iou, self.best_val)

            self.save_model(
                only_state_dict=False,
                epoch=sub_epoch,
                model_name=f"checkpoint_{epoch}_{sub_epoch}.pt",
            )

    def prep_for_training(self, run_config: SAMRunConfig, amp="fp32") -> None:
        self.run_config = run_config
        self.model = nn.parallel.DistributedDataParallel(
            self.model.cuda(),
            device_ids=[get_dist_local_rank()],
            find_unused_parameters=True,
        )

        self.run_config.global_step = 0
        self.run_config.batch_per_epoch = len(self.data_provider.train)
        assert self.run_config.batch_per_epoch > 0, "Training set is empty"

        # build optimizer
        self.optimizer, self.lr_scheduler = self.run_config.build_optimizer(self.model)

        # amp
        self.amp = amp
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.enable_amp)
