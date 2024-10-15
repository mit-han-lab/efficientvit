import os
import sys
from dataclasses import dataclass

from omegaconf import MISSING, OmegaConf

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(BASE_DIR))
sys.path.append(ROOT_DIR)

from efficientvit.apps.utils.dist import is_master
from efficientvit.diffusion_model_zoo import DCAE_Diffusion_HF, create_dc_ae_diffusion_model_cfg
from efficientvit.diffusioncore.evaluator import Evaluator, EvaluatorConfig


@dataclass
class EvalDiffusionModelConfig:
    dataset: str = MISSING
    model: str = MISSING
    amp: str = "fp32"
    autoencoder_dtype: str = "fp32"
    cfg_scale: float = 1.0
    run_dir: str = MISSING


def main():
    cfg: EvalDiffusionModelConfig = OmegaConf.to_object(
        OmegaConf.merge(OmegaConf.structured(EvalDiffusionModelConfig), OmegaConf.from_cli())
    )

    evaluator_cfg: EvaluatorConfig = OmegaConf.structured(EvaluatorConfig)
    evaluator_cfg.run_dir = cfg.run_dir
    if cfg.dataset == "imagenet_512":
        evaluator_cfg.resolution = 512
        evaluator_cfg.evaluate_dataset = "sample_class"
        evaluator_cfg.sample_class.batch_size = 64
    else:
        raise NotImplementedError
    evaluator_cfg = OmegaConf.merge(evaluator_cfg, OmegaConf.structured(create_dc_ae_diffusion_model_cfg(cfg.model)))
    evaluator_cfg.amp = cfg.amp
    evaluator_cfg.autoencoder_dtype = cfg.autoencoder_dtype
    evaluator_cfg.cfg_scale = cfg.cfg_scale
    evaluator_cfg = OmegaConf.to_object(evaluator_cfg)
    evaluator = Evaluator(evaluator_cfg)
    dc_ae_diffusion = DCAE_Diffusion_HF.from_pretrained(f"mit-han-lab/{cfg.model}")
    evaluator.network.load_state_dict(dc_ae_diffusion.diffusion_model.state_dict())
    valid_info_dict = evaluator.evaluate(-1)
    if is_master():
        for key, value in valid_info_dict.items():
            if key in ["fid"]:
                print(f"{key}: {value}")


if __name__ == "__main__":
    main()
