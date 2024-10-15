import os
import sys
from dataclasses import dataclass
from typing import Optional

from omegaconf import MISSING, OmegaConf

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(BASE_DIR))
sys.path.append(ROOT_DIR)

from efficientvit.aecore.evaluator import Evaluator, EvaluatorConfig
from efficientvit.apps.utils.dist import is_master


@dataclass
class EvalAEModelConfig:
    dataset: str = MISSING
    model: str = MISSING
    amp: str = "fp32"
    pretrained_path: Optional[str] = None
    run_dir: str = MISSING


def main():
    cfg: EvalAEModelConfig = OmegaConf.to_object(
        OmegaConf.merge(OmegaConf.structured(EvalAEModelConfig), OmegaConf.from_cli())
    )

    evaluator_cfg: EvaluatorConfig = OmegaConf.structured(EvaluatorConfig)
    evaluator_cfg.run_dir = cfg.run_dir
    if cfg.dataset == "imagenet_512":
        evaluator_cfg.resolution = 512
        evaluator_cfg.dataset = "imagenet"
        evaluator_cfg.imagenet.batch_size = 64
        evaluator_cfg.fid.ref_path = "assets/data/fid/imagenet_512_val.npz"
    else:
        raise NotImplementedError
    evaluator_cfg.model = cfg.model
    evaluator_cfg.amp = cfg.amp
    evaluator_cfg = OmegaConf.to_object(evaluator_cfg)

    evaluator = Evaluator(evaluator_cfg)
    valid_info_dict = evaluator.evaluate(-1)
    if is_master():
        for key, value in valid_info_dict.items():
            if key in ["fid", "psnr", "ssim", "lpips"]:
                print(f"{key}: {value}")


if __name__ == "__main__":
    main()
