import os
import sys

from omegaconf import OmegaConf

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(BASE_DIR))
sys.path.append(ROOT_DIR)

from efficientvit.diffusioncore.trainer import Trainer, TrainerConfig


def main():
    cfg: TrainerConfig = OmegaConf.to_object(OmegaConf.merge(OmegaConf.structured(TrainerConfig), OmegaConf.from_cli()))
    trainer = Trainer(cfg)
    trainer.train()


if __name__ == "__main__":
    main()
