import torch
from pytorch_lightning.cli import LightningCLI
from pytorch_lightning.strategies import DDPStrategy

torch.set_float32_matmul_precision("high")


def cli_main():
    cli = LightningCLI(
        trainer_defaults={
            "accelerator": "gpu",
            "strategy": "ddp",
            "devices": -1,
            "num_sanity_val_steps": 2,
            "check_val_every_n_epoch": 1,
            "log_every_n_steps": 100,
            "sync_batchnorm": True,
            "benchmark": True,
        },
        save_config_kwargs={
            "config_filename": "config.yaml",
            "overwrite": True,
        },
    )


if __name__ == "__main__":
    cli_main()
