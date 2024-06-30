import os
import wandb
import torch
import shutil
import numpy as np
import pytorch_lightning as pl
import matplotlib.pyplot as plt

from typing import Any
from sklearn.metrics import confusion_matrix
from pytorch_lightning.utilities.types import STEP_OUTPUT


# trainer.logger.experiment.dir:  /import/c4dm-datasets-ext/wandb/run-20230826_192342-4zjz1inp/files


class LogAudioCallback(pl.Callback):
    def __init__(self) -> None:
        super().__init__()

    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        # Only log audio for the first batch
        if batch_idx > 0:
            return
        # Log audio to wandb
        else:
            data_dict = outputs
            silence_gap = 48000

            # check the batch size
            batch_size = data_dict["input_audio_A"].shape[0]

            for bidx in range(min(4, batch_size)):
                audio_tensors = []
                for key in ["input_audio_A", "target_audio_A", "output_audio_A"]:
                    audio_segment = data_dict[key][bidx, ...]
                    chs, seq_len = audio_segment.shape

                    if chs == 1:  # make it stereo
                        audio_segment = audio_segment.repeat(2, 1)

                    audio_tensors.append(audio_segment)

                # check the length of the audio tensors
                total_audio_length = sum(
                    [audio_tensor.shape[-1] for audio_tensor in audio_tensors]
                )

                # create a single tensor to hold the audio
                output_tensor = torch.zeros((total_audio_length + 2 * silence_gap, 2))

                # copy the audio tensors into the single tensor
                start_idx = 0
                for audio_tensor in audio_tensors:
                    end_idx = start_idx + audio_tensor.shape[-1]
                    output_tensor[start_idx:end_idx, :] = audio_tensor.permute(1, 0)
                    start_idx = end_idx + silence_gap

                trainer.logger.experiment.log(
                    {
                        f"{bidx}-audio": wandb.Audio(
                            output_tensor.cpu().numpy(), sample_rate=48000
                        ),
                    }
                )


class MoveConfigCallback(pl.Callback):
    def __init__(self, project_name: str) -> None:
        super().__init__()
        self.project_name = project_name

    def on_fit_start(self, trainer, pl_module):
        full_run_dir = trainer.logger.experiment.dir
        run_id = full_run_dir.split(os.sep)[-2].split("-")[-1]
        src_dir = os.path.join(trainer.log_dir, "config.yaml")
        dest_dir = os.path.join(
            trainer.log_dir, self.project_name, run_id, "checkpoints", "config.yaml"
        )

        run_dir = os.path.join(trainer.log_dir, self.project_name, run_id)
        ckpt_dir = os.path.join(run_dir, "checkpoints")
        os.makedirs(run_dir, exist_ok=True)
        os.makedirs(ckpt_dir, exist_ok=True)

        shutil.copyfile(src_dir, dest_dir)


class ConfusionMatrixCallback(pl.Callback):
    def __init__(self) -> None:
        super().__init__()
        self.outputs = []

    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        logits, labels = outputs
        self.outputs.append(outputs)

    def on_validation_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        labels = list(trainer.datamodule.val_dataset.plugins.keys())

        # Concatenate logits and labels from all batches in the validation set
        all_logits = torch.cat([output[0] for output in self.outputs], dim=0)
        all_labels = torch.cat([output[1] for output in self.outputs], dim=0)

        # Convert logits to predicted labels
        predicted_labels = torch.argmax(all_logits, dim=-1)

        # Compute confusion matrix
        cm = confusion_matrix(all_labels.cpu().numpy(), predicted_labels.cpu().numpy())

        # Plot confusion matrix using matplotlib
        num_classes = cm.shape[0]
        class_labels = [str(i) for i in labels]

        if num_classes != len(class_labels):
            print(
                f"Number of classes ({num_classes}) does not match number of class labels ({len(class_labels)}). Skipping."
            )
            return

        fig, ax = plt.subplots(figsize=(num_classes // 2, num_classes // 2))
        plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        plt.title("Confusion Matrix")
        plt.colorbar()

        tick_marks = np.arange(num_classes)
        plt.xticks(tick_marks, class_labels, rotation=90)
        ax.xaxis.set_tick_params(top=True)
        plt.yticks(tick_marks, class_labels)

        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.tight_layout()

        for i in range(num_classes):
            for j in range(num_classes):
                plt.text(
                    j, i, str(cm[i, j]), horizontalalignment="center", color="black"
                )

        # log confusion matrix to wandb
        trainer.logger.experiment.log({"confusion_matrix": plt})
        plt.close("all")

        # reset outputs
        self.outputs = []
