import os
import glob
import json
import torch
import tarfile
import torchaudio
import numpy as np
import pytorch_lightning as pl

from typing import List, Tuple, Dict, Optional
from st_ito.dataset.utils import tarfile_worker_init_fn, torchaudio_decode


class StyleTransferTarfileDataset(torch.utils.data.IterableDataset):
    def __init__(
        self,
        tar_files: List[str],
        length: int,
        num_examples: int = 10000,
        audio_format: str = "wav",
        input_only: bool = False,
    ):
        super().__init__()
        self.tar_files = tar_files
        self.length = length
        self.num_examples = num_examples
        self.audio_format = audio_format
        self.input_only = input_only

        # get the tar file handles
        self.tar_handles = []
        for tar_file in tar_files:
            tar_handle = tarfile.open(tar_file, "r:")
            tar_handle.next()  # skip the first root directory
            tar_handle.next()  # skip the first example directory
            self.tar_handles.append(tar_handle)

    def __len__(self):
        return self.num_examples

    def next_example(self):
        while True:
            # select tar file at random
            tar_index = np.random.choice(len(self.tar_handles))

            # get the next member (should be root directory of this example)
            members = {}
            done = False
            handle_reset = False
            while not done:
                # get next member
                member = self.tar_handles[tar_index].next()

                if member is None:
                    # we have reached the end of this tar file
                    # so, we will close it and remove open tar handle

                    # close this tar handle
                    self.tar_handles[tar_index].close()

                    # reopen
                    tar_handle = tarfile.open(self.tar_files[tar_index], "r:")
                    tar_handle.next()  # skip the first root directory
                    tar_handle.next()  # skip the first example directory
                    self.tar_handles[tar_index] = tar_handle  # replace handle
                    print(f"Resetting tar handle {self.tar_files[tar_index]}.")
                    handle_reset = True
                    break

                if member.isdir():
                    done = True
                    break

                members[os.path.basename(member.name)] = member

            if handle_reset:
                continue

            # get input file
            input_file = self.tar_handles[tar_index].extractfile(
                members[f"input.{self.audio_format}"]
            )
            audio_input = torchaudio_decode(input_file)

            if self.input_only:

                num_frames = audio_input.shape[1]
                start_frame = torch.randint(0, num_frames - self.length, (1,)).item()
                audio_input = audio_input[:, start_frame : start_frame + self.length]

                # we will generate the output in the system
                audio_output = torch.empty_like(audio_input)
                params = torch.empty(1)

            else:
                # get the output file
                output_file = self.tar_handles[tar_index].extractfile(
                    members[f"output.{self.audio_format}"]
                )
                audio_output = torchaudio_decode(output_file)

                # get the params file
                params_file = self.tar_handles[tar_index].extractfile(
                    members["params.json"]
                )
                params = json.load(params_file)
                # convert params to tensor
                params = torch.tensor(
                    [val for val in params.values()], dtype=torch.float32
                )

                # crop the audio to the desired length
                num_frames = audio_input.shape[1]
                start_frame = torch.randint(0, num_frames - self.length, (1,)).item()
                audio_input = audio_input[:, start_frame : start_frame + self.length]
                audio_output = audio_output[:, start_frame : start_frame + self.length]

                # check for silence in the output
                input_energy = torch.mean(audio_input**2, dim=1)
                output_energy = torch.mean(audio_output**2, dim=1)

                if output_energy < 1e-6 or input_energy < 1e-6:
                    # skip this example
                    continue

            yield audio_input, audio_output, params

    def __iter__(self):
        return iter(self.next_example())


class StyleTransferTarfileDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_tarfiles: List[str],
        val_tarfiles: List[str],
        batch_size: int,
        num_workers: int,
        input_only: bool = False,
        length: int = 262144,
        test_tarfiles: List[str] = None,
        pin_memory: bool = False,
        persistent_workers: bool = True,
        num_train_examples: int = 20000,
        num_val_examples: int = 2000,
        num_test_examples: int = 2000,
        audio_format: str = "wav",
    ) -> None:
        super().__init__()
        """ Pytorch Lightning DataModule for loading data from tar files.
        
        Args:
            train_tarfiles (List[str]): List of tar files to use for training.
            val_tarfiles (List[str]): List of tar files to use for validation.
            batch_size (int): Batch size.
            num_workers (int): Number of workers.
            input_only (bool): Whether to use only the input audio. Defaults to False.
            length (int): Length of audio to return. Defaults to 262144.
            test_tarfiles (List[str]): List of tar files to use for testing.
            pin_memory (bool): Whether to pin memory. Defaults to False.
            persistent_workers (bool): Whether to use persistent workers. Defaults to True.
            num_train_examples (int): Number of training examples to return. Defaults to 20000.
            num_val_examples (int): Number of validation examples to return. Defaults to 2000.
            num_test_examples (int): Number of testing examples to return. Defaults to 2000.
            audio_format (str): Audio format to use. Defaults to "wav".
        """
        self.save_hyperparameters()

    def setup(self, stage=None):
        self.train_dataset = StyleTransferTarfileDataset(
            self.hparams.train_tarfiles,
            self.hparams.length,
            self.hparams.num_train_examples,
            self.hparams.audio_format,
            self.hparams.input_only,
        )

        self.val_dataset = StyleTransferTarfileDataset(
            self.hparams.val_tarfiles,
            self.hparams.length,
            self.hparams.num_val_examples,
            self.hparams.audio_format,
            self.hparams.input_only,
        )

        if self.hparams.test_tarfiles is not None:
            self.test_dataset = StyleTransferTarfileDataset(
                self.hparams.test_tarfiles,
                self.hparams.length,
                self.hparams.num_test_examples,
                self.hparams.audio_format,
                self.hparams.input_only,
            )

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            drop_last=True,
            pin_memory=self.hparams.pin_memory,
            persistent_workers=self.hparams.persistent_workers,
            worker_init_fn=tarfile_worker_init_fn,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            persistent_workers=self.hparams.persistent_workers,
            worker_init_fn=tarfile_worker_init_fn,
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            persistent_workers=self.hparams.persistent_workers,
            worker_init_fn=tarfile_worker_init_fn,
        )


# ------------------------------ File based dataset --------------------------------


class StyleTransferDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root_dirs: List[str],
        length: int,
    ):
        super().__init__()
        self.root_dirs = root_dirs
        self.length = length

        self.examples = []
        for root_dir in root_dirs:
            # get all examples in the root dir
            print("Searching for examples in", root_dir)
            examples = glob.glob(os.path.join(root_dir, "*"))
            self.examples.extend(examples)

        # for testing use only the first 100 examples
        # self.examples = self.examples[:100]
        # repeat those examples 10 times
        # self.examples = self.examples * 100

        print(f"Found {len(self.examples)} examples.")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]

        input_filepath = os.path.join(example, "input.wav")
        output_filepath = os.path.join(example, "output.wav")
        params_filepath = os.path.join(example, "params.json")

        # check the length of the audio
        md = torchaudio.info(input_filepath, backend="soundfile")
        num_frames = md.num_frames

        # pick random start frame
        start_frame = torch.randint(0, num_frames - self.length, (1,)).item()
        # start_frame = 0  # for now, just use the first frame

        # load the audio
        input_audio, sr = torchaudio.load(
            input_filepath,
            frame_offset=start_frame,
            num_frames=self.length,
            backend="soundfile",
        )
        output_audio, sr = torchaudio.load(
            output_filepath,
            frame_offset=start_frame,
            num_frames=self.length,
            backend="soundfile",
        )

        # load the params
        with open(params_filepath, "r") as fp:
            params = json.load(fp)

        # convert params to tensor
        params = torch.tensor([val for val in params.values()], dtype=torch.float32)

        return input_audio, output_audio, params


class StyleTransferDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_root_dirs: List[str],
        val_root_dirs: List[str],
        batch_size: int,
        num_workers: int,
        length: int = 262144,
        test_root_dirs: List[str] = None,
        pin_memory: bool = False,
        persistent_workers: bool = False,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

    def setup(self, stage=None):
        self.train_dataset = StyleTransferDataset(
            self.hparams.train_root_dirs,
            self.hparams.length,
        )
        self.val_dataset = StyleTransferDataset(
            self.hparams.val_root_dirs,
            self.hparams.length,
        )
        if self.hparams.test_root_dirs is not None:
            self.test_dataset = StyleTransferDataset(
                self.hparams.test_root_dirs,
                self.hparams.length,
            )

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            drop_last=True,
            pin_memory=self.hparams.pin_memory,
            persistent_workers=self.hparams.persistent_workers,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            persistent_workers=self.hparams.persistent_workers,
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            persistent_workers=self.hparams.persistent_workers,
        )
