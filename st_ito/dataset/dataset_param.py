import os
import io
import sys
import glob
import json
import math
import torch
import tarfile
import torchaudio
import numpy as np
import pytorch_lightning as pl

from tqdm import tqdm
from typing import List
from torch.utils.data import Dataset, IterableDataset


def tarfile_worker_init_fn(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset  # the dataset copy in this worker process

    # if there is more than one tar file in the dataset, then we need to
    # select one tar file for this worker
    print(f"Worker {worker_id} creating new tar file handles.")

    # reopen the tarfile handles for this worker
    dataset.tar_handles = []
    for tar_file in dataset.tar_files:
        tar_handle = tarfile.open(tar_file, "r:")
        tar_handle.next()  # skip the first root directory
        tar_handle.next()  # skip the first example directory
        dataset.tar_handles.append(tar_handle)


def torchaudio_decode(b):
    x, sr = torchaudio.load(b, format="FLAC")
    return x


class PluginTarfileDataset(IterableDataset):
    def __init__(
        self,
        tar_files: List[str],
        plugin_json: str,
        length: int = 262144,
        stereo: bool = True,
        num_examples: int = 20000,
        label_files: List[str] = None,
    ) -> None:
        super().__init__()
        """ Interable dataset that returns a single input/output pair from a tar file. 

        When multiple tar files are provided, we open a file handle to each tar file. Then on each
        iteration we select a tar file at random and then select a random example from that tar file.
        We repeat this until each tar file has been read once. Then we start over again.

        Args:
            tar_files (List[str]): List of tar files to read from.
            plugin_json (str): Path to json file containing plugin instance details.
            length (int): Length of audio to return. Defaults to 262144.
            stereo (bool): whether to return stereo audio. Defaults to True.
                When set to False, the mean of the two channels is returned.
            num_examples (int): Number of examples to return. Defaults to 20000.
            label_files (List[str]): List of label files to use for classification. Defaults to None.
                Note: they must be int he same order as the associated tar_files.

        """
        self.tar_files = tar_files
        self.length = length
        self.stereo = stereo
        self.num_examples = num_examples
        self.label_files = label_files

        # first open plugin info
        # this is used for indexing into the plugin list
        with open(plugin_json, "r") as f:
            self.plugins = json.load(f)

        # get the tar file handles
        self.tar_handles = []
        for tar_file in tar_files:
            tar_handle = tarfile.open(tar_file, "r:")
            tar_handle.next()  # skip the first root directory
            tar_handle.next()  # skip the first example directory
            self.tar_handles.append(tar_handle)

        # check if we have label files
        if label_files is not None:
            assert len(label_files) == len(tar_files)  # must be the same length
            self.labels = []
            for label_file in label_files:
                results = torch.load(label_file)
                self.labels.append(results)

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
            input_file = self.tar_handles[tar_index].extractfile(members["input.flac"])
            input = torchaudio_decode(input_file)

            parent_dir = os.path.dirname(members["input.flac"].name).split("/")[-1]

            # select at random from the processed files
            processed_members = [
                x for x in members.keys() if x != "input.flac" and ".json" not in x
            ]
            processed_member = np.random.choice(processed_members)
            processed_fp = self.tar_handles[tar_index].extractfile(
                members[processed_member]
            )
            output = torchaudio_decode(processed_fp)

            # load the json file
            json_member = members[processed_member.replace(".flac", ".json")]
            json_fp = self.tar_handles[tar_index].extractfile(json_member)
            details = json.load(json_fp)

            # if using labels, get the classification info
            if self.label_files is not None:
                print(
                    tar_index, members["input.flac"].name, self.label_files[tar_index]
                )
                label_info = self.labels[tar_index][parent_dir]
                predicted_class_id = label_info["predicted_class_ids"]
                predicted_label = label_info["predicted_label"]
                logits = label_info["logits"].squeeze()
            else:
                logits = torch.empty(0)

            # get the index of the plugin instance for classification
            instance_index = self.plugins[details["instance"]]
            instance_index -= 1  # zero index
            instance_index = torch.tensor(instance_index).long()
            preset_index = torch.tensor(details["preset"]).long()

            # fit the proper length
            if input.shape[-1] < self.length:
                input = torch.cat(
                    [
                        input,
                        torch.zeros([input.shape[0], self.length - input.shape[-1]]),
                    ],
                    dim=-1,
                )
            if output.shape[-1] < self.length:
                output = torch.cat(
                    [
                        output,
                        torch.zeros([output.shape[0], self.length - output.shape[-1]]),
                    ],
                    dim=-1,
                )

            # take a different random crop from each file
            if input.shape[-1] > self.length:
                start = torch.randint(0, (input.shape[-1] - self.length), [1])
                input = input[..., start : start + self.length]

            # take a random crop
            if output.shape[-1] > self.length:
                start = torch.randint(0, (output.shape[-1] - self.length), [1])
                output = output[..., start : start + self.length]

            # check if stereo
            if self.stereo:
                # convert mono to stereo
                if input.shape[0] == 1:
                    input = input.repeat(2, 1)
                if output.shape[0] == 1:
                    output = output.repeat(2, 1)
            else:
                # convert stereo to mono
                if input.shape[0] == 2:
                    input = torch.mean(input, dim=0, keepdim=True)
                if output.shape[0] == 2:
                    output = torch.mean(output, dim=0, keepdim=True)

            # randomize the gain of input and output
            input /= torch.max(torch.abs(input)).clamp(1e-8)
            output /= torch.max(torch.abs(output)).clamp(1e-8)

            # from 0 to -32 dB
            gain_db = torch.rand(1) * -32.0
            input *= 10.0 ** (gain_db / 20.0)

            # from 0 to -32 dB
            gain_db = torch.rand(1) * -32.0
            output *= 10.0 ** (gain_db / 20.0)

            # swap left and right channels at random
            if torch.rand(1) > 0.5:
                input = input.flip(0)
                output = output.flip(0)

            yield input, output, instance_index, preset_index, tar_index, logits

    def __iter__(self):
        return iter(self.next_example())


class PluginTarfileDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_tarfiles: List[str],
        val_tarfiles: List[str],
        plugin_json: str,
        batch_size: int,
        num_workers: int,
        length: int = 262144,
        stereo: bool = True,
        test_tarfiles: List[str] = None,
        train_labels: List[str] = None,
        val_labels: List[str] = None,
        test_labels: List[str] = None,
        pin_memory: bool = False,
        persistent_workers: bool = True,
        num_train_examples: int = 20000,
        num_val_examples: int = 2000,
        num_test_examples: int = 2000,
    ) -> None:
        super().__init__()
        """ Pytorch Lightning DataModule for loading data from tar files.
        
        Args:
            train_tarfiles (List[str]): List of tar files to use for training.
            val_tarfiles (List[str]): List of tar files to use for validation.
            plugin_json (str): Path to json file containing plugin instance details.
            batch_size (int): Batch size.
            num_workers (int): Number of workers.
            length (int): Length of audio to return. Defaults to 262144.
            stereo (bool): whether to return stereo audio. Defaults to True.
                When set to False, the mean of the two channels is returned.
            test_tarfiles (List[str]): List of tar files to use for testing.
            train_labels (List[str]): List of label files to use for training. Defaults to None.
            val_labels (List[str]): List of label files to use for validation. Defaults to None.
            test_labels (List[str]): List of label files to use for testing. Defaults to None.
            pin_memory (bool): Whether to pin memory. Defaults to False.
            persistent_workers (bool): Whether to use persistent workers. Defaults to True.
            num_train_examples (int): Number of training examples to return. Defaults to 20000.
            num_val_examples (int): Number of validation examples to return. Defaults to 2000.
            num_test_examples (int): Number of testing examples to return. Defaults to 2000.
        """
        self.save_hyperparameters()

    def setup(self, stage=None):
        self.train_dataset = PluginTarfileDataset(
            self.hparams.train_tarfiles,
            self.hparams.plugin_json,
            self.hparams.length,
            self.hparams.stereo,
            self.hparams.num_train_examples,
            label_files=self.hparams.train_labels,
        )

        self.val_dataset = PluginTarfileDataset(
            self.hparams.val_tarfiles,
            self.hparams.plugin_json,
            self.hparams.length,
            self.hparams.stereo,
            self.hparams.num_val_examples,
            label_files=self.hparams.val_labels,
        )

        if self.hparams.test_tarfiles is not None:
            self.test_dataset = PluginTarfileDataset(
                self.hparams.test_tarfiles,
                self.hparams.plugin_json,
                self.hparams.length,
                self.hparams.stereo,
                self.hparams.num_test_examples,
                label_files=self.hparams.test_labels,
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


class PluginAudioFileWebDataset(Dataset):
    def __init__(
        self,
        root_dirs: str,
        plugin_json: str,
        length: int = 524288,
        stereo: bool = True,
    ) -> None:
        """

        Args:
            root_dirs (List[str]): List of directories containing dataset files.
            plugin_json: Path to json file containing plugin instance details.
            length (int): Length of audio to return. Defaults to 524288.
            stereo (bool): whether to return stereo audio. Defaults to True.
                When set to False, the mean of the two channels is returned.

        """
        super().__init__()
        self.root_dirs = root_dirs
        self.stereo = stereo
        self.length = length

        self.examples = []

        # find all segment directories
        for root_dir in root_dirs:
            examples = []
            # first, check if there is an examples.json file
            examples_json = os.path.join(root_dir, "examples.json")
            if os.path.exists(examples_json):
                print(f"Found {examples_json}.")
                with open(examples_json, "r") as f:
                    examples = json.load(f)
            else:
                print(
                    f"No examples.json file found. Crawling to find all examples in {root_dir}..."
                )
                segment_dirs = glob.glob(os.path.join(root_dir, "*"))
                segment_dirs = sorted([x for x in segment_dirs if os.path.isdir(x)])

                for segment_dir in tqdm(segment_dirs):
                    # find the input file
                    input_file = os.path.join(segment_dir, "input.flac")
                    if not os.path.exists(input_file):
                        continue

                    # find all processed audio files
                    processed_files = glob.glob(os.path.join(segment_dir, "*.flac"))

                    for processed_file in processed_files:
                        if processed_file == input_file:
                            continue
                        # find the appropriate json file
                        json_file = processed_file.replace(".flac", ".json")

                        # check if json file exists
                        if not os.path.exists(json_file):
                            print(f"Could not find {json_file}.")
                            continue

                        # test open json file
                        # try:
                        #    with open(json_file, "r") as f:
                        #        details = json.load(f)
                        # except Exception:
                        #    print(f"Error opening {json_file}.")
                        #    continue

                        example = (input_file, processed_file, json_file)
                        examples.append(example)

            self.examples += examples
            print(
                f"Found {len(examples)} examples in {root_dir}. Total: {len(self.examples)}"
            )

        with open(plugin_json, "r") as f:
            self.plugins = json.load(f)

        # for testing, load one example into memory
        # input_file, processed_file, json_file = self.examples[0]

        # open the json file
        # with open(json_file, "r") as f:
        #    details = json.load(f)

        # input, sr = torchaudio.load(input_file, backend="soundfile")
        # output, sr = torchaudio.load(processed_file, backend="soundfile")

        # self.input = input
        # self.output = output
        # self.details = details

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        """Get a single example from the dataset."""

        if True:
            input_file, processed_file, json_file = self.examples[idx]

            # open the json file
            with open(json_file, "r") as f:
                details = json.load(f)

            input, sr = torchaudio.load(input_file, backend="soundfile")
            output, sr = torchaudio.load(processed_file, backend="soundfile")
        else:
            input = self.input
            output = self.output
            details = self.details

        # check length of input and output
        # if input.shape[-1] != output.shape[-1]:
        #    raise RuntimeError(
        #        f"Input and output shapes do not match: {input.shape} vs {output.shape}"
        #    )

        # fit the proper length
        if input.shape[-1] < self.length:
            input = torch.cat(
                [input, torch.zeros([input.shape[0], self.length - input.shape[-1]])],
                dim=-1,
            )
        if output.shape[-1] < self.length:
            output = torch.cat(
                [
                    output,
                    torch.zeros([output.shape[0], self.length - output.shape[-1]]),
                ],
                dim=-1,
            )

        # take a random crop
        start = torch.randint(
            0, (min(input.shape[-1], output.shape[-1]) - self.length), [1]
        )
        input = input[..., start : start + self.length]
        output = output[..., start : start + self.length]

        # note: now we take a crop from the same position in both files
        # however, it might be a good augmentation to take a different
        # random crop from each file

        # get the index of the plugin instance for classification
        instance_index = self.plugins[details["instance"]]
        instance_index -= 1  # zero index
        instance_index = torch.tensor(instance_index).long()
        preset_index = torch.tensor(details["preset"]).long()

        # check if stereo
        if self.stereo:
            # convert mono to stereo
            if input.shape[0] == 1:
                input = input.repeat(2, 1)
            if output.shape[0] == 1:
                output = output.repeat(2, 1)
        else:
            # convert stereo to mono
            if input.shape[0] == 2:
                input = torch.mean(input, dim=0, keepdim=True)
            if output.shape[0] == 2:
                output = torch.mean(output, dim=0, keepdim=True)

        # randomize the gain of input and output
        input /= torch.max(torch.abs(input)).clamp(1e-8)
        output /= torch.max(torch.abs(output)).clamp(1e-8)

        # from 0 to -32 dB
        gain_db = torch.rand(1) * -32.0
        input *= 10.0 ** (gain_db / 20.0)

        # from 0 to -32 dB
        gain_db = torch.rand(1) * -32.0
        output *= 10.0 ** (gain_db / 20.0)

        return input, output, instance_index, preset_index


class PluginAudioFileWebDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_root_dirs: List[str],
        val_root_dirs: List[str],
        plugin_json: str,
        batch_size: int,
        num_workers: int,
        length: int = 262144,
        test_root_dirs: List[str] = None,
        pin_memory: bool = False,
        persistent_workers: bool = False,
        use_tarfile: bool = False,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

    def setup(self, stage=None):
        if not self.hparams.use_tarfile:
            self.train_dataset = PluginAudioFileWebDataset(
                self.hparams.train_root_dirs,
                self.hparams.plugin_json,
                self.hparams.length,
            )

            self.val_dataset = PluginAudioFileWebDataset(
                self.hparams.val_root_dirs,
                self.hparams.plugin_json,
                self.hparams.length,
            )

            if self.hparams.test_root_dirs is not None:
                self.test_dataset = PluginAudioFileWebDataset(
                    self.hparams.test_root_dirs,
                    self.hparams.plugin_json,
                    self.hparams.length,
                )
        else:
            raise NotImplementedError

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            drop_last=True,
            pin_memory=self.hparams.pin_memory,
            persistent_workers=self.hparams.persistent_workers,
            prefetch_factor=8,
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
