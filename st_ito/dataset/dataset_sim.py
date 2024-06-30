import os
import glob
import torch
import torchaudio
import pedalboard
import numpy as np
import pytorch_lightning as pl

from tqdm import tqdm
from typing import List, Optional, Tuple, Iterable
from st_ito.effects import (
    BasicChorus,
    BasicCompressor,
    BasicDelay,
    BasicDistortion,
    BasicReverb,
)


def worker_init_fn(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    # re-init the vst_plugins for each worker
    vst_plugins = load_plugins(worker_info.dataset.vst_dir)
    worker_info.dataset.vst_plugins = vst_plugins


def find_files_with_glob(
    directory: str,
):
    """Find all files in the given directory with specific pattern."""
    audio_filepaths = []
    for ext in ["*.wav", "*.mp3", "*.ogg", "*.flac"]:
        audio_filepaths += glob.glob(os.path.join(directory, "**", ext), recursive=True)

    return audio_filepaths


def find_files_with_extensions(
    directory: str,
    extensions: Tuple[str] = (".wav", ".mp3", ".ogg", ".flac"),
):
    """Find all files in the given directory with specific extensions."""
    for entry in os.scandir(directory):
        if entry.is_file() and entry.name.lower().endswith(extensions):
            yield entry.path
        elif entry.is_dir():
            yield from find_files_with_extensions(entry.path, extensions)


def find_files_lazy(
    directory: str,
    extensions: Tuple[str] = (".wav", ".mp3", ".ogg", ".flac"),
):
    """Lazily find files with specific extensions in the given directory."""
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(extensions):
                yield os.path.join(root, file)


def get_nonsilent_region(
    audio_filepath: str,
    total_num_frames: int,
    length: int,
    threshold: float = 1e-8,
    max_tires: int = 100,
):
    silent = True
    tries = 0
    while silent:

        if total_num_frames > length:
            # load a random segment of the audio file
            audio, sr = torchaudio.load(
                audio_filepath,
                num_frames=length,
                frame_offset=np.random.randint(0, total_num_frames - length),
                backend="soundfile",
            )

        else:
            # load the entire audio file
            audio, sr = torchaudio.load(
                audio_filepath,
                num_frames=total_num_frames,
                backend="soundfile",
            )

            # repeat the audio to reach the desired length
            audio = torch.cat([audio] * (length // total_num_frames + 1), axis=1)
            audio = audio[:, :length]

        # check for silence
        energy = torch.mean(audio**2)
        print(energy)
        if energy > threshold:
            silent = False
        else:
            silent = False  # TODO: remove this line

        tries += 1

        if tries > max_tires:
            raise ValueError(
                f"Could not find a non-silent region in the audio: {audio_filepath}."
            )

    return audio, sr


def random_gain(audio: torch.Tensor):
    # first peak normalize
    audio = audio / torch.max(torch.abs(audio))
    # then apply random gain
    gain_db = np.random.uniform(-18, 0)
    gain_lin = 10 ** (gain_db / 20)
    audio = audio * gain_lin
    return audio


def conform_channels(audio_input: torch.Tensor, num_channels: int):
    if audio_input.shape[0] == 1 and num_channels == 2:
        audio_input = torch.cat((audio_input, audio_input), axis=0)
    if audio_input.shape[0] == 2 and num_channels == 1:
        audio_input = audio_input[0:1, :]
    return audio_input


def load_plugins(vst_dir: str, sample_rate: int = 48000):
    # find all vst3 files
    vst_filepaths = []
    for ext in ["*.vst3"]:
        vst_filepaths += glob.glob(os.path.join(vst_dir, ext), recursive=True)

    # vst_filepaths = vst_filepaths[:2]

    vst_filepaths = sorted(vst_filepaths)
    print(f"Found {len(vst_filepaths)} VST3 files in {vst_dir}.")

    vst_plugins = {}
    print("Loading plugins...")
    for vst_filepath in tqdm(vst_filepaths):
        instance_name = os.path.basename(vst_filepath).replace(".vst3", "")
        print(f"Loading {instance_name}...")
        # try to load each plugin
        try:
            plugin = pedalboard.load_plugin(vst_filepath)
        except Exception:
            print(f"Error loading {vst_filepath}.")

        # try to pass audio through the plugin
        failed = True
        try:
            num_channels = 2
            audio_input = np.random.randn(2, sample_rate * 5)
            audio_output = plugin(audio_input, sample_rate)
            failed = False
        except Exception as e:
            num_channels = 1
            print(e)
            print(f"Error processing audio with 2 channels.")

        try:
            audio_input = np.random.randn(1, sample_rate * 5)
            audio_output = plugin(audio_input, sample_rate)
            failed = False
        except Exception as e:
            num_channels = 2
            print(e)
            print(f"Failed processing audio with 1 channel.")

        if failed:
            continue

        print(f"Loaded {os.path.basename(vst_filepath)} successfully.")

        vst_plugins[instance_name] = {
            "filepath": vst_filepath,
            "instance": plugin,
            "num_channels": num_channels,
        }

    for idx, (instance_name, vst_plugin) in enumerate(vst_plugins.items()):
        print(idx + 1, instance_name)

    return vst_plugins


class PluginSimilarityDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        audio_filepaths: List[str],
        vst_dir: str,
        length: int = 262144,
        num_examples_per_epoch: int = 10000,
    ):
        self.audio_filepaths = audio_filepaths
        self.vst_dir = vst_dir
        self.length = length
        self.num_examples_per_epoch = num_examples_per_epoch
        self.plugins = load_plugins(self.vst_dir)

    def __len__(self):
        return self.num_examples_per_epoch

    def __getitem__(self, idx: int):
        # sample the first filepath
        filepath_a = np.random.choice(self.audio_filepaths)
        # sample the second filepath
        filepath_b = np.random.choice(self.audio_filepaths)

        # load the audio files
        num_frames_a = torchaudio.info(filepath_a).num_frames
        audio_a, sr_a = get_nonsilent_region(filepath_a, num_frames_a, self.length)

        num_frames_b = torchaudio.info(filepath_b).num_frames
        audio_b, sr_b = get_nonsilent_region(filepath_b, num_frames_b, self.length)

        # sample randomly a plugin from the dictionary
        plugin_name = np.random.choice(list(self.plugins.keys()))
        plugin = self.plugins[plugin_name]

        # randomize the parameters of the plugin
        for name, parameter in plugin["instance"].parameters.items():
            if "bypass" in name.lower():
                parameter.raw_value = 0
            elif name in ["buffer_size_frames", "sample_rate_frames"]:
                pass
            else:  # set the parameter to the preset value
                parameter.raw_value = np.random.rand()

        audio_a = conform_channels(audio_a, plugin["num_channels"])
        audio_b = conform_channels(audio_b, plugin["num_channels"])

        # apply the effect to the two audio files
        audio_a_out = torch.from_numpy(
            plugin["instance"].process(audio_a.numpy(), sr_a)
        )
        audio_b_out = torch.from_numpy(
            plugin["instance"].process(audio_b.numpy(), sr_b)
        )

        # peak normalize and then apply random gain
        audio_a = random_gain(audio_a)
        audio_b = random_gain(audio_b)
        audio_a_out = random_gain(audio_a_out)
        audio_b_out = random_gain(audio_b_out)

        # ensure all outputs are stereo
        audio_a = conform_channels(audio_a, 2)
        audio_b = conform_channels(audio_b, 2)
        audio_a_out = conform_channels(audio_a_out, 2)
        audio_b_out = conform_channels(audio_b_out, 2)

        return audio_a, audio_b, audio_a_out, audio_b_out


class PluginSimilarityDataModule(pl.LightningDataModule):
    def __init__(
        self,
        audio_dirs: List[str],
        vst_dir: str,
        length: int = 262144,
        batch_size: int = 32,
        num_train_examples: int = 10000,
        num_val_examples: int = 1000,
        num_workers: int = 4,
        persistent_workers: bool = True,
        pin_memory: bool = True,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

    def prepare_data(self):
        # download, split, etc...
        pass

    def setup(self, stage: Optional[str] = None):
        # find all audio files in the input directory based on file extensions
        audio_filepaths = []
        for audio_dir in self.hparams.audio_dirs:
            audio_filepaths += find_files_with_glob(audio_dir)
            print("Found", len(audio_filepaths), "audio files in", audio_dir)

        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            self.plugin_similarity_train = PluginSimilarityDataset(
                audio_filepaths,
                self.hparams.vst_dir,
                self.hparams.length,
                self.hparams.num_train_examples,
            )
            self.plugin_similarity_val = PluginSimilarityDataset(
                audio_filepaths,
                self.hparams.vst_dir,
                self.hparams.length,
                self.hparams.num_val_examples,
            )

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.plugin_similarity_train,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            persistent_workers=self.hparams.persistent_workers,
            pin_memory=self.hparams.pin_memory,
            # worker_init_fn=worker_init_fn,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.plugin_similarity_val,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            persistent_workers=self.hparams.persistent_workers,
            pin_memory=self.hparams.pin_memory,
            # worker_init_fn=worker_init_fn,
        )
