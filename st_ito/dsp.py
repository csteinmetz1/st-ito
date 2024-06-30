import torch
import numpy as np
import dasp_pytorch
from dasp_pytorch.functional import (
    simple_distortion,
    compressor,
    noise_shaped_reverberation,
)
import pyloudnorm as pyln


def apply_random_simple_distortion(
    x: torch.Tensor, sample_rate: float, use_gpu: bool = False
):
    drive_db = np.random.uniform(0, 32)
    drive_db_tensor = torch.tensor(drive_db, dtype=torch.float32)

    if use_gpu:
        drive_db_tensor = drive_db_tensor.cuda()
        x = x.cuda()

    y = simple_distortion(x, drive_db_tensor)
    return y, drive_db


def apply_random_reverb(x: torch.Tensor, sample_rate: float, use_gpu: bool = False):
    mix = np.random.uniform(0, 1.0)
    band_decays = torch.tensor(
        [0.6, 0.4, 0.4, 0.5, 0.2, 0.3, 0.3, 0.2, 0.1, 0.1, 0.2, 0.1]
    )
    band_gains = torch.tensor(
        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    )
    mix_tensor = torch.tensor(mix, dtype=torch.float32)

    if use_gpu:
        band_decays = band_decays.cuda()
        band_gains = band_gains.cuda()
        mix_tensor = mix_tensor.cuda()
        x = x.cuda()

    y = noise_shaped_reverberation(
        x, sample_rate, *band_gains, *band_decays, mix_tensor
    ).cpu()

    return y, mix


def apply_random_compressor(x: torch.Tensor, sample_rate: float, use_gpu: bool = False):
    threshold_db = np.random.uniform(-48, 0)
    threshold_db_tensor = torch.tensor(threshold_db, dtype=torch.float32)
    ratio = torch.tensor([4.0])
    attack_ms = torch.tensor([100.0])
    release_ms = torch.tensor([100.0])
    knee_db = torch.tensor([24.0])
    makeup_gain_db = torch.tensor([0.0])

    if use_gpu:
        threshold_db_tensor = threshold_db_tensor.cuda()
        ratio = ratio.cuda()
        attack_ms = attack_ms.cuda()
        release_ms = release_ms.cuda()
        knee_db = knee_db.cuda()
        makeup_gain_db = makeup_gain_db.cuda()
        x = x.cuda()

    y = compressor(
        x,
        sample_rate,
        threshold_db_tensor,
        ratio,
        attack_ms,
        release_ms,
        knee_db,
        makeup_gain_db,
    ).cpu()

    return y, threshold_db


def normalize_loudness(x: torch.Tensor, sr: float, target_lufs_db: float):
    meter = pyln.Meter(sr)
    x_lufs_db = meter.integrated_loudness(x.permute(1, 0).numpy())
    delta_lufs_db = torch.tensor([target_lufs_db - x_lufs_db]).float()
    gain_lin = 10.0 ** (delta_lufs_db.clamp(-120, 40.0) / 20.0)
    x = gain_lin * x
    return x
