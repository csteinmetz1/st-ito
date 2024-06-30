# Adapted from https://github.com/aim-qmul/sdx23-aimless/blob/master/data/augment.py
import torch
import random
import pedalboard
import torchaudio
import numpy as np
import scipy.stats
import scipy.signal
import dasp_pytorch
import pyloudnorm as pyln
from typing import List, Tuple

from pedalboard import (
    Pedalboard,
    Gain,
    Chorus,
    Reverb,
    Compressor,
    Phaser,
    Delay,
    Distortion,
    Limiter,
)

# from dasp_pytorch.functional import single_band_eq


__all__ = [
    "RandomSwapLR",
    "RandomGain",
    "RandomFlipPhase",
    "LimitAug",
    "CPUBase",
    "RandomParametricEQ",
    "RandomStereoWidener",
    "RandomVolumeAutomation",
    "RandomPedalboardCompressor",
    "RandomPedalboardDelay",
    "RandomPedalboardChorus",
    "RandomPedalboardDistortion",
    "RandomPedalboardPhaser",
    "RandomPedalboardReverb",
    "RandomPedalboardLimiter",
    "RandomSoxReverb",
    "LoudnessNormalize",
    "RandomPan",
    "Mono2Stereo",
]


def db2linear(x):
    return 10 ** (x / 20)


def loguniform(low=0, high=1):
    return scipy.stats.loguniform.rvs(low, high)


def rand(low=0, high=1):
    return (random.random() * (high - low)) + low


def randint(low=0, high=1):
    return random.randint(low, high)


def normalize_param(x: torch.Tensor, low: float, high: float):
    return (x - low) / (high - low)


# effect functions (expect audio array and return audio array, parameters, and normalized parameters)
def pedalboard_distortion(x: np.ndarray, parameters: List[float], sample_rate: int):
    drive_db = parameters[0]
    board = pedalboard.Pedalboard([pedalboard.Distortion(drive_db=drive_db)])
    output = board(x, sample_rate)
    return output


def pedalboard_reverb(x: np.ndarray, parameters: List[float], sample_rate: int):
    room_size = parameters[0]
    damping = parameters[1]
    wet_level = parameters[2]
    dry_level = 1 - wet_level
    board = pedalboard.Pedalboard(
        [
            pedalboard.Reverb(
                room_size=room_size,
                damping=damping,
                wet_level=wet_level,
                dry_level=dry_level,
            )
        ]
    )
    output = board(x, sample_rate)
    return output


def flying_reverb(x: np.ndarray, parameters: List[float], sample_rate: int):
    board = pedalboard.load_plugin("vst/FlyingReverb.vst3")
    # Print all of the parameter names:
    for parameter, value in board.parameters.items():
        print(parameter, value)


EFFECT_FUNCS = {
    "pedalboard_distortion": {
        "func": pedalboard_distortion,
        "class": "distortion",
        "instance": "pedalboard_distortion",
        "parameters": {
            "drive_db": {"min": 0, "max": 24, "scale": "linear"},
        },
    },
    "pedalboard_reverb": {
        "func": pedalboard_reverb,
        "class": "reverb",
        "instance": "pedalboard_reverb",
        "parameters": {
            "room_size": {"min": 0, "max": 1, "scale": "linear"},
            "damping": {"min": 0, "max": 1, "scale": "linear"},
            "wetdry_mix": {"min": 0, "max": 1, "scale": "linear"},
        },
    },
}


def apply_random_pedalboard_distortion(
    x: torch.Tensor,
    sample_rate: float,
    min_drive_db: float = 0.0,
    max_drive_db: float = 48.0,
):
    board = Pedalboard()
    drive_db = torch.rand(1) * (max_drive_db - min_drive_db) + min_drive_db
    board.append(Distortion(drive_db=drive_db))

    params = {"drive_db": normalize_param(drive_db, min_drive_db, max_drive_db)}

    return torch.from_numpy(board(x.numpy(), sample_rate)), params


def apply_random_pedalboard_delay(
    x: torch.Tensor,
    sample_rate: float,
    min_delay_seconds: float = 0.01,
    max_delay_seconds: float = 1.0,
    min_feedback: float = 0.05,
    max_feedback: float = 0.6,
    min_mix: float = 0.0,
    max_mix: float = 1.0,
):
    board = Pedalboard()
    delay_seconds = loguniform(min_delay_seconds, max_delay_seconds)
    feedback = rand(min_feedback, max_feedback)
    mix = rand(min_mix, max_mix)
    board.append(Delay(delay_seconds=delay_seconds, feedback=feedback, mix=mix))
    y = torch.from_numpy(board(x.numpy(), sample_rate))

    params = [
        normalize_param(delay_seconds, min_delay_seconds, max_delay_seconds),
        normalize_param(feedback, min_feedback, max_feedback),
        normalize_param(mix, min_mix, max_mix),
    ]

    params = torch.tensor(params, dtype=torch.float32)

    return y, params


def apply_random_pedalboard_chorus(
    x: torch.Tensor,
    sample_rate: float,
    min_rate_hz: float = 0.25,
    max_rate_hz: float = 4.0,
    min_depth: float = 0.0,
    max_depth: float = 0.6,
    min_centre_delay_ms: float = 5.0,
    max_centre_delay_ms: float = 10.0,
    min_feedback: float = 0.1,
    max_feedback: float = 0.6,
    min_mix: float = 0.0,
    max_mix: float = 1.0,
):
    board = Pedalboard()
    rate_hz = rand(min_rate_hz, max_rate_hz)
    depth = rand(min_depth, max_depth)
    centre_delay_ms = rand(min_centre_delay_ms, max_centre_delay_ms)
    feedback = rand(min_feedback, max_feedback)
    mix = rand(min_mix, max_mix)
    board.append(
        Chorus(
            rate_hz=rate_hz,
            depth=depth,
            centre_delay_ms=centre_delay_ms,
            feedback=feedback,
            mix=mix,
        )
    )
    # process audio using the pedalboard
    y = torch.from_numpy(board(x.numpy(), sample_rate))

    params = [
        normalize_param(rate_hz, min_rate_hz, max_rate_hz),
        normalize_param(depth, min_depth, max_depth),
        normalize_param(centre_delay_ms, min_centre_delay_ms, max_centre_delay_ms),
        normalize_param(feedback, min_feedback, max_feedback),
        normalize_param(mix, min_mix, max_mix),
    ]

    params = torch.tensor(params, dtype=torch.float32)

    return y, params


def apply_random_pedalboard_reverb(
    x: torch.Tensor,
    sample_rate: float,
    min_room_size: float = 0.0,
    max_room_size: float = 1.0,
    min_damping: float = 0.0,
    max_damping: float = 1.0,
    min_wet_dry: float = 0.0,
    max_wet_dry: float = 1.0,
    min_width: float = 0.0,
    max_width: float = 1.0,
):
    board = Pedalboard()
    room_size = rand(min_room_size, max_room_size)
    damping = rand(min_damping, max_damping)
    wet_dry = rand(min_wet_dry, max_wet_dry)
    width = rand(min_width, max_width)

    board.append(
        Reverb(
            room_size=room_size,
            damping=damping,
            wet_level=wet_dry,
            dry_level=(1 - wet_dry),
            width=width,
        )
    )

    y = torch.from_numpy(board(x.numpy(), sample_rate))

    params = [
        normalize_param(room_size, min_room_size, max_room_size),
        normalize_param(damping, min_damping, max_damping),
        normalize_param(wet_dry, min_wet_dry, max_wet_dry),
        normalize_param(width, min_width, max_width),
    ]

    params = torch.tensor(params, dtype=torch.float32)

    return y, params


def apply_random_pedalboard_compressor(
    x: torch.Tensor,
    sample_rate: float,
    min_threshold_db: float = -42.0,
    max_threshold_db: float = 0.0,
    min_ratio: float = 1.0,
    max_ratio: float = 10.0,
    min_attack_ms: float = 0.1,
    max_attack_ms: float = 100.0,
    min_release_ms: float = 10.0,
    max_release_ms: float = 1000.0,
):
    board = Pedalboard()
    threshold_db = torch.tensor([rand(min_threshold_db, max_threshold_db)])
    ratio = torch.tensor([rand(min_ratio, max_ratio)])
    attack_ms = torch.tensor([rand(min_attack_ms, max_attack_ms)])
    release_ms = torch.tensor([rand(min_release_ms, max_release_ms)])

    board.append(
        Compressor(
            threshold_db=threshold_db,
            ratio=ratio,
            attack_ms=attack_ms,
            release_ms=release_ms,
        )
    )

    y = torch.from_numpy(board(x.numpy(), sample_rate))

    params = {
        "threshold_db": normalize_param(
            threshold_db, min_threshold_db, max_threshold_db
        ).float(),
        "ratio": normalize_param(ratio, min_ratio, max_ratio).float(),
        "attack_ms": normalize_param(attack_ms, min_attack_ms, max_attack_ms).float(),
        "release_ms": normalize_param(
            release_ms, min_release_ms, max_release_ms
        ).float(),
    }

    # process audio using the pedalboard
    return y, params


def apply_random_equalizer(
    x: torch.Tensor,
    sample_rate: float,
    min_gain_db: float = -48.0,
    max_gain_db: float = 48.0,
    min_q_factor: float = 0.1,
    max_q_factor: float = 10.0,
    min_cutoff_freq: float = 100.0,
    max_cutoff_freq: float = 10000.0,
):
    gain_db = torch.tensor([rand(min_gain_db, max_gain_db)])
    cutoff_freq = torch.tensor([loguniform(min_cutoff_freq, max_cutoff_freq)])
    q_factor = torch.tensor([rand(min_q_factor, max_q_factor)])

    y = single_band_eq(
        x.unsqueeze(0),
        sample_rate,
        gain_db,
        cutoff_freq,
        q_factor,
    )

    params = {
        "gain_db": normalize_param(gain_db, min_gain_db, max_gain_db).float(),
        "cutoff_freq": normalize_param(
            cutoff_freq, min_cutoff_freq, max_cutoff_freq
        ).float(),
        "q_factor": normalize_param(q_factor, min_q_factor, max_q_factor).float(),
    }

    return y.squeeze(0), params


class BaseTransform:
    def __init__(self, p: float = 1.0):
        assert 0 <= p <= 1, "invalid probability value"
        self.p = p

    def __call__(self, inputs: Tuple[torch.Tensor, torch.Tensor]):
        """
        Args:
            inputs (Tuple[torch.Tensor, torch.Tensor]):
                x_a (torch.Tensor): (chs, seq_len)
                x_b (torch.Tensor): (chs, seq_len)
        Return:
            outputs (Tuple[torch.Tensor, torch.Tensor]):
                x_a_out (torch.Tensor): (chs, seq_len)
                x_b_out (torch.Tensor): (chs, seq_len)
        """
        x_a, x_b = inputs
        seed = random.randint(0, 2**32 - 1)  # Random seed from 0 to 2^32-1

        if random.random() < self.p:
            random.seed(seed)
            x_a_out = self._transform(x_a)

            random.seed(seed)  # reset seed to get same effect combination
            x_b_out = self._transform(x_b)
        else:
            x_a_out, x_b_out = x_a, x_b

        return (x_a_out, x_b_out)

    def _transform(self, stems: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class RandomSwapLR(BaseTransform):
    def __init__(self, p=0.5) -> None:
        super().__init__(p=p)

    def _transform(self, stems: torch.Tensor):
        return torch.flip(stems, [0])


class RandomGain(BaseTransform):
    def __init__(self, low=0.25, high=1.25, **kwargs) -> None:
        super().__init__(**kwargs)
        self.low = low
        self.high = high

    def _transform(self, stems):
        gain = random.random() * (self.high - self.low) + self.low
        return stems * gain


class RandomFlipPhase(RandomSwapLR):
    def __init__(self, p=0.5) -> None:
        super().__init__(p=p)

    def _transform(self, stems: torch.Tensor):
        return -stems


def biqaud(
    gain_db: float,
    cutoff_freq: float,
    q_factor: float,
    sample_rate: float,
    filter_type: str,
):
    """Use design parameters to generate coeffieicnets for a specific filter type.
    Args:
        gain_db (float): Shelving filter gain in dB.
        cutoff_freq (float): Cutoff frequency in Hz.
        q_factor (float): Q factor.
        sample_rate (float): Sample rate in Hz.
        filter_type (str): Filter type.
            One of "low_shelf", "high_shelf", or "peaking"
    Returns:
        b (np.ndarray): Numerator filter coefficients stored as [b0, b1, b2]
        a (np.ndarray): Denominator filter coefficients stored as [a0, a1, a2]
    """

    A = 10 ** (gain_db / 40.0)
    w0 = 2.0 * np.pi * (cutoff_freq / sample_rate)
    alpha = np.sin(w0) / (2.0 * q_factor)

    cos_w0 = np.cos(w0)
    sqrt_A = np.sqrt(A)

    if filter_type == "high_shelf":
        b0 = A * ((A + 1) + (A - 1) * cos_w0 + 2 * sqrt_A * alpha)
        b1 = -2 * A * ((A - 1) + (A + 1) * cos_w0)
        b2 = A * ((A + 1) + (A - 1) * cos_w0 - 2 * sqrt_A * alpha)
        a0 = (A + 1) - (A - 1) * cos_w0 + 2 * sqrt_A * alpha
        a1 = 2 * ((A - 1) - (A + 1) * cos_w0)
        a2 = (A + 1) - (A - 1) * cos_w0 - 2 * sqrt_A * alpha
    elif filter_type == "low_shelf":
        b0 = A * ((A + 1) - (A - 1) * cos_w0 + 2 * sqrt_A * alpha)
        b1 = 2 * A * ((A - 1) - (A + 1) * cos_w0)
        b2 = A * ((A + 1) - (A - 1) * cos_w0 - 2 * sqrt_A * alpha)
        a0 = (A + 1) + (A - 1) * cos_w0 + 2 * sqrt_A * alpha
        a1 = -2 * ((A - 1) + (A + 1) * cos_w0)
        a2 = (A + 1) + (A - 1) * cos_w0 - 2 * sqrt_A * alpha
    elif filter_type == "peaking":
        b0 = 1 + alpha * A
        b1 = -2 * cos_w0
        b2 = 1 - alpha * A
        a0 = 1 + alpha / A
        a1 = -2 * cos_w0
        a2 = 1 - alpha / A
    else:
        pass
        # raise ValueError(f"Invalid filter_type: {filter_type}.")

    b = np.array([b0, b1, b2]) / a0
    a = np.array([a0, a1, a2]) / a0

    return b, a


def parametric_eq(
    x: np.ndarray,
    sample_rate: float,
    low_shelf_gain_db: float = 0.0,
    low_shelf_cutoff_freq: float = 80.0,
    low_shelf_q_factor: float = 0.707,
    band_gains_db: List[float] = [0.0],
    band_cutoff_freqs: List[float] = [300.0],
    band_q_factors: List[float] = [0.707],
    high_shelf_gain_db: float = 0.0,
    high_shelf_cutoff_freq: float = 1000.0,
    high_shelf_q_factor: float = 0.707,
    dtype=np.float32,
):
    """Multiband parametric EQ.

    Low-shelf -> Band 1 -> ... -> Band N -> High-shelf

    Args:

    """
    assert (
        len(band_gains_db) == len(band_cutoff_freqs) == len(band_q_factors)
    )  # must define for all bands

    # -------- apply low-shelf filter --------
    b, a = biqaud(
        low_shelf_gain_db,
        low_shelf_cutoff_freq,
        low_shelf_q_factor,
        sample_rate,
        "low_shelf",
    )
    x = scipy.signal.lfilter(b, a, x)

    # -------- apply peaking filters --------
    for gain_db, cutoff_freq, q_factor in zip(
        band_gains_db, band_cutoff_freqs, band_q_factors
    ):
        b, a = biqaud(
            gain_db,
            cutoff_freq,
            q_factor,
            sample_rate,
            "peaking",
        )
        x = scipy.signal.lfilter(b, a, x)

    # -------- apply high-shelf filter --------
    b, a = biqaud(
        high_shelf_gain_db,
        high_shelf_cutoff_freq,
        high_shelf_q_factor,
        sample_rate,
        "high_shelf",
    )
    sos5 = np.concatenate((b, a))
    x = scipy.signal.lfilter(b, a, x)

    return x.astype(dtype)


# ------------- Differentiable audio effects ----------------------------


def denormalize(p: torch.Tensor, min_val: float, max_val: float):
    """When p is on (0,1) restore the original range of the parameter values.

    Args:
        p (torch.Tensor): [bs, num_params]
        min_val (float): minimum value of the parameter
        max_val (float): maximum value of the parameter

    Returns:
        torch.Tensor: [bs, num_params]
    """
    return p * (max_val - min_val) + min_val


def apply_gain(audio: torch.Tensor, params: torch.Tensor, sample_rate: int):
    bs, chs, seq_len = audio.shape
    bs, num_params = params.shape

    assert num_params == 1, "Number of parameters must be 1"

    gain_db = denormalize(params[:, 0], -48, 48)

    audio = dasp_pytorch.gain(audio, sample_rate, gain_db)

    return audio


def apply_distortion(audio: torch.Tensor, params: torch.Tensor, sample_rate: int):
    bs, chs, seq_len = audio.shape
    bs, num_params = params.shape

    assert num_params == 1, "Number of parameters must be 1"

    gain_db = denormalize(params[:, 0], 0, 48)

    audio = dasp_pytorch.distortion(audio, sample_rate, gain_db)

    return audio


def apply_reverb(audio: torch.Tensor, params: torch.Tensor, sample_rate: int):
    bs, chs, seq_len = audio.shape
    bs, num_params = params.shape

    assert num_params == 25, "Number of parameters must be 25"

    band0_gain = params[:, 0]
    band1_gain = params[:, 1]
    band2_gain = params[:, 2]
    band3_gain = params[:, 3]
    band4_gain = params[:, 4]
    band5_gain = params[:, 5]
    band6_gain = params[:, 6]
    band7_gain = params[:, 7]
    band8_gain = params[:, 8]
    band9_gain = params[:, 9]
    band10_gain = params[:, 10]
    band11_gain = params[:, 11]
    band0_decay = params[:, 12]
    band1_decay = params[:, 13]
    band2_decay = params[:, 14]
    band3_decay = params[:, 15]
    band4_decay = params[:, 16]
    band5_decay = params[:, 17]
    band6_decay = params[:, 18]
    band7_decay = params[:, 19]
    band8_decay = params[:, 20]
    band9_decay = params[:, 21]
    band10_decay = params[:, 22]
    band11_decay = params[:, 23]
    mix = params[:, 24]

    audio = dasp_pytorch.noise_shaped_reverberation(
        audio,
        sample_rate,
        band0_gain,
        band1_gain,
        band2_gain,
        band3_gain,
        band4_gain,
        band5_gain,
        band6_gain,
        band7_gain,
        band8_gain,
        band9_gain,
        band10_gain,
        band11_gain,
        band0_decay,
        band1_decay,
        band2_decay,
        band3_decay,
        band4_decay,
        band5_decay,
        band6_decay,
        band7_decay,
        band8_decay,
        band9_decay,
        band10_decay,
        band11_decay,
        mix,
    )

    return audio


def apply_compressor(audio: torch.Tensor, params: torch.Tensor, sample_rate: int):
    bs, chs, seq_len = audio.shape
    bs, num_params = params.shape

    assert num_params == 6, "Number of parameters must be 6"

    threshold_db = denormalize(params[:, 0], -60, 0)
    ratio = denormalize(params[:, 1], 1, 20)
    attack_ms = denormalize(params[:, 2], 0.1, 250)
    release_ms = denormalize(params[:, 3], 10, 2000)
    knee_db = denormalize(params[:, 4], 1.0, 24.0)
    makeup_gain_db = denormalize(params[:, 5], 0, 24.0)

    audio = dasp_pytorch.compressor(
        audio,
        sample_rate,
        threshold_db,
        ratio,
        attack_ms,
        release_ms,
        knee_db,
        makeup_gain_db,
        lookahead_samples=512,
    )

    return audio


def apply_parametric_eq(audio: torch.Tensor, params: torch.Tensor, sample_rate: int):
    bs, chs, seq_len = audio.shape
    bs, num_params = params.shape

    assert sample_rate >= 44100, "Sample rate must be at least 44100 Hz"
    assert num_params == 18, "Number of parameters must be 18"

    # convert params to correct range
    low_shelf_gain_db = denormalize(params[:, 0], -18, 18)
    low_shelf_cutoff_freq = denormalize(params[:, 1], 20, 20000)
    low_shelf_q_factor = denormalize(params[:, 2], 0.1, 10)

    band0_gain_db = denormalize(params[:, 3], -18, 18)
    band0_center_freq = denormalize(params[:, 4], 20, 20000)
    band0_q_factor = denormalize(params[:, 5], 0.1, 10)

    band1_gain_db = denormalize(params[:, 6], -18, 18)
    band1_center_freq = denormalize(params[:, 7], 20, 20000)
    band1_q_factor = denormalize(params[:, 8], 0.1, 10)

    band2_gain_db = denormalize(params[:, 9], -18, 18)
    band2_center_freq = denormalize(params[:, 10], 20, 20000)
    band2_q_factor = denormalize(params[:, 11], 0.1, 10)

    band3_gain_db = denormalize(params[:, 12], -18, 18)
    band3_center_freq = denormalize(params[:, 13], 20, 20000)
    band3_q_factor = denormalize(params[:, 14], 0.1, 10)

    high_shelf_gain_db = denormalize(params[:, 15], -18, 18)
    high_shelf_cutoff_freq = denormalize(params[:, 16], 20, 20000)
    high_shelf_q_factor = denormalize(params[:, 17], 0.1, 10)

    audio = dasp_pytorch.parametric_eq(
        audio,
        sample_rate,
        low_shelf_gain_db,
        low_shelf_cutoff_freq,
        low_shelf_q_factor,
        band0_gain_db,
        band0_center_freq,
        band0_q_factor,
        band1_gain_db,
        band1_center_freq,
        band1_q_factor,
        band2_gain_db,
        band2_center_freq,
        band2_q_factor,
        band3_gain_db,
        band3_center_freq,
        band3_q_factor,
        high_shelf_gain_db,
        high_shelf_cutoff_freq,
        high_shelf_q_factor,
    )

    return audio


def apply_simple_autodiff_processor(
    audio: torch.Tensor, params: torch.Tensor, sample_rate: int, *args
):
    bs, chs, seq_len = audio.shape
    bs, num_params = params.shape

    num_parametric_eq_params = 15
    num_compressor_params = 6

    assert num_params == num_parametric_eq_params + num_compressor_params

    parametric_eq_params = params[:, :num_parametric_eq_params]
    compressor_params = params[:, num_parametric_eq_params:]

    audio = apply_parametric_eq(audio, parametric_eq_params, sample_rate)
    audio = apply_compressor(audio, compressor_params, sample_rate)

    return audio


def apply_complex_autodiff_processor(
    audio: torch.Tensor, params: torch.Tensor, sample_rate: int, *args
):
    bs, chs, seq_len = audio.shape
    bs, num_params = params.shape

    num_parametric_eq_params = 18
    num_compressor_params = 6
    num_distortion_params = 1
    num_reverb_params = 25
    num_gain_params = 1

    assert (
        num_params
        == num_parametric_eq_params
        + num_compressor_params
        + num_distortion_params
        + num_reverb_params
        + num_gain_params
    )

    assert torch.all(params >= 0) and torch.all(params <= 1)

    # extract parameters for each effect
    eq_param_start = 0
    eq_param_end = num_parametric_eq_params
    parametric_eq_params = params[:, :num_parametric_eq_params]

    compressor_param_start = eq_param_end
    compressor_param_end = compressor_param_start + num_compressor_params
    compressor_params = params[:, compressor_param_start:compressor_param_end]

    distortion_param_start = compressor_param_end
    distortion_param_end = distortion_param_start + num_distortion_params
    distortion_params = params[:, distortion_param_start:distortion_param_end]

    reverb_param_start = distortion_param_end
    reverb_param_end = reverb_param_start + num_reverb_params
    reverb_params = params[:, reverb_param_start:reverb_param_end]

    gain_param_start = reverb_param_end
    gain_param_end = gain_param_start + num_gain_params
    gain_params = params[:, gain_param_start:gain_param_end]

    # apply effects in sequence
    audio = apply_parametric_eq(audio, parametric_eq_params, sample_rate)
    audio = apply_compressor(audio, compressor_params, sample_rate)
    audio = apply_distortion(audio, distortion_params, sample_rate)
    audio = apply_reverb(audio, reverb_params, sample_rate)
    audio = apply_gain(audio, gain_params, sample_rate)

    return audio


# ------------- VST-like effect wrapper classes ----------------------------
class Parameter:
    def __init__(self, init_value: float, min_value: float, max_value: float):
        self.min_value = min_value
        self.max_value = max_value
        self.set_value(init_value)

    def set_value(self, value: float):
        """Normalize the value to the range [0, 1] and store it as raw_value."""
        assert self.min_value <= value <= self.max_value
        self.raw_value = (value - self.min_value) / (self.max_value - self.min_value)

    def get_value(self):
        """Denormalize the value to the range [min_value, max_value]."""
        return self.raw_value * (self.max_value - self.min_value) + self.min_value


class BasicParametricEQ:
    def __init__(
        self,
        low_shelf_gain_db: float = 0.0,
        low_shelf_cutoff_freq: float = 80.0,
        low_shelf_q_factor: float = 0.707,
        band0_gain_db: float = 0.0,
        band0_cutoff_freq: float = 300.0,
        band0_q_factor: float = 0.707,
        band1_gain_db: float = 0.0,
        band1_cutoff_freq: float = 1000.0,
        band1_q_factor: float = 0.707,
        band2_gain_db: float = 0.0,
        band2_cutoff_freq: float = 3000.0,
        band2_q_factor: float = 0.707,
        band3_gain_db: float = 0.0,
        band3_cutoff_freq: float = 10000.0,
        band3_q_factor: float = 0.707,
        high_shelf_gain_db: float = 0.0,
        high_shelf_cutoff_freq: float = 1000.0,
        high_shelf_q_factor: float = 0.707,
    ):
        self.parameters = {
            "low_shelf_gain_db": Parameter(low_shelf_gain_db, -24.0, 24.0),
            "low_shelf_cutoff_freq": Parameter(low_shelf_cutoff_freq, 20.0, 4000.0),
            "low_shelf_q_factor": Parameter(low_shelf_q_factor, 0.1, 4.0),
            "band0_gain_db": Parameter(band0_gain_db, -24.0, 24.0),
            "band0_cutoff_freq": Parameter(band0_cutoff_freq, 20.0, 10000.0),
            "band0_q_factor": Parameter(band0_q_factor, 0.1, 4.0),
            "band1_gain_db": Parameter(band1_gain_db, -24.0, 24.0),
            "band1_cutoff_freq": Parameter(band1_cutoff_freq, 20.0, 10000.0),
            "band1_q_factor": Parameter(band1_q_factor, 0.1, 4.0),
            "band2_gain_db": Parameter(band2_gain_db, -24.0, 24.0),
            "band2_cutoff_freq": Parameter(band2_cutoff_freq, 20.0, 10000.0),
            "band2_q_factor": Parameter(band2_q_factor, 0.1, 4.0),
            "band3_gain_db": Parameter(band3_gain_db, -24.0, 24.0),
            "band3_cutoff_freq": Parameter(band3_cutoff_freq, 20.0, 10000.0),
            "band3_q_factor": Parameter(band3_q_factor, 0.1, 4.0),
            "high_shelf_gain_db": Parameter(high_shelf_gain_db, -24.0, 24.0),
            "high_shelf_cutoff_freq": Parameter(high_shelf_cutoff_freq, 200.0, 18000.0),
            "high_shelf_q_factor": Parameter(high_shelf_q_factor, 0.1, 4.0),
        }

    def process(self, x: np.ndarray, sample_rate: float):
        return parametric_eq(
            x,
            sample_rate,
            low_shelf_gain_db=self.parameters["low_shelf_gain_db"].get_value(),
            low_shelf_cutoff_freq=self.parameters["low_shelf_cutoff_freq"].get_value(),
            low_shelf_q_factor=self.parameters["low_shelf_q_factor"].get_value(),
            band_gains_db=[
                self.parameters["band0_gain_db"].get_value(),
                self.parameters["band1_gain_db"].get_value(),
                self.parameters["band2_gain_db"].get_value(),
                self.parameters["band3_gain_db"].get_value(),
            ],
            band_cutoff_freqs=[
                self.parameters["band0_cutoff_freq"].get_value(),
                self.parameters["band1_cutoff_freq"].get_value(),
                self.parameters["band2_cutoff_freq"].get_value(),
                self.parameters["band3_cutoff_freq"].get_value(),
            ],
            band_q_factors=[
                self.parameters["band0_q_factor"].get_value(),
                self.parameters["band1_q_factor"].get_value(),
                self.parameters["band2_q_factor"].get_value(),
                self.parameters["band3_q_factor"].get_value(),
            ],
            high_shelf_gain_db=self.parameters["high_shelf_gain_db"].get_value(),
            high_shelf_cutoff_freq=self.parameters[
                "high_shelf_cutoff_freq"
            ].get_value(),
            high_shelf_q_factor=self.parameters["high_shelf_q_factor"].get_value(),
        )


class BasicCompressor:
    def __init__(
        self,
        threshold_db: float = 0.0,
        ratio: float = 4.0,
        attack_ms: float = 1.0,
        release_ms: float = 100.0,
    ):
        self.parameters = {
            "threshold_db": Parameter(threshold_db, -80.0, 0.0),
            "ratio": Parameter(ratio, 1.0, 20.0),
            "attack_ms": Parameter(attack_ms, 0.1, 100.0),
            "release_ms": Parameter(release_ms, 10.0, 1000.0),
        }

    def process(self, x: np.ndarray, sample_rate: float):
        return pedalboard.Compressor(
            threshold_db=self.parameters["threshold_db"].get_value(),
            ratio=self.parameters["ratio"].get_value(),
            attack_ms=self.parameters["attack_ms"].get_value(),
            release_ms=self.parameters["release_ms"].get_value(),
        ).process(x, sample_rate)


class BasicDistortion:
    def __init__(self, drive_db: float = 0.0, output_gain_db: float = 0.0):
        self.parameters = {
            "drive_db": Parameter(0.0, -48.0, 48.0),
            "output_gain_db": Parameter(0.0, -24.0, 24.0),
        }

    def process(self, x: np.ndarray, sample_rate: float):
        out = pedalboard.Distortion(
            drive_db=self.parameters["drive_db"].get_value()
        ).process(x, sample_rate)

        out = pedalboard.Gain(
            gain_db=self.parameters["output_gain_db"].get_value()
        ).process(out, sample_rate)

        return out


class BasicDelay:
    def __init__(
        self, delay_seconds: float = 0.5, feedback: float = 0.5, mix: float = 0.5
    ):
        self.parameters = {
            "delay_seconds": Parameter(delay_seconds, 0.01, 1.0),
            "feedback": Parameter(feedback, 0.05, 1.0),
            "mix": Parameter(mix, 0.0, 1.0),
        }

    def process(self, x: np.ndarray, sample_rate: float):
        return pedalboard.Delay(
            delay_seconds=self.parameters["delay_seconds"].get_value(),
            feedback=self.parameters["feedback"].get_value(),
            mix=self.parameters["mix"].get_value(),
        ).process(x, sample_rate)


class BasicReverb:
    def __init__(
        self,
        room_size: float = 0.5,
        damping: float = 0.5,
        wet_dry: float = 0.5,
        width: float = 0.5,
    ):
        self.parameters = {
            "room_size": Parameter(room_size, 0.0, 1.0),
            "damping": Parameter(damping, 0.0, 1.0),
            "wet_dry": Parameter(wet_dry, 0.0, 1.0),
            "width": Parameter(width, 0.0, 1.0),
        }

    def process(self, x: np.ndarray, sample_rate: float):
        return pedalboard.Reverb(
            room_size=self.parameters["room_size"].get_value(),
            damping=self.parameters["damping"].get_value(),
            wet_level=self.parameters["wet_dry"].get_value(),
            dry_level=1 - self.parameters["wet_dry"].get_value(),
            width=self.parameters["width"].get_value(),
        ).process(x, sample_rate)


class BasicChorus:
    def __init__(
        self,
        rate_hz: float = 1.0,
        centre_delay_ms: float = 7.0,
        depth: float = 0.1,
        feedback: float = 0.5,
        mix: float = 0.5,
    ):
        self.parameters = {
            "rate_hz": Parameter(rate_hz, 0.1, 10.0),
            "centre_delay_ms": Parameter(centre_delay_ms, 0.1, 20.0),
            "depth": Parameter(depth, 0.0, 1.0),
            "feedback": Parameter(feedback, 0.0, 1.0),
            "mix": Parameter(mix, 0.0, 1.0),
        }

    def process(self, x: np.ndarray, sample_rate: float):
        return pedalboard.Chorus(
            centre_delay_ms=self.parameters["centre_delay_ms"].get_value(),
            depth=self.parameters["depth"].get_value(),
            feedback=self.parameters["feedback"].get_value(),
            mix=self.parameters["mix"].get_value(),
        ).process(x, sample_rate)


# -----------------------------------------------------------------------


class RandomParametricEQ(BaseTransform):
    def __init__(
        self,
        sample_rate: float = 44100.0,
        num_bands: int = 3,
        min_gain_db: float = -6.0,
        max_gain_db: float = +6.0,
        min_cutoff_freq: float = 1000.0,
        max_cutoff_freq: float = 10000.0,
        min_q_factor: float = 0.1,
        max_q_factor: float = 4.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.sample_rate = sample_rate
        self.num_bands = num_bands
        self.min_gain_db = min_gain_db
        self.max_gain_db = max_gain_db
        self.min_cutoff_freq = min_cutoff_freq
        self.max_cutoff_freq = max_cutoff_freq
        self.min_q_factor = min_q_factor
        self.max_q_factor = max_q_factor

    def _transform(self, x: torch.Tensor):
        """
        Args:
            x: (torch.Tensor): Array of audio samples with shape (chs, seq_leq).
                The filter will be applied the final dimension, and by default the same
                filter will be applied to all channels.
        """
        low_shelf_gain_db = rand(self.min_gain_db, self.max_gain_db)
        low_shelf_cutoff_freq = loguniform(20.0, 200.0)
        low_shelf_q_factor = rand(self.min_q_factor, self.max_q_factor)

        high_shelf_gain_db = rand(self.min_gain_db, self.max_gain_db)
        high_shelf_cutoff_freq = loguniform(8000.0, 16000.0)
        high_shelf_q_factor = rand(self.min_q_factor, self.max_q_factor)

        band_gain_dbs = []
        band_cutoff_freqs = []
        band_q_factors = []
        for _ in range(self.num_bands):
            band_gain_dbs.append(rand(self.min_gain_db, self.max_gain_db))
            band_cutoff_freqs.append(
                loguniform(self.min_cutoff_freq, self.max_cutoff_freq)
            )
            band_q_factors.append(rand(self.min_q_factor, self.max_q_factor))

        y = parametric_eq(
            x.numpy(),
            self.sample_rate,
            low_shelf_gain_db=low_shelf_gain_db,
            low_shelf_cutoff_freq=low_shelf_cutoff_freq,
            low_shelf_q_factor=low_shelf_q_factor,
            band_gains_db=band_gain_dbs,
            band_cutoff_freqs=band_cutoff_freqs,
            band_q_factors=band_q_factors,
            high_shelf_gain_db=high_shelf_gain_db,
            high_shelf_cutoff_freq=high_shelf_cutoff_freq,
            high_shelf_q_factor=high_shelf_q_factor,
        )

        return torch.from_numpy(y)


def stereo_widener(x: torch.Tensor, width: torch.Tensor):
    sqrt2 = np.sqrt(2)

    left = x[0, ...]
    right = x[1, ...]

    mid = (left + right) / sqrt2
    side = (left - right) / sqrt2

    # amplify mid and side signal seperately:
    mid *= 2 * (1 - width)
    side *= 2 * width

    left = (mid + side) / sqrt2
    right = (mid - side) / sqrt2

    x = torch.stack((left, right), dim=0)

    return x


class RandomStereoWidener(BaseTransform):
    def __init__(
        self,
        sample_rate: float = 44100.0,
        min_width: float = 0.0,
        max_width: float = 1.0,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.sample_rate = sample_rate
        self.min_width = min_width
        self.max_width = max_width

    def _transform(self, x: torch.Tensor):
        width = rand(self.min_width, self.max_width)
        return stereo_widener(x, width)


class RandomVolumeAutomation(BaseTransform):
    def __init__(
        self,
        sample_rate: float = 44100.0,
        min_segment_seconds: float = 3.0,
        min_gain_db: float = -6.0,
        max_gain_db: float = 6.0,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.sample_rate = sample_rate
        self.min_segment_seconds = min_segment_seconds
        self.min_gain_db = min_gain_db
        self.max_gain_db = max_gain_db

    def _transform(self, x: torch.Tensor):
        gain_db = torch.zeros(x.shape[-1]).type_as(x)

        seconds = x.shape[-1] / self.sample_rate
        max_num_segments = max(1, int(seconds // self.min_segment_seconds))

        num_segments = randint(1, max_num_segments)
        segment_lengths = (
            x.shape[-1]
            * np.random.dirichlet(
                [rand(0, 10) for _ in range(num_segments)], 1
            )  # TODO(cm): this can crash training
        ).astype("int")[0]

        segment_lengths = np.maximum(segment_lengths, 1)

        # check the sum is equal to the length of the signal
        diff = segment_lengths.sum() - x.shape[-1]
        if diff < 0:
            segment_lengths[-1] -= diff
        elif diff > 0:
            for idx in range(num_segments):
                if segment_lengths[idx] > diff + 1:
                    segment_lengths[idx] -= diff
                    break

        samples_filled = 0
        start_gain_db = 0
        for idx in range(num_segments):
            segment_samples = segment_lengths[idx]
            if idx != 0:
                start_gain_db = end_gain_db

            # sample random end gain
            end_gain_db = rand(self.min_gain_db, self.max_gain_db)
            fade = torch.linspace(start_gain_db, end_gain_db, steps=segment_samples)
            gain_db[samples_filled : samples_filled + segment_samples] = fade
            samples_filled = samples_filled + segment_samples

        # print(gain_db)
        x *= 10 ** (gain_db / 20.0)
        return x


class RandomPedalboardCompressor(BaseTransform):
    def __init__(
        self,
        sample_rate: float = 44100.0,
        min_threshold_db: float = -42.0,
        max_threshold_db: float = -6.0,
        min_ratio: float = 1.5,
        max_ratio: float = 4.0,
        min_attack_ms: float = 1.0,
        max_attack_ms: float = 50.0,
        min_release_ms: float = 10.0,
        max_release_ms: float = 250.0,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.sample_rate = sample_rate
        self.min_threshold_db = min_threshold_db
        self.max_threshold_db = max_threshold_db
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio
        self.min_attack_ms = min_attack_ms
        self.max_attack_ms = max_attack_ms
        self.min_release_ms = min_release_ms
        self.max_release_ms = max_release_ms

    def _transform(self, x: torch.Tensor):
        board = Pedalboard()
        threshold_db = rand(self.min_threshold_db, self.max_threshold_db)
        ratio = rand(self.min_ratio, self.max_ratio)
        attack_ms = rand(self.min_attack_ms, self.max_attack_ms)
        release_ms = rand(self.min_release_ms, self.max_release_ms)

        board.append(
            Compressor(
                threshold_db=threshold_db,
                ratio=ratio,
                attack_ms=attack_ms,
                release_ms=release_ms,
            )
        )

        # process audio using the pedalboard
        return torch.from_numpy(board(x.numpy(), self.sample_rate))


class RandomPedalboardDelay(BaseTransform):
    def __init__(
        self,
        sample_rate: float = 44100.0,
        min_delay_seconds: float = 0.1,
        max_delay_sconds: float = 1.0,
        min_feedback: float = 0.05,
        max_feedback: float = 0.6,
        min_mix: float = 0.0,
        max_mix: float = 0.7,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.sample_rate = sample_rate
        self.min_delay_seconds = min_delay_seconds
        self.max_delay_seconds = max_delay_sconds
        self.min_feedback = min_feedback
        self.max_feedback = max_feedback
        self.min_mix = min_mix
        self.max_mix = max_mix

    def _transform(self, x: torch.Tensor):
        board = Pedalboard()
        delay_seconds = loguniform(self.min_delay_seconds, self.max_delay_seconds)
        feedback = rand(self.min_feedback, self.max_feedback)
        mix = rand(self.min_mix, self.max_mix)
        board.append(Delay(delay_seconds=delay_seconds, feedback=feedback, mix=mix))
        return torch.from_numpy(board(x.numpy(), self.sample_rate))


class RandomPedalboardChorus(BaseTransform):
    def __init__(
        self,
        sample_rate: float = 44100.0,
        min_rate_hz: float = 0.25,
        max_rate_hz: float = 4.0,
        min_depth: float = 0.0,
        max_depth: float = 0.6,
        min_centre_delay_ms: float = 5.0,
        max_centre_delay_ms: float = 10.0,
        min_feedback: float = 0.1,
        max_feedback: float = 0.6,
        min_mix: float = 0.1,
        max_mix: float = 0.7,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.sample_rate = sample_rate
        self.min_rate_hz = min_rate_hz
        self.max_rate_hz = max_rate_hz
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.min_centre_delay_ms = min_centre_delay_ms
        self.max_centre_delay_ms = max_centre_delay_ms
        self.min_feedback = min_feedback
        self.max_feedback = max_feedback
        self.min_mix = min_mix
        self.max_mix = max_mix

    def _transform(self, x: torch.Tensor):
        board = Pedalboard()
        rate_hz = rand(self.min_rate_hz, self.max_rate_hz)
        depth = rand(self.min_depth, self.max_depth)
        centre_delay_ms = rand(self.min_centre_delay_ms, self.max_centre_delay_ms)
        feedback = rand(self.min_feedback, self.max_feedback)
        mix = rand(self.min_mix, self.max_mix)
        board.append(
            Chorus(
                rate_hz=rate_hz,
                depth=depth,
                centre_delay_ms=centre_delay_ms,
                feedback=feedback,
                mix=mix,
            )
        )
        # process audio using the pedalboard
        return torch.from_numpy(board(x.numpy(), self.sample_rate))


class RandomPedalboardPhaser(BaseTransform):
    def __init__(
        self,
        sample_rate: float = 44100.0,
        min_rate_hz: float = 0.25,
        max_rate_hz: float = 5.0,
        min_depth: float = 0.1,
        max_depth: float = 0.6,
        min_centre_frequency_hz: float = 200.0,
        max_centre_frequency_hz: float = 600.0,
        min_feedback: float = 0.1,
        max_feedback: float = 0.6,
        min_mix: float = 0.1,
        max_mix: float = 0.7,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.sample_rate = sample_rate
        self.min_rate_hz = min_rate_hz
        self.max_rate_hz = max_rate_hz
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.min_centre_frequency_hz = min_centre_frequency_hz
        self.max_centre_frequency_hz = max_centre_frequency_hz
        self.min_feedback = min_feedback
        self.max_feedback = max_feedback
        self.min_mix = min_mix
        self.max_mix = max_mix

    def _transform(self, x: torch.Tensor):
        board = Pedalboard()
        rate_hz = rand(self.min_rate_hz, self.max_rate_hz)
        depth = rand(self.min_depth, self.max_depth)
        centre_frequency_hz = rand(
            self.min_centre_frequency_hz, self.min_centre_frequency_hz
        )
        feedback = rand(self.min_feedback, self.max_feedback)
        mix = rand(self.min_mix, self.max_mix)
        board.append(
            Phaser(
                rate_hz=rate_hz,
                depth=depth,
                centre_frequency_hz=centre_frequency_hz,
                feedback=feedback,
                mix=mix,
            )
        )
        # process audio using the pedalboard
        return torch.from_numpy(board(x.numpy(), self.sample_rate))


class RandomPedalboardLimiter(BaseTransform):
    def __init__(
        self,
        sample_rate: float = 44100.0,
        min_threshold_db: float = -32.0,
        max_threshold_db: float = -6.0,
        min_release_ms: float = 10.0,
        max_release_ms: float = 300.0,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.sample_rate = sample_rate
        self.min_threshold_db = min_threshold_db
        self.max_threshold_db = max_threshold_db
        self.min_release_ms = min_release_ms
        self.max_release_ms = max_release_ms

    def _transform(self, x: torch.Tensor):
        board = Pedalboard()
        threshold_db = rand(self.min_threshold_db, self.max_threshold_db)
        release_ms = rand(self.min_release_ms, self.max_release_ms)
        board.append(
            Limiter(
                threshold_db=threshold_db,
                release_ms=release_ms,
            )
        )
        return torch.from_numpy(board(x.numpy(), self.sample_rate))


class RandomPedalboardDistortion(BaseTransform):
    def __init__(
        self,
        sample_rate: float = 44100.0,
        min_drive_db: float = -12.0,
        max_drive_db: float = 48.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.sample_rate = sample_rate
        self.min_drive_db = min_drive_db
        self.max_drive_db = max_drive_db

    def _transform(self, x: torch.Tensor):
        board = Pedalboard()
        drive_db = rand(self.min_drive_db, self.max_drive_db)
        board.append(Distortion(drive_db=drive_db))
        return torch.from_numpy(board(x.numpy(), self.sample_rate))


class RandomSoxReverb(BaseTransform):
    def __init__(
        self,
        sample_rate: float = 44100.0,
        min_reverberance: float = 10.0,
        max_reverberance: float = 100.0,
        min_high_freq_damping: float = 0.0,
        max_high_freq_damping: float = 100.0,
        min_wet_dry: float = 0.0,
        max_wet_dry: float = 1.0,
        min_room_scale: float = 5.0,
        max_room_scale: float = 100.0,
        min_stereo_depth: float = 20.0,
        max_stereo_depth: float = 100.0,
        min_pre_delay: float = 0.0,
        max_pre_delay: float = 100.0,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.sample_rate = sample_rate
        self.min_reverberance = min_reverberance
        self.max_reverberance = max_reverberance
        self.min_high_freq_damping = min_high_freq_damping
        self.max_high_freq_damping = max_high_freq_damping
        self.min_wet_dry = min_wet_dry
        self.max_wet_dry = max_wet_dry
        self.min_room_scale = min_room_scale
        self.max_room_scale = max_room_scale
        self.min_stereo_depth = min_stereo_depth
        self.max_stereo_depth = max_stereo_depth
        self.min_pre_delay = min_pre_delay
        self.max_pre_delay = max_pre_delay

    def _transform(self, x: torch.Tensor):
        reverberance = rand(self.min_reverberance, self.max_reverberance)
        high_freq_damping = rand(self.min_high_freq_damping, self.max_high_freq_damping)
        room_scale = rand(self.min_room_scale, self.max_room_scale)
        stereo_depth = rand(self.min_stereo_depth, self.max_stereo_depth)
        wet_dry = rand(self.min_wet_dry, self.max_wet_dry)
        pre_delay = rand(self.min_pre_delay, self.max_pre_delay)

        effects = [
            [
                "reverb",
                f"{reverberance}",
                f"{high_freq_damping}",
                f"{room_scale}",
                f"{stereo_depth}",
                f"{pre_delay}",
                "--wet-only",
            ]
        ]
        y, _ = torchaudio.sox_effects.apply_effects_tensor(
            x, self.sample_rate, effects, channels_first=True
        )

        # manual wet/dry mix
        return (x * (1 - wet_dry)) + (y * wet_dry)


class RandomPedalboardReverb(BaseTransform):
    def __init__(
        self,
        sample_rate: float = 44100.0,
        min_room_size: float = 0.0,
        max_room_size: float = 1.0,
        min_damping: float = 0.0,
        max_damping: float = 1.0,
        min_wet_dry: float = 0.0,
        max_wet_dry: float = 0.7,
        min_width: float = 0.0,
        max_width: float = 1.0,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.sample_rate = sample_rate
        self.min_room_size = min_room_size
        self.max_room_size = max_room_size
        self.min_damping = min_damping
        self.max_damping = max_damping
        self.min_wet_dry = min_wet_dry
        self.max_wet_dry = max_wet_dry
        self.min_width = min_width
        self.max_width = max_width

    def _transform(self, x: torch.Tensor):
        board = Pedalboard()
        room_size = rand(self.min_room_size, self.max_room_size)
        damping = rand(self.min_damping, self.max_damping)
        wet_dry = rand(self.min_wet_dry, self.max_wet_dry)
        width = rand(self.min_width, self.max_width)

        board.append(
            Reverb(
                room_size=room_size,
                damping=damping,
                wet_level=wet_dry,
                dry_level=(1 - wet_dry),
                width=width,
            )
        )

        return torch.from_numpy(board(x.numpy(), self.sample_rate))


class LoudnessNormalize(BaseTransform):
    def __init__(
        self,
        sample_rate: float = 44100.0,
        target_lufs_db: float = -32.0,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.sample_rate = sample_rate
        self.meter = pyln.Meter(sample_rate)
        self.target_lufs_db = target_lufs_db

    def _transform(self, x: torch.Tensor):
        x_lufs_db = self.meter.integrated_loudness(x.permute(1, 0).numpy())
        delta_lufs_db = torch.tensor([self.target_lufs_db - x_lufs_db]).float()
        gain_lin = 10.0 ** (delta_lufs_db.clamp(-120, 40.0) / 20.0)
        return gain_lin * x


class Mono2Stereo(BaseTransform):
    def __init__(self) -> None:
        super().__init__(p=1.0)

    def _transform(self, x: torch.Tensor):
        assert x.ndim == 2 and x.shape[0] == 1, "x must be mono"
        return torch.cat([x, x], dim=0)


class RandomPan(BaseTransform):
    def __init__(
        self,
        min_pan: float = -1.0,
        max_pan: float = 1.0,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.min_pan = min_pan
        self.max_pan = max_pan

    def _transform(self, x: torch.Tensor):
        """Constant power panning"""
        assert x.ndim == 2 and x.shape[0] == 2, "x must be stereo"
        theta = rand(self.min_pan, self.max_pan) * np.pi / 4
        x = x * 0.707  # normalize to prevent clipping
        left_x, right_x = x[0], x[1]
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        left_x = left_x * (cos_theta - sin_theta)
        right_x = right_x * (cos_theta + sin_theta)
        return torch.stack([left_x, right_x], dim=0)
