import os
import yaml
import torch
import torchaudio
import pedalboard
import numpy as np
import scipy.signal
import pyloudnorm as pyln
import matplotlib.pyplot as plt

from tqdm import tqdm
from numba import jit
from typing import List, Callable
from importlib import import_module

# ------------------ Audio processing functions ------------------


@jit(nopython=True)
def biqaud(
    gain_dB: float,
    cutoff_freq: float,
    q_factor: float,
    sample_rate: float,
    filter_type: str,
):
    """Use design parameters to generate coeffieicnets for a specific filter type.

    Args:
        gain_dB (float): Shelving filter gain in dB.
        cutoff_freq (float): Cutoff frequency in Hz.
        q_factor (float): Q factor.
        sample_rate (float): Sample rate in Hz.
        filter_type (str): Filter type.
            One of "low_shelf", "high_shelf", or "peaking"

    Returns:
        b (np.ndarray): Numerator filter coefficients stored as [b0, b1, b2]
        a (np.ndarray): Denominator filter coefficients stored as [a0, a1, a2]
    """

    A = 10 ** (gain_dB / 40.0)
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


# Adapted from https://github.com/csteinmetz1/pyloudnorm/blob/master/pyloudnorm/iirfilter.py
def parametric_eq(
    x: torch.Tensor,
    sample_rate: float,
    low_shelf_gain_dB: torch.Tensor,
    low_shelf_cutoff_freq: torch.Tensor,
    low_shelf_q_factor: torch.Tensor,
    first_band_gain_dB: torch.Tensor,
    first_band_cutoff_freq: torch.Tensor,
    first_band_q_factor: torch.Tensor,
    second_band_gain_dB: torch.Tensor,
    second_band_cutoff_freq: torch.Tensor,
    second_band_q_factor: torch.Tensor,
    third_band_gain_dB: torch.Tensor,
    third_band_cutoff_freq: torch.Tensor,
    third_band_q_factor: torch.Tensor,
    fourth_band_gain_dB: torch.Tensor,
    fourth_band_cutoff_freq: torch.Tensor,
    fourth_band_q_factor: torch.Tensor,
    high_shelf_gain_dB: torch.Tensor,
    high_shelf_cutoff_freq: torch.Tensor,
    high_shelf_q_factor: torch.Tensor,
):
    """Six-band parametric EQ.

    Low-shelf -> Band 1 -> Band 2 -> Band 3 -> Band 4 -> High-shelf

    Args:


    """
    x = x[0, :].numpy()

    # move to numpy
    low_shelf_gain_dB = low_shelf_gain_dB.squeeze().numpy()
    low_shelf_cutoff_freq = low_shelf_cutoff_freq.squeeze().numpy()
    low_shelf_q_factor = low_shelf_q_factor.squeeze().numpy()
    first_band_gain_dB = first_band_gain_dB.squeeze().numpy()
    first_band_cutoff_freq = first_band_cutoff_freq.squeeze().numpy()
    first_band_q_factor = first_band_q_factor.squeeze().numpy()
    second_band_gain_dB = second_band_gain_dB.squeeze().numpy()
    second_band_cutoff_freq = second_band_cutoff_freq.squeeze().numpy()
    second_band_q_factor = second_band_q_factor.squeeze().numpy()
    third_band_gain_dB = third_band_gain_dB.squeeze().numpy()
    third_band_cutoff_freq = third_band_cutoff_freq.squeeze().numpy()
    third_band_q_factor = third_band_q_factor.squeeze().numpy()
    fourth_band_gain_dB = fourth_band_gain_dB.squeeze().numpy()
    fourth_band_cutoff_freq = fourth_band_cutoff_freq.squeeze().numpy()
    fourth_band_q_factor = fourth_band_q_factor.squeeze().numpy()
    high_shelf_gain_dB = high_shelf_gain_dB.squeeze().numpy()
    high_shelf_cutoff_freq = high_shelf_cutoff_freq.squeeze().numpy()
    high_shelf_q_factor = high_shelf_q_factor.squeeze().numpy()

    # -------- apply low-shelf filter --------
    b, a = biqaud(
        low_shelf_gain_dB,
        low_shelf_cutoff_freq,
        low_shelf_q_factor,
        sample_rate,
        "low_shelf",
    )
    b = b.astype(np.float32)
    a = a.astype(np.float32)
    x = x.astype(np.float32)
    sos0 = np.concatenate((b, a))
    x = scipy.signal.lfilter(b, a, x)

    if False:
        # -------- apply first-band peaking filter --------
        b, a = biqaud(
            first_band_gain_dB,
            first_band_cutoff_freq,
            first_band_q_factor,
            sample_rate,
            "peaking",
        )
        sos1 = np.concatenate((b, a))
        x = scipy.signal.lfilter(b, a, x)

        # -------- apply second-band peaking filter --------
        b, a = biqaud(
            second_band_gain_dB,
            second_band_cutoff_freq,
            second_band_q_factor,
            sample_rate,
            "peaking",
        )
        sos2 = np.concatenate((b, a))
        x = scipy.signal.lfilter(b, a, x)

        # -------- apply third-band peaking filter --------
        b, a = biqaud(
            third_band_gain_dB,
            third_band_cutoff_freq,
            third_band_q_factor,
            sample_rate,
            "peaking",
        )
        sos3 = np.concatenate((b, a))
        x = scipy.signal.lfilter(b, a, x)

        # -------- apply fourth-band peaking filter --------
        b, a = biqaud(
            fourth_band_gain_dB,
            fourth_band_cutoff_freq,
            fourth_band_q_factor,
            sample_rate,
            "peaking",
        )
        sos4 = np.concatenate((b, a))
        x = scipy.signal.lfilter(b, a, x)

    # -------- apply high-shelf filter --------
    b, a = biqaud(
        high_shelf_gain_dB,
        high_shelf_cutoff_freq,
        high_shelf_q_factor,
        sample_rate,
        "high_shelf",
    )
    sos5 = np.concatenate((b, a))
    x = scipy.signal.lfilter(b, a, x)

    return torch.from_numpy(x).unsqueeze(0).float()


def compressor(
    x: torch.Tensor,
    sample_rate: float,
    threshold_db: torch.Tensor,
    ratio: torch.Tensor,
    attack_ms: torch.Tensor,
    release_ms: torch.Tensor,
    makeup_gain_db: torch.Tensor,
):
    """Apply compressor effect to audio tensor."""

    effect = pedalboard.Compressor(
        threshold_db=threshold_db.numpy(),
        ratio=ratio.numpy(),
        attack_ms=attack_ms.numpy(),
        release_ms=release_ms.numpy(),
    )
    output = effect.process(x.numpy(), sample_rate)
    output = torch.from_numpy(output)

    # apply makeup gain
    gain_lin = 10 ** (makeup_gain_db / 20)
    output *= gain_lin
    return output


def distortion(
    x: torch.Tensor,
    sample_rate: float,
    drive_db: torch.Tensor,
):
    """Apply tanh-based distortion effect to audio tensor."""
    effect = pedalboard.Distortion(drive_db=drive_db.numpy())
    output = effect.process(x.numpy(), sample_rate)
    output = torch.from_numpy(output)
    return output


# ------------------ Normalization  functions ------------------


def batch_peak_normalize(x: torch.Tensor):
    peak = torch.max(torch.abs(x), dim=1)[0]
    x = x / peak[:, None].clamp(min=1e-8)
    return x


def batch_loudness_normalize(x: torch.Tensor, meter: pyln.Meter, target_lufs: float):
    for batch_idx in range(x.shape[0]):
        lufs = meter.integrated_loudness(
            x[batch_idx : batch_idx + 1, ...].permute(1, 0).cpu().numpy()
        )
        gain_db = target_lufs - lufs
        gain_lin = 10 ** (gain_db / 20)
        x[batch_idx, :] = gain_lin * x[batch_idx, :]
    return x


# ------------------ Embedding functions ------------------


def get_param_embeds(
    x: torch.Tensor,
    model: torch.nn.Module,
    sample_rate: float,
    normalization: str = "peak",
):
    if normalization == "peak":
        x = batch_peak_normalize(x)
    elif normalization == "loudness":
        x = batch_loudness_normalize(x, model.meter, model.target_lufs)
    elif normalization == "none":
        pass
    else:
        raise ValueError(f"Unknown normalization: {normalization}")

    x = x.unsqueeze(1)  # add ch dimension
    if sample_rate != 48000:
        x = torchaudio.functional.resample(x, sample_rate, 48000)
    embeddings = model(x)
    embeddings = embeddings.nan_to_num()

    return embeddings


def load_param_model(ckpt_path: str):
    config_path = os.path.join(os.path.dirname(ckpt_path), "config.yaml")

    with open(config_path) as f:
        config = yaml.safe_load(f)

    encoder_configs = config["model"]["init_args"]["encoder"]

    module_path, class_name = encoder_configs["class_path"].rsplit(".", 1)
    module = import_module(module_path)
    model = getattr(module, class_name)(**encoder_configs["init_args"])

    checkpoint = torch.load(ckpt_path, map_location="cpu")

    # load state dicts
    state_dict = {}
    for k, v in checkpoint["state_dict"].items():
        if k.startswith("encoder"):
            state_dict[k.replace("encoder.", "", 1)] = v

    model.load_state_dict(state_dict)
    model.eval()
    return model


def evaluate(
    x: torch.Tensor,
    sr: int,
    plugins: List[pedalboard.VST3Plugin],
    channels: List[int],
    w: torch.Tensor,
    target_embedding: torch.Tensor,
    model: torch.nn.Module,
    ignored_parameters: List[List[str]] = [],
    use_gpu: bool = False,
):
    # w = torch.tanh(w)  # bound the parameters to [-1, 1]
    # w = torch.sigmoid(w)
    w = torch.clamp(w, min=0, max=1)

    w_idx = 0
    for plugin, num_channels, ignored_params in zip(
        plugins, channels, ignored_parameters
    ):
        # randomize parameters
        for name, parameter in plugin.parameters.items():
            if name in ignored_params:
                w_idx += 1
                continue
            else:
                parameter.raw_value = w[w_idx]
                w_idx += 1

        if num_channels == 2:
            x = x.repeat(2, 1)

        # process audio
        x = plugin.process(x.cpu().numpy(), sample_rate=sr)
        x = torch.from_numpy(x)

        if num_channels == 2:
            x = x[0:1, :]

    # crop early transient and make mono
    x = x[:, 1024:]

    x /= torch.max(torch.abs(x)).clamp(min=1e-8)

    # extract embedding from the input
    if use_gpu:
        x = x.cuda()

    output_embedding = get_param_embeds(x, model, sample_rate=sr, normalization="peak")

    x = x.cpu()

    # measure the cosine distance between the embeddings
    sim = torch.nn.functional.cosine_similarity(
        output_embedding, target_embedding, dim=1
    )

    return sim, x


if __name__ == "__main__":
    # load vst plugin
    vst_filepaths = [
        "plugins/valid/ZamEQ2.vst3",
        "plugins/valid/ZaMultiCompX2.vst3",
    ]
    channels = [1, 2]

    ignored_parameters = [
        [
            "buffer_size_frames",
            "sample_rate_frames",
            "current_program",
            "peaks_on",
        ],
        [],
    ]

    plugins = []
    num_params = 0
    for vst_filepath in vst_filepaths:
        plugin = pedalboard.load_plugin(vst_filepath)
        plugins.append(plugin)

        for name, parameter in plugin.parameters.items():
            print(name)
            num_params += 1

    # load model
    ckpt_path = "/import/c4dm-datasets-ext/lcap/lcap/dsus0kmd/checkpoints/epoch=146-step=321489.ckpt"
    model = load_param_model(ckpt_path)

    # input_filepath = "./audio/simple-di.wav"
    # target_filepath = "./audio/Gtr_Clean.wav"
    # target_filepath = "/import/c4dm-datasets-ext/podcasts/Am I the Jerk?.mp3"
    # target_filepath = "audio/simple-distortion.wav"

    input_filepath = "audio/m9_script5_produced_corrupt.wav"
    target_filepath = "/import/c4dm-datasets-ext/podcasts/Am I the Jerk?.mp3"

    input_audio, sr = torchaudio.load(input_filepath, backend="soundfile")
    input_audio = torchaudio.functional.resample(input_audio, sr, 48000)

    target_audio, sr = torchaudio.load(target_filepath, backend="soundfile")
    target_audio = torchaudio.functional.resample(target_audio, sr, 48000)

    sr = 48000

    input_audio = input_audio[0:1, 131072:262144]
    target_audio = target_audio[0:1, 131072:262144]

    outputs = {}
    rewards = []
    num_epochs = 1000
    num_pop = 8
    sigma = 0.2
    alpha = 0.1
    use_gpu = False

    if use_gpu:
        target_audio = target_audio.cuda()
        model.cuda()

    # extract embedding from the target
    target_embedding = get_param_embeds(
        target_audio, model, sample_rate=sr, normalization="peak"
    )

    w = torch.sigmoid(torch.randn(num_params) * 0.001)

    for n in tqdm(range(num_epochs)):
        x = input_audio.clone()

        # samples from a normal distribution N(0,1)
        N = torch.randn((num_pop, num_params))
        R = torch.zeros(num_pop)
        for j in range(num_pop):
            w_try = w + sigma * N[j]  # jitter w using gaussian of sigma 0.1
            print(w_try)

            sim, x = evaluate(
                x,
                sr,
                plugins,
                channels,
                w_try,
                target_embedding,
                model,
                ignored_parameters=ignored_parameters,
                use_gpu=use_gpu,
            )

            R[j] = sim  # evaluate the jittered version

        # standardize the rewards to have a gaussian distribution
        A = (R - torch.mean(R)) / torch.std(R)
        # perform the parameter update. The matrix multiply below
        # is just an efficient way to sum up all the rows of the noise matrix N,
        # where each row N[j] is weighted by A[j]
        w = w + alpha / (num_pop * sigma) * torch.matmul(N.T, A)

        # save the current solution
        sim, output_audio = evaluate(
            input_audio,
            sr,
            plugins,
            channels,
            w,
            target_embedding,
            model,
            ignored_parameters=ignored_parameters,
        )

        print(f"{n} - {sim.item():0.4f}")
        # w_norm = torch.tanh(w)  # bound the parameters to [-1, 1]
        w_norm = w  # torch.sigmoid(w)
        # print(w)
        # print(w_norm)
        rewards.append(sim.item())

        fig, axs = plt.subplots(1, 1, figsize=(10, 10))
        plt.plot(rewards)
        plt.savefig("sweep/rewards.png")
        plt.close("all")

        if n % 10 == 0:
            # save the current solution
            torchaudio.save(
                f"sweep/{n}_output_audio.wav", output_audio, sr, backend="soundfile"
            )

            if n == 0:
                torchaudio.save(
                    "sweep/input_audio.wav", input_audio, sr, backend="soundfile"
                )
                torchaudio.save(
                    "sweep/target_audio.wav", target_audio, sr, backend="soundfile"
                )
