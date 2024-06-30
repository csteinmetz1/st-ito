# inference-time optimization methods
import os
import cma
import torch
import torchaudio
import pedalboard
import numpy as np
import scipy.signal
import pyloudnorm as pyln
import multiprocessing as mp

from typing import List

# ------- audio processing methods -------


def load_plugins(plugins: dict):
    total_num_params = 0
    init_params = []
    for plugin_name, plugin in plugins.items():
        if "vst_filepath" in plugin:
            plugin_instance = pedalboard.load_plugin(plugin["vst_filepath"])
        elif "class_path" in plugin:
            plugin_instance = plugin["class_path"]()
        else:
            raise ValueError(f"Plugin must contain 'vst_filepath' or 'class_path'.")

        plugin["parameter_names"] = ["our_bypass"]
        init_params.append(0.0)
        num_params = 1
        for name, parameter in plugin_instance.parameters.items():
            num_params += 1
            print(f"{plugin_name}: {name} = {parameter.raw_value}")
            init_params.append(parameter.raw_value)
            plugin["parameter_names"].append(name)
        print()

        plugin["num_params"] = num_params
        plugin["instance"] = plugin_instance
        total_num_params += num_params

    return plugins, total_num_params, init_params


def process_audio(
    x: np.ndarray,
    w: np.ndarray,
    sr: int,
    plugins: List[dict],
    normalize_stages: bool = False,
):
    """Process audio with plugins and provided parameters on [0, 1].

    Args:
        x (np.ndarray): Audio vector of shape (1, num_samples)
        w (np.ndarray): Parameter vector of shape (num_params,)
        sr (int): Sample rate
        plugins (List[dict]): List of plugin dicts
        normalize_stages (bool): Normalize the output of each stage
    """

    # reflect pad to avoid transient artifacts
    # x = np.pad(x, ((0, 0), (2048, 0)), mode="reflect")

    widx = 0
    for plugin_name, plugin in plugins.items():
        if "instance" not in plugin:
            if "vst_filepath" in plugin:
                plugin_instance = pedalboard.load_plugin(plugin["vst_filepath"])
            elif "class_path" in plugin:
                plugin_instance = plugin["class_path"]()
            else:
                raise ValueError(f"Plugin must contain 'vst_filepath' or 'class_path'.")

            plugin["instance"] = plugin_instance
        for name in plugin["parameter_names"]:
            if not name == "our_bypass":
                parameter = plugin["instance"].parameters[name]
                if name in plugin["fixed_parameters"]:
                    if "vst_filepath" in plugin:
                        parameter.raw_value = plugin["fixed_parameters"][name]
                    else:
                        parameter.set_value(plugin["fixed_parameters"][name])
                    widx += 1
                else:
                    parameter.raw_value = w[widx]
                    widx += 1
            else:
                if w[widx] > 0.5:
                    widx += 1
                    continue  # don't run this plugin
                widx += 1

        if plugin["num_channels"] == 2 and x.shape[0] == 1:
            x = np.concatenate((x, x), axis=0)

        # process left and right channels separately
        if plugin["num_channels"] == 1 and x.shape[0] == 2:
            # process audio
            x_l = plugin["instance"].process(x[0:1, :], sample_rate=sr)
            x_r = plugin["instance"].process(x[1:2, :], sample_rate=sr)
            x = np.concatenate((x_l, x_r), axis=0)
        else:
            x = plugin["instance"].process(x, sample_rate=sr)

        if normalize_stages:
            x /= np.clip(np.max(np.abs(x)), a_min=1e-8, a_max=None)

    # crop early transient
    # x = x[:, 2048:]

    # peak normalize
    x /= np.clip(np.max(np.abs(x)), a_min=1e-8, a_max=None)

    return x


# ------ main methods -------


def run_input(
    input_audio: torch.Tensor,
    target_audio: torch.Tensor,
    sample_rate: int,
    plugins: List[dict],
    model: torch.nn.Module,
    *args,
    **kwargs,
):
    bs, chs, seq_len = input_audio.shape
    result = {
        "output_audio": input_audio,
    }

    return result


def run_random(
    input_audio: torch.Tensor,
    target_audio: torch.Tensor,
    sample_rate: int,
    plugins: List[dict],
    model: torch.nn.Module,
    *args,
    **kwargs,
):
    bs, chs, seq_len = input_audio.shape
    total_num_params = sum([plugin["num_params"] for plugin in plugins.values()])
    w = torch.rand(total_num_params)
    output_audio = process_audio(
        input_audio.squeeze(0).numpy(), w.numpy(), sample_rate, plugins
    )
    output_audio = torch.from_numpy(output_audio).unsqueeze(0)

    result = {
        "output_audio": output_audio,
        "param_dict": parameters_to_dict(w.numpy(), plugins),
    }

    return result


def smooth_spectrum(H: np.ndarray):
    # apply Savgol filter for smoothed target curve
    return scipy.signal.savgol_filter(H, 1025, 2)


def get_average_spectrum(x: torch.Tensor, n_fft: int = 16384):
    print(x.shape)
    # if stereo sum to mono
    if x.shape[0] == 2:
        x = x.mean(dim=0, keepdim=True)
    # x = x[:, : self.n_fft]
    X = torch.stft(x, n_fft, return_complex=True, normalized=True)
    # fft_size = self.next_power_of_2(x.shape[-1])
    # X = torch.fft.rfft(x, n=fft_size)

    X = X.abs()  # convert to magnitude
    X = X.mean(dim=-1).view(-1)  # average across frames

    return X


def run_rule_based(
    input_audio: torch.Tensor,
    target_audio: torch.Tensor,
    sample_rate: int,
    plugins: List[dict],
    model: torch.nn.Module,
    n_fft: int = 16384,
    n_taps: int = 2048,
    **kwargs,
):
    """Rule-based style transfer baseline.

    This approach performs style transfer by designing a matched EQ curve based on
    the target audio and performs a simple iterative hill climbing optimization to
    find the best value of the threshold parameter for the dynamics processor.

    Args:
        input_audio (torch.Tensor): Input audio tensor of shape (bs, chs, seq_len)
        target_audio (torch.Tensor): Target audio tensor of shape (bs, chs, seq_len)
        sample_rate (int): Sample rate
        plugins (List[dict]): List of plugin dicts (Not used in this method)
        model (torch.nn.Module): Model (Not used in this method)

    """
    bs, chs, seq_len = input_audio.shape
    bs, chs, seq_len = target_audio.shape

    audio_outputs = []
    for bidx in range(bs):
        # get batch audio
        input_audio_elem = input_audio[bidx, :, :]
        target_audio_elem = target_audio[bidx, :, :]

        # peak normalize to -12 dBFS
        input_audio_elem /= torch.max(torch.abs(input_audio_elem)).clamp(min=1e-8)
        target_audio_elem /= torch.max(torch.abs(target_audio_elem)).clamp(min=1e-8)
        input_audio_elem *= 10 ** (-12 / 20)
        target_audio_elem *= 10 ** (-12 / 20)

        # ------------ design the matched EQ ------------
        in_spec = get_average_spectrum(input_audio_elem, n_fft=n_fft)
        ref_spec = get_average_spectrum(target_audio_elem, n_fft=n_fft)
        sm_in_spec = smooth_spectrum(in_spec)
        sm_ref_spec = smooth_spectrum(ref_spec)

        # design inverse FIR filter to match target EQ
        freqs = np.linspace(0, 1.0, num=(n_fft // 2) + 1)
        response = sm_ref_spec / sm_in_spec
        response[-1] = 0.0  # zero gain at nyquist

        b = scipy.signal.firwin2(
            n_taps,
            freqs * (sample_rate / 2),
            response,
            fs=sample_rate,
        )

        # apply the filter
        x_filt = scipy.signal.lfilter(b, [1.0], input_audio_elem.numpy())
        input_audio_elem = torch.tensor(x_filt.astype("float32"))

        # peak normalize again
        input_audio_elem /= torch.max(torch.abs(input_audio_elem)).clamp(min=1e-8)
        input_audio_elem *= 10 ** (-12 / 20)

        # ------------ apply dynamics processor ------------
        meter = pyln.Meter(sample_rate)
        input_lufs = meter.integrated_loudness(input_audio_elem.permute(1, 0).numpy())
        target_lufs = meter.integrated_loudness(target_audio_elem.permute(1, 0).numpy())

        delta_lufs = target_lufs - input_lufs

        threshold_db = 0.0
        x_comp = input_audio_elem.numpy()
        x_comp_new = input_audio_elem.numpy()
        while delta_lufs > 0.25 and threshold_db > -80.0:
            x_comp = x_comp_new  # use the last setting
            x_comp_new = pedalboard.Compressor(
                threshold_db=threshold_db, ratio=3.0, attack_ms=1.0, release_ms=100.0
            ).process(x_comp, sample_rate=sample_rate)
            x_comp_new /= np.max(np.abs(x_comp_new))
            x_comp_new *= 10 ** (-12 / 20)
            x_lufs = meter.integrated_loudness(x_comp_new.T)
            delta_lufs = target_lufs - x_lufs
            threshold_db -= 0.5  # decrease threshold

        # convert to torch
        x_comp_new = torch.tensor(x_comp_new.astype("float32"))
        audio_outputs.append(x_comp_new)

    result = {
        "output_audio": torch.stack(audio_outputs, dim=0),
    }

    return result


def run_deepafx_st(
    input_audio: torch.Tensor,
    target_audio: torch.Tensor,
    sample_rate: int,
    plugins: List[dict],
    model: torch.nn.Module,
    *args,
    **kwargs,
):
    bs, chs, seq_len = input_audio.size()
    bs, chs, seq_len = target_audio.size()

    with torch.no_grad():
        # peak normalize
        # input_audio /= torch.max(torch.abs(input_audio)).clamp(min=1e-8)
        # target_audio /= torch.max(torch.abs(target_audio)).clamp(min=1e-8)

        output_audio, params, params_logits = model(
            input_audio,
            target_audio,
            sample_rate=sample_rate,
            render_audio=True,
        )

    if output_audio.shape[1] == 1:
        output_audio = output_audio.repeat(1, chs, 1)

    # store parameters in dict
    param_dict = {}
    for idx in range(params.shape[-1]):
        param_dict[f"{idx}"] = params[..., idx].item()

    result = {
        "output_audio": output_audio,
        "params": param_dict,
    }

    return result


# ----------- Evolutionary Strategies ------------


def parameters_to_dict(w: np.ndarray, plugins: List[dict]):
    """Convert parameter vector to dictionary.

    Args:
        w (np.ndarray): Parameter vector
        plugins (List[dict]): List of plugin dicts
    """
    widx = 0
    w_dict = {}
    for plugin_name, plugin in plugins.items():
        if plugin_name not in w_dict:
            w_dict[plugin_name] = {}

        for name in plugin["parameter_names"]:
            if name == "our_bypass":
                w_dict[plugin_name][name] = w[widx]
                widx += 1
                continue
            else:
                parameter = plugin["instance"].parameters[name]
                if name in plugin["fixed_parameters"]:
                    if "vst_filepath" in plugin:
                        parameter.raw_value = plugin["fixed_parameters"][name]
                    else:
                        parameter.set_value(plugin["fixed_parameters"][name])
                    widx += 1
                else:
                    parameter.raw_value = w[widx]
                    widx += 1

                if hasattr(parameter, "get_value"):
                    w_dict[plugin_name][name] = parameter.get_value()
                else:
                    w_dict[plugin_name][name] = parameter.raw_value

    return w_dict


def savepop_to_disk(
    iteration: str,
    fvals: List[float],
    output_embeds: List[torch.Tensor],
    output_audios: List[torch.Tensor],
    run_dir: str,
    sample_rate: int,
):
    # create directory for the current iteration
    pop_dir = os.path.join(run_dir, f"pop_{iteration}")
    os.makedirs(pop_dir, exist_ok=True)
    # save the current population
    pop_members = []

    for idx, (fval, output_audio, output_embed) in enumerate(
        zip(fvals, output_audios, output_embeds)
    ):
        pop_members.append((fval, output_audio, output_embed))

    # sort by fval
    pop_members = sorted(pop_members, key=lambda x: x[0])

    # save to disk
    for idx, (fval, output_audio, _) in enumerate(pop_members):
        output_audio_filepath = os.path.join(
            pop_dir,
            f"output_audio_pop_{idx}_fval_{fval:0.4e}.wav",
        )
        output_audio /= torch.max(torch.abs(output_audio)).clamp(min=1e-8)
        torchaudio.save(
            output_audio_filepath,
            output_audio.squeeze(0),
            sample_rate,
            backend="soundfile",
        )


def run_es(
    input_audio: torch.Tensor,
    target_audio: torch.Tensor,
    sample_rate: int,
    plugins: List[dict],
    model: torch.nn.Module,
    embed_func: callable,
    content_model: torch.nn.Module = None,
    content_embed_func: callable = None,
    max_iters: int = 100,
    w0: torch.Tensor = None,
    find_w0: bool = True,
    sigma0: float = 0.1,
    distance: str = "cosine",
    random_crop: bool = False,
    popsize: int = 32,
    parallel: bool = False,
    dropout: float = 0.0,
    savepop: bool = False,
    run_dir: str = ".",
    *args,
    **kwargs,
):
    """Run CMA-ES optimization to find the best parameters.

    Args:
        input_audio (torch.Tensor): Input audio tensor of shape (bs, chs, seq_len)
        target_audio (torch.Tensor): Target audio tensor of shape (bs, chs, seq_len)
        sample_rate (int): Sample rate
        plugins (List[dict]): List of plugin dicts
        model (torch.nn.Module): Model
        embed_func (callable): Embedding function
        normalization (str): Normalization type
        max_iters (int): Max iterations
        w0 (torch.Tensor): Initial parameters
        find_w0 (bool): Evaluate random initial parameters selecting the best
        sigma0 (float): Initial sigma
        distance (str): Distance type
        random_crop (bool): Apply random crop to the input audio
        popsize (int): Population size
        parallel (bool): Use parallel processing
        savepop (bool): Save population to disk

    """
    # count total number of parameters
    total_num_params = sum([plugin["num_params"] for plugin in plugins.values()])

    bs, chs, seq_len = input_audio.shape

    # input_audio = input_audio.unsqueeze(0)
    # target_audio = target_audio.unsqueeze(0)

    # peak normalize
    input_audio /= torch.max(torch.abs(input_audio)).clamp(min=1e-8)
    target_audio /= torch.max(torch.abs(target_audio)).clamp(min=1e-8)

    # compute target embedding (only once)
    target_embed = embed_func(
        target_audio,
        model,
        sample_rate,
    )

    print(content_model)

    if content_model is not None:
        target_content_embeds = content_embed_func(
            target_audio,
            content_model,
            sample_rate,
        )
    else:
        target_content_embeds = None

    # define evaluation function
    def evaluate(
        W: torch.Tensor,
        x: torch.Tensor,
        sample_rate: int,
        plugins: List[dict],
        target_embeds: torch.Tensor,
        target_content_embeds: torch.Tensor = None,
        parallel: bool = False,
        dropout: float = 0.0,
    ):
        """Evaluate the current solution.

        Args:
            W (np.ndarray): List of plugin parameters
            x (np.ndarray): input audio
            sample_rate (int): Sample rate
            plugins (List[dict]): List of plugin dicts
            target_embeds (torch.Tensor): Target embedding
            parallel (bool): Use parallel processing
            dropout (float): Dropout rate

        Returns:
            dict : Dictionary containing the output audio, parameters, and distance
        """

        if parallel:
            with mp.Pool(processes=16) as pool:
                args = [(x.squeeze(0).numpy(), w, sample_rate, plugins) for w in W]
                output_audios = pool.starmap(process_audio, args)
        else:
            output_audios = []
            crop_len = 262144
            if (
                random_crop and (x.shape[-1] - crop_len) > 16384
            ):  # apply random crop (same crop for all in population)
                start_idx = np.random.randint(16384, x.shape[-1] - crop_len)
            else:
                start_idx = 0
            for w in W:
                if x.shape[-1] > crop_len:
                    x_crop = (
                        x[:, :, start_idx : start_idx + crop_len] if random_crop else x
                    )
                else:
                    x_crop = torch.nn.functional.pad(x, (0, 262144 - x.shape[-1]))
                output_audios.append(
                    process_audio(x_crop.squeeze(0).numpy(), w, sample_rate, plugins)
                )

        output_audios = [
            torch.from_numpy(output_audio) for output_audio in output_audios
        ]

        # cat along batch dim
        output_audios = torch.stack(output_audios, dim=0)

        # compute embedding
        output_embeds = embed_func(
            output_audios,
            model,
            sample_rate,
        )

        if content_model is not None:
            output_content_embeds = content_embed_func(
                output_audios,
                content_model,
                sample_rate,
            )

        dists = []
        for output_idx, (embed_name, output_embed) in enumerate(output_embeds.items()):

            target_embed = target_embeds[embed_name]

            # apply dropout to the embeddings
            if dropout > 0.0:
                output_embed = torch.nn.functional.dropout(output_embed, p=dropout)

            # compute distance
            with torch.no_grad():
                dist = torch.cosine_similarity(output_embed, target_embed, dim=-1)
                dist = -dist

            # aggregate distances
            dists.append(dist)

        # now add the content distance
        if target_content_embeds is not None:
            for embed_name, output_content_embed in output_content_embeds.items():
                dist = torch.cosine_similarity(
                    output_content_embed, target_content_embeds[embed_name], dim=-1
                )
                dist = -dist
                dists.append(2 * dist)

        dists = torch.stack(dists, dim=0)
        dist = dists.mean(dim=0)

        return dist.tolist(), output_embeds, output_audios

    # setup CMA-ES
    if find_w0:
        print("Finding the best w0...")
        tmp_w0s = []
        for n in range(popsize):
            tmp_w0 = np.random.rand(total_num_params)
            tmp_w0s.append(tmp_w0)
        fvals, output_embeds, output_audios = evaluate(
            tmp_w0s,
            input_audio,
            sample_rate,
            plugins,
            target_embed,
            target_content_embeds=target_content_embeds,
            parallel=parallel,
            dropout=dropout,
        )
        # find the best w0
        print(fvals)
        w0 = tmp_w0s[np.argmin(fvals)]
        if savepop:
            savepop_to_disk(
                -1,
                fvals,
                output_embeds,
                output_audios,
                run_dir,
                sample_rate,
            )
    else:
        if w0 is None:
            w0 = np.ones(total_num_params) * 0.5
        else:
            w0 = w0.numpy()

    # convert parameters to dictionary
    init_param_dict = parameters_to_dict(w0, plugins)
    print(init_param_dict)

    es = cma.CMAEvolutionStrategy(w0, sigma0, {"bounds": [0, 1], "popsize": popsize})

    # tracking history
    fval_history = []
    wopt_history = []
    iters_without_improvement = 0

    # run CMA-ES
    for iteration in range(max_iters):
        x = input_audio.clone()
        W = es.ask()
        fvals, output_embeds, output_audios = evaluate(
            W,
            x,
            sample_rate,
            plugins,
            target_embed,
            target_content_embeds=target_content_embeds,
            parallel=parallel,
            dropout=(
                dropout if (iteration + 1) < max_iters else 0.0
            ),  # on the last iteration, do not apply dropout
        )

        # save best
        wopt_history.append(es.result[0])
        fval_history.append(es.result[1])

        if savepop:
            savepop_to_disk(
                iteration,
                fvals,
                output_embeds,
                output_audios,
                run_dir,
                sample_rate,
            )
        es.tell(W, fvals)
        es.disp()

        # check if the the best fval has improved
        if iteration > 0:
            fval_delta = min(fvals) - min(fval_history)
        else:
            fval_delta = -0.02

        if fval_delta > -0.01:
            iters_without_improvement += 1
            print(
                f"Solution has not improved for {iters_without_improvement} iterations."
            )
        else:
            iters_without_improvement = 0

        if iters_without_improvement > 10:
            print("Stopping early due to no improvement.")
            break

    wopt = es.result[0]
    fopt = es.result[1]

    # save the current solution
    output_audio = torch.from_numpy(
        process_audio(x.squeeze(0).numpy(), wopt, sample_rate, plugins)
    )

    # convert parameters to dictionary
    param_dict = parameters_to_dict(wopt, plugins)

    result = {
        "output_audio": output_audio,
        "params": param_dict,
        "fopt": fopt,
        "wopt": wopt,
        "fval_history": fval_history,
        "wopt_history": wopt_history,
    }

    return result
