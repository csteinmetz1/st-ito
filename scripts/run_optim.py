import os
import cma
import json
import torch
import argparse
import torchaudio
import pedalboard
import numpy as np
import dasp_pytorch
import multiprocessing as mp
import matplotlib.pyplot as plt

from tqdm import tqdm
from typing import List

from st_ito.effects import (
    BasicParametricEQ,
    BasicCompressor,
    BasicDistortion,
    BasicDelay,
    BasicReverb,
    apply_complex_autodiff_processor,
)
from st_ito.style_transfer import (
    run_input,
    run_es,
    run_deepafx_st,
    process_audio,
)

from st_ito.utils import (
    load_param_model,
    get_param_embeds,
    load_clap_model,
    get_clap_embeds,
)


def run_staged_es(
    input_audio: torch.Tensor,
    target_audio: torch.Tensor,
    sample_rate: int,
    plugins: List[dict],
    model: torch.nn.Module,
    embed_func: callable,
    normalization: str = "peak",
    max_iters: int = 100,
    w0: torch.Tensor = None,
    popsize: int = 10,
    sigma0: float = 0.1,
    distance: str = "cosine",
    parallel: bool = False,
    save_pop: bool = False,
    *args,
    **kwargs,
):
    """Run CMA-ES to optimize audio processing parameters.

    Args:
        input_audio (torch.Tensor): Input audio tensor
        target_audio (torch.Tensor): Target audio tensor
        sample_rate (int): Sample rate
        plugins (List[dict]): List of plugin dicts
        model (torch.nn.Module): Embedding model
        embed_func (callable): Embedding function
        normalization (str): Normalization type
        max_iters (int): Maximum number of iterations
        w0 (torch.Tensor): Initial parameters
        popsize (int): Population size
        sigma0 (float): Initial sigma
        distance (str): Distance type (l2 or cosine)
        parallel (bool): Use parallel processing
        save_pop (bool): Save the population audio and params at each iteration
    """
    # count total number of parameters
    total_num_parms = sum([plugin["num_params"] for plugin in plugins.values()])

    # compute target embedding (only once)
    target_embed = embed_func(
        target_audio.unsqueeze(0),
        model,
        sample_rate,
    )

    # define evaluation function
    def evaluate(
        W: torch.Tensor,
        x: torch.Tensor,
        sr: int,
        plugins: List[dict],
        target_embeds: torch.Tensor,
        parallel: bool = True,
    ):
        """Evaluate the current solution.

        W (np.ndarray): List of plugin parameters
        x (np.ndarray): input audio

        """

        if parallel:
            with mp.Pool(processes=32) as pool:
                args = [(x.numpy(), w, sr, plugins) for w in W]
                output_audios = pool.starmap(process_audio, args)
        else:
            output_audios = []
            for w in W:
                output_audios.append(process_audio(x, w, sr, plugins))

        output_audios = [
            torch.from_numpy(output_audio) for output_audio in output_audios
        ]

        # cat along batch dim
        output_audios = torch.stack(output_audios, dim=0)

        # compute embedding
        output_embeds = embed_func(
            output_audios,
            model,
            sr,
        )

        # compute distance
        with torch.no_grad():
            if distance == "l2":
                dist = torch.nn.functional.mse_loss(
                    output_embeds, target_embeds, reduction="none"
                )
                dist = torch.mean(dist, dim=-1)
            elif distance == "cosine":
                dist = torch.cosine_similarity(output_embeds, target_embeds, dim=-1)
                dist = -dist
            else:
                raise ValueError(f"Unknown distance: {distance}")

        return dist.tolist(), output_audios

    wopt = None  # start with no initial parameters
    wopt_overall = None

    fval_history = []
    wopt_history = []

    # optimize each stage separately
    for stage_idx in range(len(plugins)):
        # create a sub-chain with only the current stage (and previous stages)
        stage_plugins = {
            k: v for idx, (k, v) in enumerate(plugins.items()) if idx <= stage_idx
        }

        stage_plugin_names = list(stage_plugins.keys())
        print(f"Optimizing stage {stage_idx} ({stage_plugin_names})")

        total_num_parms = sum(
            [plugin["num_params"] for plugin in stage_plugins.values()]
        )
        total_num_stage_params = list(stage_plugins.values())[-1]["num_params"]

        # randomly init parameters for this stage
        w0_stage = np.ones(total_num_stage_params) * 0.5

        # setup CMA-ES
        es = cma.CMAEvolutionStrategy(
            w0_stage, sigma0, {"bounds": [0, 1], "popsize": popsize}
        )

        for iteration in range(max_iters // len(plugins)):
            x = input_audio.clone()
            W = es.ask()

            # use the parameters from the previous stage
            # combine the parameters from the previous stages with the current stage
            if stage_idx > 0:
                W_stage = []
                for idx, w in enumerate(W):
                    W_stage.append(np.zeros(total_num_parms))
                    W_stage[idx][: len(wopt_overall)] = wopt_overall
                    W_stage[idx][len(wopt_overall) :] = w
            else:
                W_stage = W

            fvals, output_audios = evaluate(
                W_stage, x, sample_rate, stage_plugins, target_embed
            )
            es.tell(W, fvals)
            es.disp()

            if save_pop:
                # create directory for the current iteration

                # save the current population
                for idx, (fval, output_audio) in enumerate(zip(fvals, output_audios)):
                    output_audio_filepath = os.path.join(
                        run_dir,
                        f"output_audio_stage_{stage_idx}_pop_{idx}_fval_{fval:0.3f}.wav",
                    )
                    output_audio /= torch.max(torch.abs(output_audio)).clamp(min=1e-8)
                    torchaudio.save(
                        output_audio_filepath,
                        output_audio.squeeze(0),
                        sample_rate,
                        backend="soundfile",
                    )

            # best solution
            wopt_stage = es.result[0]
            fopt_stage = es.result[1]
            fval_history.append(fopt_stage)
            wopt_history.append(wopt_stage)

        wopt = es.result[0]
        fopt = es.result[1]

        if wopt_overall is None:
            wopt_overall = wopt
        else:
            wopt_overall_new = np.zeros(total_num_parms)
            wopt_overall_new[: len(wopt_overall)] = wopt_overall
            wopt_overall_new[len(wopt_overall) :] = wopt
            wopt_overall = wopt_overall_new  # replace

        # save the current solution
        output_audio = torch.from_numpy(
            process_audio(x.numpy(), wopt_overall, sample_rate, stage_plugins)
        )

        output_filepath = os.path.join(run_dir, f"output_audio_stage_{stage_idx}.wav")
        torchaudio.save(output_filepath, output_audio, sample_rate, backend="soundfile")

    # convert parameters to dictionary
    param_dict = parameters_to_dict(wopt_overall, plugins)

    return output_audio, param_dict, fopt, fval_history, wopt_history


def run_autodiff(
    input_audio: torch.Tensor,
    target_audio: torch.Tensor,
    sample_rate: int,
    model: torch.nn.Module,
    embed_func: callable,
    lr: float = 1e-1,
    n_iters: int = 100,
):
    # init parameters to optimize
    num_params = 51
    w = torch.nn.Parameter(
        torch.rand((1, num_params), device="cuda"), requires_grad=True
    )

    # create optimizer
    optimizer = torch.optim.Adam([w], lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_iters)

    # move to GPU
    input_audio = input_audio.cuda()
    target_audio = target_audio.cuda()
    # w = w.cuda()
    model.cuda()

    # compute target embedding
    target_embed = embed_func(target_audio.unsqueeze(0), model, sample_rate)

    wopt_history = []
    fval_history = []

    pbar = tqdm(range(n_iters))
    for n in pbar:
        optimizer.zero_grad()

        # forward pass
        output_audio = apply_complex_autodiff_processor(
            input_audio.unsqueeze(0), torch.sigmoid(w), sample_rate
        )
        output_embed = embed_func(output_audio, model, sample_rate, requires_grad=True)

        # compute loss
        # loss = torch.nn.functional.mse_loss(output_embed, target_embed)
        loss = torch.cosine_similarity(output_embed, target_embed, dim=-1)

        # save history
        fval_history.append(loss.item())
        wopt_history.append(torch.sigmoid(w).detach().cpu().numpy())

        # backward pass
        loss.backward()
        optimizer.step()
        # scheduler.step()

        pbar.set_description(f"Loss: {loss.item():0.4f}")

    # save the current solution
    param_dict = {}
    fopt = fval_history[-1]

    return output_audio.detach().cpu(), param_dict, fopt, fval_history, wopt_history


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str)
    parser.add_argument("target", type=str)
    parser.add_argument("--max-iters", type=int, default=300)
    parser.add_argument("--popsize", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=262144)
    parser.add_argument("--staged", action="store_true")
    parser.add_argument("--savepop", action="store_true")
    parser.add_argument("--normalize-stages", action="store_true")
    parser.add_argument("--use-gpu", action="store_true")
    parser.add_argument("--parallel", action="store_true")
    parser.add_argument(
        "--effect-type", type=str, default="vst", choices=["vst", "basic"]
    )
    parser.add_argument(
        "--algorithm", type=str, default="es", choices=["es", "autodiff"]
    )
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument(
        "--metric", type=str, default="param", choices=["param", "clap"]
    )
    args = parser.parse_args()

    sample_rate = 48000
    os.makedirs("output/optim", exist_ok=True)

    # check for plugins directory
    if not os.path.exists("plugins"):
        os.system(
            "wget https://huggingface.co/csteinmetz1/afx-rep/resolve/main/plugins.tar"
        )
        os.system("tar -xvf plugins.tar")
        os.system("rm plugins.tar")

    # define plugins
    if args.algorithm == "autodiff":
        plugins = {}
        total_num_params = 48
    elif args.algorithm == "es":
        if args.effect_type == "vst":
            buffer_size_frames = 0.03125
            sample_rate_frames = 0.11484374850988388
            current_program = 0.0
            plugins = {
                "ZamEQ2": {
                    "vst_filepath": "plugins/valid/ZamEQ2.vst3",
                    "num_params": None,
                    "num_channels": 1,
                    "fixed_parameters": {
                        "buffer_size_frames": buffer_size_frames,
                        "sample_rate_frames": sample_rate_frames,
                        "current_program": current_program,
                        "peaks_on": 0,
                    },
                },
                # "STR-X": {
                #    "vst_filepath": "plugins/valid/STR-X.vst3",
                #    "num_params": None,
                #    "num_channels": 1,
                #    "fixed_parameters": {"bypass": 0.0},
                # },
                "FlyingDelay": {
                    "vst_filepath": "plugins/valid/FlyingDelay.vst3",
                    "num_params": None,
                    "num_channels": 2,
                    "fixed_parameters": {"bypass": 0.0},
                },
                "TAL-Reverb-4": {
                    "vst_filepath": "plugins/valid/TAL-Reverb-4.vst3",
                    "num_params": None,
                    "num_channels": 2,
                    "fixed_parameters": {"bypass": 0.0, "on_off": 1.0},
                },
            }
        elif args.effect_type == "basic":
            plugins = {
                "ParametricEQ": {
                    "class_path": BasicParametricEQ,
                    "num_params": None,
                    "num_channels": 1,
                    "fixed_parameters": {},
                },
                "Compressor": {
                    "class_path": BasicCompressor,
                    "num_params": None,
                    "num_channels": 1,
                    "fixed_parameters": {},
                },
                "Distortion": {
                    "class_path": BasicDistortion,
                    "num_params": None,
                    "num_channels": 1,
                    "fixed_parameters": {},
                },
                "Delay": {
                    "class_path": BasicDelay,
                    "num_params": None,
                    "num_channels": 2,
                    "fixed_parameters": {},
                },
                "Reverb": {
                    "class_path": BasicReverb,
                    "num_params": None,
                    "num_channels": 2,
                    "fixed_parameters": {},
                },
            }

        # load plugins
        total_num_params = 0
        init_params = []
        for plugin_name, plugin in plugins.items():
            if "vst_filepath" in plugin:
                plugin_instance = pedalboard.load_plugin(plugin["vst_filepath"])
            elif "class_path" in plugin:
                plugin_instance = plugin["class_path"]()
            else:
                raise ValueError(f"Plugin must contain 'vst_filepath' or 'class_path'.")

            num_params = 0
            for name, parameter in plugin_instance.parameters.items():
                num_params += 1
                print(f"{plugin_name}: {name} = {parameter.raw_value}")
                init_params.append(parameter.raw_value)
            print()

            plugin["num_params"] = num_params
            plugin["instance"] = plugin_instance
            total_num_params += num_params

            # create vector of initial parameter values
            w0 = torch.zeros(total_num_params)
            for idx, param in enumerate(init_params):
                w0[idx] = param
    else:
        raise ValueError(f"Unknown algorithm: {args.algorithm}")

    # load audio
    input_audio, input_sr = torchaudio.load(args.input, backend="soundfile")
    input_name = os.path.basename(args.input).replace(".wav", "")

    if input_sr != sample_rate:
        input_audio = torchaudio.functional.resample(
            input_audio,
            input_sr,
            sample_rate,
        )

    if args.target is None:
        # if not target audio is provided, use the input audio
        # to generate a target example by adding some random effects
        # w_target = torch.rand(total_num_params)
        w_parametric_eq = [
            0.1,
            0.5,
            0.2,  # low shelf
            0.5,
            0.5,
            0.2,  # band0
            0.5,
            0.75,
            0.2,  # band1
            0.5,
            0.5,
            0.2,  # band2
            0.5,
            0.5,
            0.2,  # band3
            0.7,
            0.5,
            0.2,  # high shelf
        ]
        w_compressor = [0.8, 0.3, 0.1, 0.1, 0.5, 0.1]
        w_distortion = [0.0]
        w_reverb = [
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
            0.8,
            0.9,
            0.8,
            0.4,
            0.5,
            0.5,
            0.6,
            0.4,
            0.3,
            0.3,
            0.3,
            0.2,
            0.5,
        ]
        w_gain = [0.5]
        w_target = w_parametric_eq + w_compressor + w_distortion + w_reverb + w_gain
        w_target = torch.tensor(w_target, dtype=torch.float32)
        print(w_target.shape)

        if args.algorithm == "autodiff":
            target_audio = apply_complex_autodiff_processor(
                input_audio.unsqueeze(0), w_target.unsqueeze(0), sample_rate
            )
            target_audio = target_audio.squeeze(0)
        elif args.algorithm == "es":
            target_audio = process_audio(input_audio, w_target, sample_rate, plugins)
            target_audio = torch.from_numpy(target_audio)
        else:
            raise ValueError(f"Unknown algorithm: {args.algorithm}")

        target_name = f"synthetic_target"
    else:
        target_audio, target_sr = torchaudio.load(args.target, backend="soundfile")
        target_name = os.path.basename(args.target).replace(".wav", "")
        if target_sr != sample_rate:
            target_audio = torchaudio.functional.resample(
                target_audio,
                target_sr,
                sample_rate,
            )

    # crop to max length
    input_audio = input_audio[:, : args.max_length]
    target_audio = target_audio[:, : args.max_length]

    run_name = f"{input_name}_to_{target_name}_{args.algorithm}"
    run_dir = os.path.join("output", "optim", run_name)

    os.makedirs(run_dir, exist_ok=True)

    # ------ load model and embedding function ------
    if args.metric == "param":
        model = load_param_model(use_gpu=args.use_gpu)
        embed_func = get_param_embeds
    elif args.metric == "clap":
        model = load_clap_model(use_gpu=args.use_gpu)
        embed_func = get_clap_embeds
    else:
        raise ValueError(f"Unknown metric: {args.metric}")

    # save input and target audio
    torchaudio.save(
        os.path.join(run_dir, "input_audio.wav"),
        input_audio,
        sample_rate,
        backend="soundfile",
    )

    target_audio /= torch.max(torch.abs(target_audio)).clamp(min=1e-8)
    torchaudio.save(
        os.path.join(run_dir, "target_audio.wav"),
        target_audio,
        sample_rate,
        backend="soundfile",
    )

    # plotting
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    if args.algorithm == "autodiff":
        output_audio, param_dict, metrics, fval_history, wopt_history = run_autodiff(
            input_audio,
            target_audio,
            sample_rate,
            model,
            embed_func,
            lr=1e-2,
            n_iters=args.max_iters,
        )
        sigma0 = 1e-3
    elif args.algorithm == "es":
        es_func = run_staged_es if args.staged else run_es

        input_audio = input_audio.unsqueeze(0)
        target_audio = target_audio.unsqueeze(0)

        # run ES
        # sweep sigma value so that we can find the best one
        # sigmas = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
        for sigma0 in [0.33]:
            print(f"Running ES with sigma0 = {sigma0}")
            result = es_func(
                input_audio,
                target_audio,
                sample_rate,
                plugins,
                model,
                embed_func,
                max_iters=args.max_iters,
                popsize=args.popsize,
                w0=w0,
                find_w0=True,
                sigma0=sigma0,
                distance="cosine",
                parallel=args.parallel,
                dropout=args.dropout,
                savepop=args.savepop,
                normalize_stages=args.normalize_stages,
                run_dir=run_dir,
            )

            output_audio = result["output_audio"]
            param_dict = result["param_dict"]
            fopt = result["fopt"]
            fval_history = result["fval_history"]
            wopt_history = result["wopt_history"]

    axs[0].plot(fval_history, label=f"sigma0={sigma0:0.2f}")
    axs[0].set_xlabel("Iteration")
    axs[0].set_ylabel("Distance")
    axs[0].legend()

    # plot the parameters
    # wopt_history = np.array(wopt_history)
    # axs[1].plot(wopt_history, label=f"sigma0={sigma0:0.2f}")
    # axs[1].set_xlabel("Iteration")
    # axs[1].set_ylabel("Parameter value")
    plt.savefig(os.path.join(run_dir, "plot.png"), dpi=300)

    # save audio, parameters, and metrics
    output_audio_filepath = os.path.join(
        run_dir, f"output_audio_sigma={sigma0:0.2f}.wav"
    )

    output_audio /= torch.max(torch.abs(output_audio)).clamp(min=1e-8)
    torchaudio.save(
        output_audio_filepath,
        output_audio.squeeze(0),
        sample_rate,
        backend="soundfile",
    )

    param_filepath = os.path.join(run_dir, f"parameters_sigma={sigma0:0.2f}.json")
    with open(param_filepath, "w") as f:
        json.dump(param_dict, f, indent=4)
