import json
import torch
import auraloss
import pedalboard
import torchaudio
import numpy as np
import dasp_pytorch
import pytorch_lightning as pl

from typing import Any, List


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

    gain_db = denormalize(params[:, 0], -48, 48)

    if chs == 2:
        gain_db = gain_db.unsqueeze(1).repeat(1, 2)

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
    assert num_params == 15, "Number of parameters must be 15"

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

    high_shelf_gain_db = denormalize(params[:, 12], -18, 18)
    high_shelf_cutoff_freq = denormalize(params[:, 13], 20, 20000)
    high_shelf_q_factor = denormalize(params[:, 14], 0.1, 10)

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

    num_parametric_eq_params = 15
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

    audio = apply_parametric_eq(audio, parametric_eq_params, sample_rate)
    audio = apply_compressor(audio, compressor_params, sample_rate)
    audio = apply_distortion(audio, distortion_params, sample_rate)
    audio = apply_reverb(audio, reverb_params, sample_rate)
    audio = apply_gain(audio, gain_params, sample_rate)

    return audio


def parameter_tensor_to_dict(params: torch.Tensor):
    pass


def rademacher(size):
    """Generates random samples from a Rademacher distribution +-1

    Args:
        size (int): Number of samples.

    Returns:
        torch.Tensor: Tensor of shape (size,).

    """
    m = torch.distributions.binomial.Binomial(1, 0.5)
    x = m.sample(size)
    x[x == 0] = -1
    return x


class SPSAProcessor(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        audio: torch.Tensor,
        params: torch.Tensor,
        sample_rate: int,
        plugins: List[dict],
        epsilon: float = 0.01,
    ):
        ctx.save_for_backward(audio.cpu(), params.cpu())
        ctx.sample_rate = sample_rate
        ctx.plugins = plugins
        ctx.epsilon = epsilon

        # process audio
        output = process_audio(
            audio.detach().cpu(),
            params.detach().cpu(),
            sample_rate,
            plugins,
        )

        # precompute the random perturbation
        delta_k = rademacher(params.shape)
        ctx.delta_k = delta_k

        # run the processor with the positive perturbation
        params_pos = (params.cpu() + epsilon * delta_k).clamp(0, 1)
        output_pos = process_audio(
            audio.detach().cpu(), params_pos, sample_rate, plugins
        )
        ctx.output_pos = output_pos

        # run the processor with the negative perturbation
        params_neg = (params.cpu() - epsilon * delta_k).clamp(0, 1)
        output_neg = process_audio(
            audio.detach().cpu(), params_neg, sample_rate, plugins
        )
        ctx.output_neg = output_neg

        return output.type_as(audio)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        audio, params = ctx.saved_tensors
        plugins = ctx.plugins
        sample_rate = ctx.sample_rate
        epsilon = ctx.epsilon
        output_pos = ctx.output_pos
        output_neg = ctx.output_neg
        delta_k = ctx.delta_k

        grads_input = None
        grads_params = None

        needs_audio_grad = ctx.needs_input_grad[0]
        needs_param_grad = ctx.needs_input_grad[1]

        if needs_audio_grad:
            raise NotImplementedError("Input gradient not implemented.")

        # only estimate grads for parameters
        if needs_param_grad:
            grad_param = (output_pos - output_neg).mean(dim=-1)

            params_list = []
            # compute the gradient by iterating over each parameter
            for pidx in range(params.shape[1]):
                grad_p = grad_param / (2 * epsilon * delta_k[:, pidx : pidx + 1])
                grad_p_sum = (
                    (grad_output * grad_p.type_as(grad_output).unsqueeze(-1))
                    .sum(dim=-1)
                    .sum(dim=-1)
                )
                params_list.append(grad_p_sum)

            grads_params = torch.stack(params_list, dim=-1)

        return grads_input, grads_params, None, None


def process_audio(
    input_audio: torch.Tensor,
    params: torch.Tensor,
    sample_rate: int,
    plugins: dict,
):
    """Process audio with plugins and provided parameters on [0, 1].

    Args:
        input_audio (torch.Tensor): Audio tensor of shape (batch_size, 1, num_samples)
        params (torch.Tensor): Parameter tensor of shape (batch_size, num_params)
        sample_rate (int): Sample rate
        plugins (dict): Dictionary of plugins

    Returns:
        torch.Tensor: Processed audio tensor of shape (batch_size, 1, num_samples)

    """
    bs, chs, seq_len = input_audio.shape

    assert chs == 1, "Only mono audio is supported."

    input_audio = input_audio[:, 0:1, :]  # convert to mono

    # move to cpu
    input_audio_cpu = input_audio.detach().cpu().numpy()
    params_cpu = params.detach().cpu()

    # allocate output
    output_audio_numpy = np.zeros((bs, seq_len))

    # iterate over batch
    for bidx in range(bs):
        pidx = 0  # reset parameter index
        x = input_audio_cpu[bidx, :]  # get the audio for this batch
        # set the plugin parameters
        for plugin_name, plugin in plugins.items():
            for name, parameter in plugin["instance"].parameters.items():
                if name in plugin["fixed_parameters"]:
                    parameter.raw_value = plugin["fixed_parameters"][name]
                    pidx += 1
                else:
                    parameter.raw_value = params_cpu[bidx, pidx].item()
                    pidx += 1

            # double if stereo
            if plugin["num_channels"] == 2:
                x = np.concatenate((x, x), axis=0)

            # process audio
            x = plugin["instance"].process(x, sample_rate=sample_rate)

            # convert to mono
            if plugin["num_channels"] == 2:
                x = x[0:1, :]

        output_audio_numpy[bidx, :] = x[0, :]  # store the output

    output_audio_cpu = torch.from_numpy(output_audio_numpy)  # convert back to tensor
    output_audio = output_audio_cpu.type_as(input_audio)  # move back to device
    output_audio = output_audio.unsqueeze(1)  # add channel dim back

    return output_audio


class ParameterRegressor(torch.nn.Module):
    def __init__(self, input_dim: int, num_params: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.num_params = num_params
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 2 * input_dim),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(2 * input_dim, num_params),
            torch.nn.Sigmoid(),
        )

    def forward(self, embed: torch.Tensor) -> None:
        return self.net(embed)


class ParameterClassifier(torch.nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_params: int,
        hidden_dim: int = 256,
        num_bins: int = 64,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.num_params = num_params
        self.hidden_dim = hidden_dim
        self.num_bins = num_bins
        self.nets = torch.nn.ModuleList()
        param_vals = torch.linspace(0, 1, num_bins)
        self.register_buffer("param_vals", param_vals)
        for _ in range(num_params):
            self.nets.append(
                torch.nn.Sequential(
                    torch.nn.Linear(input_dim, hidden_dim),
                    torch.nn.ReLU(inplace=True),
                    torch.nn.Linear(hidden_dim, num_bins),
                )
            )

    def params_to_onehot(self, params: torch.Tensor) -> torch.Tensor:
        params_onehot = torch.zeros(
            params.shape[0], self.num_params, self.num_bins
        ).type_as(params)
        for pidx in range(self.num_params):
            param = params[:, pidx]
            bin_idx = torch.searchsorted(self.param_vals, param)
            params_onehot[:, pidx, bin_idx] = 1.0
        return params_onehot

    def params_to_index(self, params: torch.Tensor) -> torch.Tensor:
        params_index = torch.zeros(params.shape[0], self.num_params).type_as(params)
        for pidx in range(self.num_params):
            param = params[:, pidx]
            bin_idx = torch.searchsorted(self.param_vals, param)
            params_index[:, pidx] = bin_idx
        return params_index.long()

    def index_to_params(self, params_index: torch.Tensor) -> torch.Tensor:
        params = torch.zeros(params_index.shape[0], self.num_params)
        for pidx in range(self.num_params):
            bin_idx = params_index[:, pidx]
            param_val = torch.tensor([self.param_vals[bidx] for bidx in bin_idx])
            params[:, pidx] = param_val
        return params

    def logits_to_params(self, param_logits: torch.Tensor) -> torch.Tensor:
        """Convert logits to parameters.

        Args:
            param_logits (torch.Tensor): Tensor of shape (batch_size, num_params, num_bins)

        Returns:
            torch.Tensor: Tensor of shape (batch_size, num_params)
        """
        params = torch.zeros(param_logits.shape[0], self.num_params)
        for pidx in range(self.num_params):
            bin_idx = torch.argmax(param_logits[:, pidx, :], dim=-1)
            param_val = torch.tensor([self.param_vals[bidx] for bidx in bin_idx])
            params[:, pidx] = param_val
        return params

    def forward(self, embed: torch.Tensor) -> None:
        params = torch.zeros(embed.shape[0], self.num_params)
        params_logits = torch.zeros(
            embed.shape[0], self.num_params, self.num_bins
        ).type_as(embed)
        for pidx, net in enumerate(self.nets):
            param_logits = net(embed)  # make prediction
            params_logits[:, pidx, :] = param_logits

        # convert logits to parameters
        params = self.logits_to_params(params_logits)

        return params, params_logits


class StyleTransferSystem(pl.LightningModule):
    def __init__(
        self,
        vst_json: str,
        encoder: torch.nn.Module,
        lr: float,
        analysis_length: int = 131072,
        weight_decay: float = 0.001,
        max_epochs: int = 250,
        loss_type: str = "parameter-regression",
        grad_type: str = "autodiff",
        autodiff_processor: str = "simple",
        on_the_fly: bool = False,
        split_section: bool = False,
        num_bins: int = 64,
    ):
        """

        Args:
            vst_json (str): JSON file specifying the VST effect chain.
            encoder (torch.nn.Module): Encoder model
            lr (float): Learning rate
            analysis_length (int, optional): Length of audio. Defaults to 131072.
            weight_decay (float): Weight decay
            max_epochs (int, optional): Maximum number of epochs. Defaults to 500.
            loss_type (str, optional): Loss type. Choices are "parameter-regression", "parameter-classification", or "audio". Defaults to "parameter".
            grad_type (str, optional): Gradient type. Choices are "spsa" or "autodiff". Defaults to "autodiff".
            autodiff_processor (str, optional): Autodiff processor. Choices are "simple" or "complex". Defaults to "simple".
            on_the_fly (bool, optional): Whether to use on-the-fly processing. Defaults to False.
            split_section (bool, optional): Whether to split the audio into different sections. Defaults to False.
        """
        super().__init__()
        self.save_hyperparameters(ignore="encoder")

        # load the json file specifying the VST effect chain
        with open(vst_json, "r") as fp:
            plugins = json.load(fp)

        if grad_type == "autodiff" and loss_type == "audio":
            print(
                "Skipping loading of VSTs for audio loss when using autodiff effects."
            )
            if self.hparams.autodiff_processor == "simple":
                total_num_params = 21
            elif self.hparams.autodiff_processor == "complex":
                total_num_params = 48
            else:
                raise ValueError(
                    f"Invalid autodiff_processor {self.hparams.autodiff_processor}"
                )
        else:
            total_num_params = 0
            for plugin_name, plugin in plugins.items():
                plugin_instance = pedalboard.load_plugin(plugin["vst_filepath"])
                num_params = 0
                for name, parameter in plugin_instance.parameters.items():
                    num_params += 1
                    print(f"{plugin_name}: {name} = {parameter.raw_value}")
                print()

                plugin["num_params"] = num_params
                plugin["instance"] = plugin_instance
                total_num_params += num_params

        self.plugins = plugins
        self.total_num_params = total_num_params
        self.sample_rate = 48000

        # use multiresolution stft loss
        self.mrstft_loss = auraloss.freq.MultiResolutionSTFTLoss()

        # Base model f(.)
        self.encoder = encoder
        input_dim = self.encoder.embed_dim * 4

        if self.hparams.loss_type == "parameter-regression":
            self.parameter_estimator = ParameterRegressor(input_dim, total_num_params)
            self.process_fn = process_audio
        elif self.hparams.loss_type == "parameter-classification":
            self.parameter_estimator = ParameterClassifier(
                input_dim, total_num_params, num_bins=self.hparams.num_bins
            )
            self.process_fn = process_audio
        elif self.hparams.loss_type == "audio" and self.hparams.grad_type == "spsa":
            self.parameter_estimator = ParameterRegressor(input_dim, total_num_params)
            self.process_fn = SPSAProcessor.apply
        elif self.hparams.loss_type == "audio" and self.hparams.grad_type == "autodiff":
            self.parameter_estimator = ParameterRegressor(input_dim, total_num_params)
            if self.hparams.autodiff_processor == "simple":
                self.process_fn = apply_simple_autodiff_processor
            elif self.hparams.autodiff_processor == "complex":
                self.process_fn = apply_complex_autodiff_processor
            else:
                raise ValueError(
                    f"Invalid autodiff_processor {self.hparams.autodiff_processor}"
                )
        else:
            raise ValueError(f"Invalid loss_type {self.hparams.loss_type}")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        ms1 = int(self.hparams.max_epochs * 0.8)
        ms2 = int(self.hparams.max_epochs * 0.95)
        print(
            "Learning rate schedule:",
            f"0 {self.hparams.lr:0.2e} -> ",
            f"{ms1} {self.hparams.lr*0.1:0.2e} -> ",
            f"{ms2} {self.hparams.lr*0.01:0.2e}",
        )
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[ms1, ms2],
            gamma=0.1,
        )
        return [optimizer], [lr_scheduler]

    def predict_params(self, input_audio: torch.Tensor, target_audio: torch.Tensor):
        bs, chs, seq_len = target_audio.shape
        assert chs == 2, "Target audio must be stereo."
        bs, chs, seq_len = input_audio.shape
        assert chs == 2, "Input audio must be stereo."

        start_idx = (seq_len - self.hparams.analysis_length) // 2
        end_idx = start_idx + self.hparams.analysis_length

        # crop audio to proper length
        if input_audio.shape[-1] > self.hparams.analysis_length:
            input_audio = input_audio[:, :, start_idx:end_idx]
        elif target_audio.shape[-1] > self.hparams.analysis_length:
            target_audio = target_audio[:, :, start_idx:end_idx]

        input_mid_embeds, input_side_embeds = self.encoder(input_audio)
        target_mid_embeds, target_side_embeds = self.encoder(target_audio)

        # concatenate input and target embeddings
        final_feats = torch.cat(
            (
                input_mid_embeds,
                input_side_embeds,
                target_mid_embeds,
                target_side_embeds,
            ),
            dim=-1,
        )

        # predict parameters
        if self.hparams.loss_type == "parameter-regression":
            params = self.parameter_estimator(final_feats)
            params_logits = None
        elif self.hparams.loss_type == "parameter-classification":
            params, params_logits = self.parameter_estimator(final_feats)
        elif self.hparams.loss_type == "audio":
            params = self.parameter_estimator(final_feats)
            params_logits = None

        return params, params_logits

    def forward(
        self,
        input_audio: torch.Tensor,
        target_audio: torch.Tensor,
        sample_rate: int = 48000,
        render_audio: bool = False,
    ):
        bs, chs, seq_len = target_audio.size()
        assert chs == 2, "Target audio must be stereo."
        bs, chs, seq_len = input_audio.size()
        assert chs == 2, "Input audio must be stereo."

        params, params_logits = self.predict_params(input_audio.clone(), target_audio)

        if render_audio:
            output_audio = self.process_fn(
                input_audio, params, sample_rate, self.plugins
            )
        else:
            output_audio = torch.zeros_like(input_audio)

        return output_audio, params, params_logits

    def common_step(self, batch, batch_idx: int, mode="train"):
        input_audio, target_audio, target_params = batch
        bs, chs, seq_len = input_audio.shape

        # peak normalize input
        input_audio /= input_audio.abs().max().clamp(1e-8)

        # apply random gain to input
        input_gain_db = -(torch.rand(bs) * 12).type_as(input_audio)
        input_gain_lin = 10 ** (input_gain_db / 20)
        input_audio *= input_gain_lin.view(bs, 1, 1)

        if self.hparams.on_the_fly:
            with torch.no_grad():
                target_params = torch.rand(bs, self.total_num_params).type_as(
                    input_audio
                )
                # hard code the gain parameter (the last one) to 0.5 (0 dB)
                target_params[:, -1] = 0.5

                # sometimes randomly bypass the reverb by setting mix to 0
                reverb_mix_params = torch.where(torch.rand(bs) > 0.5, 0.0, 1.0).type_as(
                    target_params
                )
                target_params[:, -2] *= reverb_mix_params

                # apply random gain to input audio
                input_audio /= input_audio.abs().max().clamp(1e-8)
                gain_db = -(torch.rand(bs) * 24).type_as(input_audio)
                gain_lin = 10 ** (gain_db / 20)
                input_audio *= gain_lin.view(bs, 1, 1)

                target_audio = self.process_fn(
                    input_audio, target_params, self.sample_rate, self.plugins
                )

                # peak normalize within -48 dB
                peaks = torch.max(torch.abs(target_audio), dim=-1).values
                gain_lin = 1 / peaks
                gain_db = 20 * torch.log10(gain_lin)
                gain_db = gain_db.clamp(-48, 48)
                gain_lin = 10 ** (gain_db / 20)
                gain_lin = gain_lin.unsqueeze(-1)
                target_audio *= gain_lin

        if input_audio.shape[1] == 1:
            input_audio = torch.cat((input_audio, input_audio), dim=1)

        if target_audio.shape[1] == 1:
            target_audio = torch.cat((target_audio, target_audio), dim=1)

        target_gain_db = -(torch.rand(bs) * 12).type_as(target_audio)
        target_gain_lin = 10 ** (target_gain_db / 20)
        target_audio *= target_gain_lin.view(bs, 1, 1)

        # section splitting augmentation
        if self.hparams.split_section:
            # split into A and B sections
            input_audio_A = input_audio[:, :, : seq_len // 2]
            input_audio_B = input_audio[:, :, seq_len // 2 :]

            target_audio_A = target_audio[:, :, : seq_len // 2]
            target_audio_B = target_audio[:, :, seq_len // 2 :]
        else:
            input_audio_A = input_audio
            input_audio_B = input_audio

            target_audio_A = target_audio
            target_audio_B = target_audio

        # only render audio if needed
        if self.hparams.loss_type == "audio":
            render_audio = True
        elif (
            self.hparams.loss_type == "parameter-regression"
            or self.hparams.loss_type == "parameter-classification"
        ) and mode == "val":
            render_audio = True
        else:
            render_audio = False

        # run the model and get predicted parameters
        output_audio_A, params, params_logits = self(
            input_audio_A,
            target_audio_B,
            self.sample_rate,
            render_audio=render_audio,
        )

        loss = 0

        if self.hparams.loss_type == "audio":
            audio_loss = self.mrstft_loss(output_audio_A, target_audio_A)
            loss += audio_loss

            self.log(
                f"{mode}_audio_loss",
                audio_loss,
                on_step=True,
                on_epoch=True,
                sync_dist=True,
            )

            if self.hparams.grad_type == "spsa":
                param_loss = torch.nn.functional.mse_loss(params, target_params)
            else:
                param_loss = 0

        elif self.hparams.loss_type == "parameter-regression":
            param_loss = torch.nn.functional.mse_loss(params, target_params)
            loss += param_loss
        elif self.hparams.loss_type == "parameter-classification":
            # convert target params to one-hot
            target_params_index = self.parameter_estimator.params_to_index(
                target_params
            )
            # cross entropy loss expects shape: bs, num_bins, num_params
            params_logits = params_logits.permute(0, 2, 1)
            param_loss = torch.nn.functional.cross_entropy(
                params_logits, target_params_index
            )
            loss += param_loss

        if self.hparams.loss_type != "audio" and mode == "val":
            # compute audio loss
            audio_loss = self.mrstft_loss(output_audio_A, target_audio_A)
            self.log(
                f"{mode}_audio_loss",
                audio_loss,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )

        self.log(
            f"{mode}_param_loss",
            param_loss,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
        )

        self.log(
            f"{mode}_loss",
            loss,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
        )

        # store in dict and move to cpu
        data_dict = {
            "input_audio_A": input_audio_A.detach().cpu(),
            "input_audio_B": input_audio_B.detach().cpu(),
            "target_audio_A": target_audio_A.detach().cpu(),
            "target_audio_B": target_audio_B.detach().cpu(),
            "output_audio_A": output_audio_A.detach().cpu(),
            "params": params.detach().cpu(),
        }

        return loss, data_dict

    def training_step(self, batch, batch_idx):
        loss, _ = self.common_step(batch, batch_idx, mode="train")
        return loss

    def validation_step(self, batch, batch_idx):
        loss, data_dict = self.common_step(batch, batch_idx, mode="val")
        return data_dict
