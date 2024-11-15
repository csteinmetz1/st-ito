import os
import yaml
import torch
import torchaudio
import pyloudnorm as pyln

from collections import OrderedDict
from importlib import import_module

# external models
import wav2clip
import laion_clap
from transformers import Wav2Vec2Model
from st_ito.models.beats.beats import BEATs, BEATsConfig

from st_ito.models.fx_encoder import FXencoder
from st_ito.methods.style import StyleTransferSystem

# features
from st_ito.features import (
    compute_lufs,
    compute_rms_energy,
    compute_barkspectrum,
    compute_crest_factor,
    compute_spectral_centroid,
)

# ------------------ Normalization  functions ------------------


def apply_fade_in(x: torch.Tensor, num_samples: int = 16384):
    """Apply fade in to the first num_samples of the audio signal.

    Args:
        x (torch.Tensor): Input audio tensor
        num_samples (int, optional): Number of samples to apply fade in. Defaults to 16384.

    Returns:
        torch.Tensor: Audio tensor with fade in applied
    """
    fade = torch.linspace(0, 1, num_samples, device=x.device)
    x[..., :num_samples] = x[..., :num_samples] * fade
    return x


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


# ------------------- MIR feature extract---------------- #


def load_mir_feature_extractor(use_gpu: bool = False):
    class Model:
        def __init__(self) -> None:
            self.embed_dim = 49

    model = Model()

    return model


def get_mir_feature_embeds(
    x: torch.Tensor, model: torch.nn.Module, sample_rate: float, **kwargs
):

    lufs_feat = compute_lufs(x, sample_rate)
    rms_feat = compute_rms_energy(x)
    crest_feat = compute_crest_factor(x)
    barkspectrum_feat = compute_barkspectrum(x, sample_rate, mode="mono")
    spectral_centroid_feat = compute_spectral_centroid(x, sample_rate)

    # combine features into tensor
    features = {
        "lufs": lufs_feat,
        "rms": rms_feat,
        "crest": crest_feat,
        "barkspectrum": barkspectrum_feat,
        "spectral_centroid": spectral_centroid_feat,
    }
    return features


# ------------------- audio feature (MFCC) extractor ---------------- #


def load_mfcc_feature_extractor(use_gpu: bool = False):
    transform = torchaudio.transforms.MFCC(
        sample_rate=48000,
        n_mfcc=25,
        melkwargs={
            "n_fft": 2048,
            "hop_length": 1024,
            "n_mels": 128,
            "center": False,
        },
    )
    transform.embed_dim = 25 * 3
    if torch.cuda.is_available() and use_gpu:
        transform = transform.cuda()
        transform.device = "cuda"
    return transform


def get_mfcc_feature_embeds(
    x: torch.Tensor,
    model: torch.nn.Module,
    sample_rate: float,
    midside: bool = False,
    **kwargs,
):
    bs, chs, seq_len = x.shape

    # in_device = x.device
    # x = x.to(model.device)

    if sample_rate != 48000:
        x = torchaudio.functional.resample(x, sample_rate, 48000)

    if chs == 2 and midside:  # process as mid-side
        x_mid = x[:, 0, :] + x[:, 1, :]
        x_side = x[:, 0, :] - x[:, 1, :]
        # stack back
        x = torch.stack([x_mid, x_side], dim=1)
    else:
        x = x.mean(dim=1, keepdim=True)

    # Get embeddings
    with torch.no_grad():
        embeddings = model(x)
        embeddings_mean = embeddings.mean(dim=-1)
        embeddings_std = embeddings.std(dim=-1)
        embeddings_max = embeddings.max(dim=-1)[0]
        embeddings = torch.cat(
            [embeddings_mean, embeddings_std, embeddings_max], dim=-1
        )
        embeddings = embeddings.view(bs, -1)

    # l2 normalize
    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=-1)

    embeddings = {
        "mono": embeddings,
    }

    return embeddings


# ------------------- deepafx-st (style transfer) ---------------- #
def load_deepafx_st_model(
    ckpt_path: str, use_gpu: bool = False, encoder_only: bool = False
):
    config_path = os.path.join(os.path.dirname(ckpt_path), "config.yaml")

    with open(config_path) as f:
        config = yaml.safe_load(f)

    encoder_configs = config["model"]["init_args"]["encoder"]

    module_path, class_name = encoder_configs["class_path"].rsplit(".", 1)
    module = import_module(module_path)
    encoder_model = getattr(module, class_name)(**encoder_configs["init_args"])

    model = StyleTransferSystem.load_from_checkpoint(
        ckpt_path,
        vst_json=config["model"]["init_args"]["vst_json"],
        encoder=encoder_model,
        map_location="cpu",
    )
    model.eval()

    if encoder_only:
        model = model.encoder

    if use_gpu:
        model = model.cuda()

    return model


def get_deepafx_st_embeds(
    x: torch.Tensor, model: torch.nn.Module, sample_rate: float, **kwargs
):
    bs, chs, seq_len = x.shape

    x_device = x

    # move audio to model device
    x = x.type_as(next(model.parameters()))

    if sample_rate != 48000:
        x = torchaudio.functional.resample(x, sample_rate, 48000)

    # if stereo -> mono
    if x.shape[1] == 2:
        x = torch.mean(x, dim=1, keepdim=True)

    with torch.no_grad():
        mid_embeddings, side_embeddings = model(x)

    embeddings = {
        "mid": mid_embeddings.type_as(x_device),
        "side": side_embeddings.type_as(x_device),
    }

    return embeddings


# ------------------- BEATs  ---------------- #


def load_beats_model(use_gpu: bool = False):
    # load the pre-trained checkpoints
    ckpt_path = "checkpoints/BEATs_iter3_plus_AS2M.pt"

    if not os.path.isfile(ckpt_path):
        os.system(
            f"""wget -P checkpoints/ "https://valle.blob.core.windows.net/share/BEATs/BEATs_iter3_plus_AS2M.pt?sv=2020-08-04&st=2023-03-01T07%3A51%3A05Z&se=2033-03-02T07%3A51%3A00Z&sr=c&sp=rl&sig=QJXmSJG9DbMKf48UDIU1MfzIro8HQOf3sqlNXiflY1I%3D" """
        )
        os.rename(
            "BEATs_iter3_plus_AS2M.pt?sv=2020-08-04&st=2023-03-01T07%3A51%3A05Z&se=2033-03-02T07%3A51%3A00Z&sr=c&sp=rl&sig=QJXmSJG9DbMKf48UDIU1MfzIro8HQOf3sqlNXiflY1I%3D",
            "BEATs_iter3_plus_AS2M.pt",
        )

    checkpoint = torch.load(ckpt_path, map_location="cpu")

    cfg = BEATsConfig(checkpoint["cfg"])
    BEATs_model = BEATs(cfg)
    BEATs_model.load_state_dict(checkpoint["model"])
    BEATs_model.eval()

    if use_gpu:
        BEATs_model.cuda()

    return BEATs_model


def get_beats_embeds(
    x: torch.Tensor,
    model: torch.nn.Module,
    sample_rate: float,
    **kwargs,
):
    bs, chs, seq_len = x.shape

    x = x.mean(dim=1, keepdim=False)

    # extract the the audio representation
    padding_mask = torch.zeros(1, seq_len).bool().type_as(x)
    model.embed_dim = 768

    with torch.no_grad():
        embed = model.extract_features(x, padding_mask=padding_mask)[0]

    # take mean across time (2nd dim)
    embed = embed.mean(dim=1)

    embeddings = {
        "mono": embed,
    }

    return embeddings


# ------------------- pretrained wav2vec ---------------- #
def load_wav2vec2_model(use_gpu: bool = False):
    model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-large-960h-lv60-self")
    if torch.cuda.is_available() and use_gpu:
        model = model.cuda()
    model.embed_dim = 1024
    model.eval()

    return model


def get_wav2vec2_embeds(
    x: torch.Tensor,
    model: torch.nn.Module,
    sample_rate: float,
    no_grad: bool = False,
    **kwargs,
):
    bs, chs, seq_len = x.shape

    x = x.mean(dim=1, keepdim=True)

    if sample_rate != 16000:
        x = torchaudio.functional.resample(x, sample_rate, 16000)

    with torch.no_grad():
        x = x.squeeze(1)
        output = model(x, output_hidden_states=True)
        out = output.hidden_states[0]
        for i in range(1, len(output.hidden_states)):
            out = out + output.hidden_states[i]
        out = out / len(output.hidden_states)

    embeddings = {
        "mono": out.mean(dim=1),
    }
    return embeddings


# ------------------- pretrained wav2clip ---------------- #


def load_wav2clip_model(use_gpu: bool = False):
    model = wav2clip.get_model()
    if torch.cuda.is_available() and use_gpu:
        model = model.cuda()
    model.eval()
    model.embed_dim = 512
    return model


def get_wav2clip_embeds(
    x: torch.Tensor,
    model: torch.nn.Module,
    sample_rate: float,
    no_grad: bool = False,
    **kwargs,
):
    bs, chs, seq_len = x.shape
    x = x.mean(dim=1, keepdim=True)

    if sample_rate != 16000:
        x = torchaudio.functional.resample(x, sample_rate, 16000)
    with torch.no_grad():
        embeddings = model.forward(x.view(bs, -1).detach())

    embeddings = {
        "mono": embeddings,
    }
    return embeddings


# ------------------- pretrained vggish ---------------- #


def load_vggish_model(use_gpu: bool = False):
    model = torch.hub.load("harritaylor/torchvggish", "vggish")
    if torch.cuda.is_available() and use_gpu:
        model = model.cuda()
    model.eval()
    model.embed_dim = 128
    return model


def get_vggish_embeds(
    x: torch.Tensor,
    model: torch.nn.Module,
    sample_rate: float,
    no_grad: bool = False,
    **kwargs,
):
    bs, chs, seq_len = x.shape
    x = torch.mean(x, dim=1)

    embeddings = []

    for batch_idx in range(bs):
        batch_audio = x[batch_idx, :].detach().cpu().numpy()
        embedding = model.forward(batch_audio, fs=sample_rate)
        embedding = embedding.sum(dim=0, keepdim=True) / embedding.shape[0]
        embeddings.append(embedding)

    embeddings = torch.cat(embeddings, dim=0)
    embeddings = {"mono": embeddings}

    return embeddings


# ------------------- pretrained CLAP model ---------------- #


def load_clap_model(use_gpu: bool = False, midside: bool = False):
    model = laion_clap.CLAP_Module(enable_fusion=False)
    model.load_ckpt()  # download the default pretrained checkpoint.
    model.midside = midside
    if torch.cuda.is_available() and use_gpu:
        model = model.cuda()
        model.device = "cuda"
    model.eval()
    model.embed_dim = 512
    return model


def get_clap_embeds(
    x: torch.Tensor,
    model: torch.nn.Module,
    sample_rate: float,
    midside: bool = False,
    **kwargs,
):
    bs, chs, seq_len = x.shape

    if sample_rate != 48000:
        x = torchaudio.functional.resample(x, sample_rate, 48000)

    # if stereo -> mono
    if x.shape[1] == 2:
        if midside:
            x_mid = x[:, 0, :] + x[:, 1, :]
            x_side = x[:, 0, :] - x[:, 1, :]
            with torch.no_grad():
                embeddings_mid = model.get_audio_embedding_from_data(
                    x=x_mid, use_tensor=True
                )
                embeddings_side = model.get_audio_embedding_from_data(
                    x=x_side, use_tensor=True
                )

            embeddings = {"mid": embeddings_mid, "side": embeddings_side}

        else:  # mono
            x = torch.mean(x, dim=1, keepdim=False)
            with torch.no_grad():
                embeddings = model.get_audio_embedding_from_data(x=x, use_tensor=True)
            embeddings = {"mono": embeddings}
    else:
        with torch.no_grad():
            embeddings = model.get_audio_embedding_from_data(x=x, use_tensor=True)
        embeddings = {"mono": embeddings}

    return embeddings


# -------- self-supervised parameter estimation model -------- #


def get_param_embeds(
    x: torch.Tensor,
    model: torch.nn.Module,
    sample_rate: float,
    requires_grad: bool = False,
    peak_normalize: bool = False,
    dropout: float = 0.0,
):
    bs, chs, seq_len = x.shape

    x_device = x

    # move audio to model device
    x = x.type_as(next(model.parameters()))

    # if peak_normalize:
    #    x = batch_peak_normalize(x)

    if sample_rate != 48000:
        x = torchaudio.functional.resample(x, sample_rate, 48000)

    seq_len = x.shape[-1]  # update seq_len after resampling
    # if longer than 262144 crop, else repeat pad to 262144
    # if seq_len > 262144:
    #    x = x[:, :, :262144]
    # else:
    #    x = torch.nn.functional.pad(x, (0, 262144 - seq_len), "replicate")

    # peak normalize each batch item
    for batch_idx in range(bs):
        x[batch_idx, ...] /= x[batch_idx, ...].abs().max().clamp(1e-8)

    if not requires_grad:
        with torch.no_grad():
            mid_embeddings, side_embeddings = model(x)
    else:
        mid_embeddings, side_embeddings = model(x)

    # add dropout
    if dropout > 0.0:
        mid_embeddings = torch.nn.functional.dropout(
            mid_embeddings, p=dropout, training=True
        )
        side_embeddings = torch.nn.functional.dropout(
            side_embeddings, p=dropout, training=True
        )

    # check for nan
    if torch.isnan(mid_embeddings).any():
        print("Warning: NaNs found in mid_embeddings")
        mid_embeddings = torch.nan_to_num(mid_embeddings)
    elif torch.isnan(side_embeddings).any():
        print("Warning: NaNs found in side_embeddings")
        side_embeddings = torch.nan_to_num(side_embeddings)

    # l2 normalize
    mid_embeddings = torch.nn.functional.normalize(mid_embeddings, p=2, dim=-1)
    side_embeddings = torch.nn.functional.normalize(side_embeddings, p=2, dim=-1)

    embeddings = {
        "mid": mid_embeddings.type_as(x_device),
        "side": side_embeddings.type_as(x_device),
    }

    return embeddings


def load_param_model(ckpt_path: str = None, use_gpu: bool = False):

    if ckpt_path is None:  # look in tmp direcory
        ckpt_path = os.path.join(os.getcwd(), "tmp", "afx-rep.ckpt")
        os.makedirs("tmp", exist_ok=True)
        if not os.path.isfile(ckpt_path):
            # download from huggingfacehub
            os.system(
                "wget -O tmp/afx-rep.ckpt https://huggingface.co/csteinmetz1/afx-rep/resolve/main/afx-rep.ckpt"
            )
            os.system(
                "wget -O tmp/config.yaml https://huggingface.co/csteinmetz1/afx-rep/resolve/main/config.yaml"
            )

    config_path = os.path.join(os.path.dirname(ckpt_path), "config.yaml")

    with open(config_path) as f:
        config = yaml.safe_load(f)

    encoder_configs = config["model"]["init_args"]["encoder"]

    module_path, class_name = encoder_configs["class_path"].rsplit(".", 1)
    module_path = module_path.replace("lcap", "st_ito")
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

    if use_gpu:
        model.cuda()

    return model


def get_fx_encoder_embeds(
    x: torch.Tensor, model: torch.nn.Module, sample_rate: float, **kwargs
):
    bs, chs, seq_len = x.shape

    x_in = x
    x = x.type_as(next(model.parameters()))

    # resample and ensure stereo
    x = torchaudio.functional.resample(x, sample_rate, 44100)

    # peak normalize
    x /= x.abs().max().clamp(1e-8)

    if chs == 1:
        x = x.repeat(1, 2, 1)

    # signal shape should be : [channel, signal duration]
    with torch.no_grad():
        embed = model(x)

    embeddings = {
        "stereo": embed.type_as(x_in),
    }

    return embeddings


def load_fx_encoder_model(use_gpu: bool = False):
    ckpt_path = "checkpoints/FXencoder_ps.pt"
    # look if checkpoint exists
    if not os.path.exists(ckpt_path):
        print(
            f"Checkpoint file not found at {ckpt_path}. Download from https://drive.google.com/file/d/1BFABsJRUVgJS5UE5iuM03dbfBjmI9LT5/view"
        )

    ddp = True
    model = FXencoder()  # default configs
    checkpoint = torch.load(ckpt_path, map_location="cpu")

    new_state_dict = OrderedDict()
    for k, v in checkpoint["model"].items():
        # remove `module.` if the model was trained with DDP
        name = k[7:] if ddp else k
        new_state_dict[name] = v

    # load params
    model.load_state_dict(new_state_dict)
    model.embed_dim = 2048
    model.eval()

    if use_gpu:
        model.cuda()

    return model
