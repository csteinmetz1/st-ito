import torch
import laion_clap
import torch
import sys

# sys.path.append("VGGish")
# from VGGish.network.vggish import VGGish
# from VGGish.vggish.vggish_input import waveform_to_examples
# import VGGish.vggish.vggish_params as vggish_params
import resampy
import torchaudio
import numpy as np
import wav2clip
from panns_inference import AudioTagging
from transformers import Wav2Vec2Model


class CLAP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = laion_clap.CLAP_Module(enable_fusion=False)
        self.model.load_ckpt()  # download the default pretrained checkpoint.

    def forward(self, x):
        # Remove channel dim
        x = x.squeeze(1, -1)
        embeddings = self.model.get_audio_embedding_from_data(x=x, use_tensor=True)
        return embeddings


class HEAR(torch.nn.Module):
    def __init__(self, model_choice: str):
        super().__init__()
        if model_choice == "hearbaseline":
            from hearbaseline import load_model, get_scene_embeddings

            raise NotImplementedError("Model not implemented")
        self.model = load_model()
        self.get_scene_embeddings = get_scene_embeddings

    def forward(self, x):
        # Remove channel dim
        x = x.squeeze(1)
        embeddings = self.get_scene_embeddings(x, self.model)
        return embeddings


class Wav2Vec2(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = Wav2Vec2Model.from_pretrained(
            "facebook/wav2vec2-large-960h-lv60-self"
        )

    def forward(self, x):
        x = x.squeeze(1)
        output = self.model(x, output_hidden_states=True)
        out = output.hidden_states[0]
        for i in range(1, len(output.hidden_states)):
            out = out + output.hidden_states[i]
        return out / len(output.hidden_states)


class Wav2Clip(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = wav2clip.get_model()

    def forward(self, x):
        # Remove channel dim
        x = x.squeeze(1)
        embeddings = self.model(x)
        return embeddings


class PANNS(torch.nn.Module):
    def __init__(self, device: str = "cpu"):
        super().__init__()
        self.model = AudioTagging(checkpoint_path=None, device=device)
        self.model.model.eval()

    def forward(self, x):
        # Remove channel dim
        x = x.squeeze(1)
        output_dict = self.model.model(x, None)
        embedding = output_dict["embedding"]
        return embedding


class VGGISH(torch.nn.Module):
    def __init__(self, sr: int = 16000):
        super().__init__()
        self.sr = sr
        self.model = VGGish()
        self.model.load_state_dict(torch.load("VGGish/checkpoints/vggish_model.pt"))
        # Remove the last layer
        layers = list(self.model.children())
        layers[-1] = layers[-1][:-1]
        self.model = torch.nn.Sequential(*layers)
        self.linear = torch.nn.Linear(131072, 128)
        n_fft = 2 ** int(
            np.ceil(
                np.log(
                    int(
                        vggish_params.STFT_WINDOW_LENGTH_SECONDS
                        * vggish_params.SAMPLE_RATE
                    )
                )
                / np.log(2.0)
            )
        )
        self.melspec = torchaudio.transforms.MelSpectrogram(
            n_fft=n_fft,
            sample_rate=vggish_params.SAMPLE_RATE,
            win_length=int(
                vggish_params.STFT_WINDOW_LENGTH_SECONDS * vggish_params.SAMPLE_RATE
            ),
            hop_length=int(
                vggish_params.STFT_HOP_LENGTH_SECONDS * vggish_params.SAMPLE_RATE
            ),
            n_mels=vggish_params.NUM_MEL_BINS,
            f_min=vggish_params.MEL_MIN_HZ,
            f_max=vggish_params.MEL_MAX_HZ,
        )

    def forward(self, x):
        bs = x.shape[0]
        # # Remove channel dim
        x = x.view(bs, -1)
        embeddings = []
        for i in range(bs):
            # Original code
            # input_batch = waveform_to_examples(x[i].detach().numpy(), self.sr)
            input_batch = self.waveform_to_examples(x[i])
            input_batch = input_batch.unsqueeze(1)
            embedding_batch_torch = self.model(input_batch)
            embedding_batch_torch = torch.mean(embedding_batch_torch.float(), axis=0)
            embeddings.append(embedding_batch_torch)
        embeddings = torch.stack(embeddings)
        return embeddings

    # Following functions are modified from VGGish to allow for gradiens to pass through
    def waveform_to_examples(self, data):
        """Converts audio waveform into an array of examples for VGGish.
        Args:
        data: np.array of either one dimension (mono) or two dimensions
            (multi-channel, with the outer dimension representing channels).
            Each sample is generally expected to lie in the range [-1.0, +1.0],
            although this is not required.
        sample_rate: Sample rate of data.
        Returns:
        3-D np.array of shape [num_examples, num_frames, num_bands] which represents
        a sequence of examples, each of which contains a patch of log mel
        spectrogram, covering num_frames frames of audio and num_bands mel frequency
        bands, where the frame length is vggish_params.STFT_HOP_LENGTH_SECONDS.
        """
        # Resample to the rate assumed by VGGish.
        if self.sr != vggish_params.SAMPLE_RATE:
            data = resampy.resample(data, self.sr, vggish_params.SAMPLE_RATE)

        # Compute mel spectrogram features.
        mel = self.melspec(data)
        # Convert to log scale.
        log_mel = torch.log(mel + vggish_params.LOG_OFFSET).T
        # Frame features into examples.
        features_sample_rate = 1.0 / vggish_params.STFT_HOP_LENGTH_SECONDS
        example_window_length = int(
            round(vggish_params.EXAMPLE_WINDOW_SECONDS * features_sample_rate)
        )
        example_hop_length = int(
            round(vggish_params.EXAMPLE_HOP_SECONDS * features_sample_rate)
        )
        log_mel_examples = frame(
            log_mel, window_length=example_window_length, hop_length=example_hop_length
        )
        return log_mel_examples


def frame(data, window_length, hop_length):
    """Convert array into a sequence of successive possibly overlapping frames.
    An n-dimensional array of shape (num_samples, ...) is converted into an
    (n+1)-D array of shape (num_frames, window_length, ...), where each frame
    starts hop_length points after the preceding one.
    This is accomplished using stride_tricks, so the original data is not
    copied.  However, there is no zero-padding, so any incomplete frames at the
    end are not included.
    Args:
      data: np.array of dimension N >= 1.
      window_length: Number of samples in each frame.
      hop_length: Advance (in samples) between each window.
    Returns:
      (N+1)-D np.array with as many rows as there are complete frames that can be
      extracted.
    """
    num_samples = data.shape[0]
    num_frames = 1 + int(np.floor((num_samples - window_length) / hop_length))
    shape = (num_frames, window_length) + data.shape[1:]
    strides = (data.stride()[0] * hop_length,) + data.stride()
    return torch.as_strided(data, size=shape, stride=strides)
