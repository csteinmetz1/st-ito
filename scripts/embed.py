import torch
import torchaudio
from st_ito.utils import load_param_model, get_param_embeds

# load pretrained model
model = load_param_model(use_gpu=True)

# load audio file
# audio, sr = torchaudio.load("input.wav")
audio = torch.randn(2, 48000)
sr = 48000

# audio must be of shape bs, chs, seq_len
audio = audio.unsqueeze(0)
print(audio.shape)

# extract embeddings
embed_dict = get_param_embeds(audio, model, sr)
for embed_name, embed in embed_dict.items():
    print(embed_name, embed.shape)
