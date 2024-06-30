# st-ito
Audio production style transfer with inference-time optimization

## Quick Start

Setup
```Bash
git clone 
cd st-ito
pip install -e . 
```

Running style transfer on an audio file
```Bash
python scripts/run_optim.py \
"input.wav" \
--target "target.wav" \
--algorithm es \
--effect-type vst \
--dropout 0.0 \
--max-iters 25 \
--metric param
```

Using AFx-Rep to extract embeddings
```Python
import torch
import torchaudio
from st_ito.utils import load_param_model, get_param_embeds

# load pretrained model
model = load_param_model(use_gpu=True)

# load audio file
audio, sr = torchaudio.load("input.wav")

# audio must be of shape bs, chs, seq_len
audio = audio.unsqueeze(0)

# extract embeddings
embed_dict = get_param_embeds(audio, model, sr)
for embed_name, embed in embed_dict.items():
    print(embed_name, embed.shape)

# mid torch.Size([1, 512])
# side torch.Size([1, 512])
```

## Training

```

```