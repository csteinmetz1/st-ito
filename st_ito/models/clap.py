import torch
import laion_clap
import torchaudio


class CLAP(torch.nn.Module):
    def __init__(self, pretrained: bool = False, frozen: bool = False) -> None:
        super().__init__()
        self.pretrained = pretrained
        self.frozen = frozen

        # load the model
        self.encoder = laion_clap.CLAP_Module(enable_fusion=False)
        if self.pretrained:
            self.encoder.load_ckpt()  # download the default pretrained checkpoint.

        self.embed_dim = 512

    def forward(self, x: torch.Tensor):
        """ """
        bs, chs, seq_len = x.size()

        x_mid = (x[:, 0, :] + x[:, 1, :]) / 2
        x_side = (x[:, 0, :] - x[:, 1, :]) / 2

        if self.frozen:
            with torch.no_grad():
                embed_mid = self.encoder.get_audio_embedding_from_data(
                    x=x_mid, use_tensor=True
                )
                embed_side = self.encoder.get_audio_embedding_from_data(
                    x=x_side, use_tensor=True
                )
        else:
            embed_mid = self.encoder.get_audio_embedding_from_data(
                x=x_mid, use_tensor=True
            )
            embed_side = self.encoder.get_audio_embedding_from_data(
                x=x_side, use_tensor=True
            )

        return embed_mid, embed_side
