import torch
import pytorch_lightning as pl


def get_activation(act_type, ch=None):
    """Helper function to construct activation functions by a string.

    Args:
        act_type (str): One of 'ReLU', 'PReLU', 'SELU', 'ELU'.
        ch (int, optional): Number of channels to use for PReLU.

    Returns:
        torch.nn.Module activation function.
    """

    if act_type == "PReLU":
        return torch.nn.PReLU(ch)
    elif act_type == "ReLU":
        return torch.nn.ReLU()
    elif act_type == "SELU":
        return torch.nn.SELU()
    elif act_type == "ELU":
        return torch.nn.ELU()


class dsTCNBlock(torch.nn.Module):
    def __init__(
        self,
        in_ch,
        out_ch,
        kernel_size,
        stride=1,
        dilation=1,
        norm_type=None,
        act_type="PReLU",
    ):
        super(dsTCNBlock, self).__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.kernel_size = kernel_size
        self.stride = stride
        self.norm_type = norm_type

        pad_value = ((kernel_size - 1) * dilation) // 2

        self.conv1 = torch.nn.Conv1d(
            in_ch,
            out_ch,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            padding=pad_value,
        )
        self.act1 = get_activation(act_type, out_ch)

        if norm_type == "BatchNorm":
            self.norm1 = torch.nn.BatchNorm1d(out_ch)
            # self.norm2 = torch.nn.BatchNorm1d(out_ch)
            self.res_norm = torch.nn.BatchNorm1d(out_ch)
        else:
            self.norm1 = torch.nn.Identity()
            self.res_norm = torch.nn.Identity()

        self.res_conv = torch.nn.Conv1d(in_ch, out_ch, kernel_size=1, stride=stride)

    def forward(self, x):
        x_res = x  # store input for later

        # -- first section --
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act1(x)

        # -- residual connection --
        x_res = self.res_conv(x_res)
        x_res = self.res_norm(x_res)

        return x + x_res


class dsTCNModel(torch.nn.Module):
    """Downsampling Temporal convolutional network.

    Args:
        ninputs (int): Number of input channels (mono = 1, stereo 2). Default: 1
        nblocks (int): Number of total TCN blocks. Default: 8
        kernel_size (int): Width of the convolutional kernels. Default: 15
        stride (int): Stide size when applying convolutional filter. Default: 2
        dialation_growth (int): Compute the dilation factor at each block as dilation_growth ** (n % stack_size). Default: 8
        channel_growth (int): Compute the output channels at each black as in_ch * channel_growth. Default: 1
        channel_width (int): When channel_growth = 1 all blocks use convolutions with this many channels. Default: 32
        stack_size (int): Number of blocks that constitute a single stack of blocks. Default: 10
        grouped (bool): Use grouped convolutions to reduce the total number of parameters. Default: False
        causal (bool): Causal TCN configuration does not consider future input values. Default: False
        skip_connections (bool): Skip connections from each block to the output. Default: False
        norm_type (str): Type of normalization layer to use 'BatchNorm', 'LayerNorm', 'InstanceNorm'. Default: None
    """

    def __init__(
        self,
        embed_dim: int,
        ninputs: int = 1,
        nblocks: int = 8,
        kernel_size: int = 13,
        stride: int = 4,
        dilation_growth: int = 8,
        channel_growth: int = 2,
        channel_width: int = 32,
        stack_size: int = 4,
        norm_type: str = None,
        act_type: str = "PReLU",
        *args,
        **kwargs,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.nblocks = nblocks
        self.kernel_size = kernel_size
        self.stack_size = stack_size
        self.stride = stride

        self.blocks = torch.nn.ModuleList()
        for n in range(nblocks):
            in_ch = ninputs if n == 0 else out_ch
            out_ch = channel_width if n == 0 else in_ch * channel_growth
            dilation = dilation_growth ** (n % stack_size)
            self.blocks.append(
                dsTCNBlock(
                    in_ch,
                    out_ch,
                    kernel_size,
                    stride,
                    dilation,
                    norm_type,
                    act_type,
                )
            )

        self.output = torch.nn.Conv1d(out_ch, 1, 1)
        self.fc = torch.nn.Linear(out_ch, embed_dim)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)

        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2
        x = torch.nn.functional.dropout(x, p=0.5, training=self.training)
        x = self.fc(x)

        return x
