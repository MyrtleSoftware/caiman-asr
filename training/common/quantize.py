# Copyright (c) 2022 Myrtle.ai
# iria

import torch
from qtorch import FloatingPoint, BlockFloatingPoint
from qtorch.quant import Quantizer # this import is _very_ slow when it is first run

class NullClass(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, in_tensor):
        return in_tensor


class BrainFloatQuantizer(torch.nn.Module):
    def __init__(self,
                fp_exp,
                fp_man,
                forward_rounding="nearest"):
        super().__init__()
        self.fp_exp = fp_exp
        self.fp_man = fp_man
        self.float_quantizer = Quantizer(
            forward_number=FloatingPoint(exp=fp_exp, man=fp_man),
            forward_rounding=forward_rounding,
        )

    def forward(self, input):
        bf_input = self.float_quantizer(input)
        return bf_input


class BfpQuantizer(torch.nn.Module):
    """Wrapper for a Block Floating Point quantizer for layers.

    This class helps to apply the Block Floating Point quantization to the specified
    dimension of the inputs and weights of the layers.

    Args:
        dim: Dimension where to compute the blocks.
        block_size: Size of each block.
        fp_exp: Number of exponent bits, used when converting the inputs to
                floating point before applying the bfp quantization.
        fp_man: Number of mantissa bits, used when converting the inputs to
                floating point before applying the bfp quantization.
        forward_rounding: Whether to use nearest or stochastic rounding.
    """

    def __init__(
        self,
        dim,
        block_size,
        fp_exp,
        fp_man,
        forward_rounding="nearest",
    ):
        super(BfpQuantizer, self).__init__()
        self.dim = dim
        self.block_size = block_size
        self.float_quantizer = Quantizer(
            forward_number=FloatingPoint(exp=fp_exp, man=fp_man),
            forward_rounding=forward_rounding,
        )
        self.block_quantizer = Quantizer(
            forward_number=BlockFloatingPoint(
                wl=fp_man + 1,
                dim=0, # we want to block quantize along same dimension of tensors
            ),
            forward_rounding=forward_rounding,
        )

    def forward(self, input):
        """Convert input to Block Floating Point 16.

        The blocks are calculated over the `dim` dimension specified during
        initialisation.

        """
        # Set the specified `dim` as the last dimension
        transp_input = torch.transpose(input, self.dim, -1)
        reshaped_input = torch.reshape(
            transp_input, (-1, transp_input.shape[-1])
        )

        # Pad input if necessary
        if reshaped_input.shape[-1] % self.block_size > 0:
            # Pad over the `dim` dimension
            pad = self.block_size - (
                reshaped_input.shape[-1] % self.block_size
            )
            # Get the minimum values of the last padded blocks
            padded_blocks = reshaped_input[:, -(self.block_size - pad) :]
            min_values, _ = torch.min(padded_blocks, dim=1)
            pad_tensor = (
                min_values.unsqueeze(1).repeat(1, pad).to(input.device)
            )
            # Pad input with minimum values of the last padded blocks
            padded_input = torch.cat((reshaped_input, pad_tensor), dim=1)
        else:
            pad = 0
            padded_input = reshaped_input

        # Reshape to create blocks of the specified size
        blocks_input = torch.reshape(padded_input, (-1, self.block_size))
        # Quantize input to fp and then to bfp
        fp_input = self.float_quantizer(blocks_input)
        bfp_input = self.block_quantizer(fp_input)
        # Reshape to the padded input shape
        bfp_input = torch.reshape(bfp_input, padded_input.shape)
        # Remove pad if necessary
        if pad > 0:
            bfp_input = bfp_input[:, :-pad]
        # Reshape to the initial shape
        bfp_input = torch.reshape(bfp_input, transp_input.shape)
        bfp_input = torch.transpose(bfp_input, self.dim, -1)

        return bfp_input

