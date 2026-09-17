import torch
from beartype import beartype

_FREQ_AXIS = 1  # Frequency axis in the spectrogram tensor
_TIME_AXIS = 2  # Time axis in the spectrogram tensor


@beartype
class SpecAugment(torch.nn.Module):
    def __init__(
        self,
        freq_masks: int = 0,
        freq_width: int = 10,
        time_masks: int = 0,
        time_width: int | float = 10,
    ):
        """
        Regularize by masking entire time steps/frequency bands.

        Implements SpecAugment (https://arxiv.org/abs/1904.08779) with adaptive
        masking (https://arxiv.org/abs/1912.05533), without time warping.

        Args:
            freq_masks (int): number of masks for frequency bands
            max_freq (int): maximum number of frequencies in a single mask
            time_masks (int or float): number of masks or adaptive percentage
            max_time (int or float): maximum number of masked time steps per mask,
                value 0 < 1 then denotes adaptive percentage
        """
        super(SpecAugment, self).__init__()

        self.freq_masks = freq_masks
        self.freq_width = freq_width

        self.time_masks = time_masks
        self.time_width = time_width

    def forward(self, x: torch.Tensor, x_lens: torch.Tensor) -> torch.Tensor:
        """
        x : (B, Freq, Time) -> (B, Freq, Time)
        """

        if self.time_masks > 0:
            x = self._apply_masks(
                x=x,
                num_masks=self.time_masks,
                x_lens=x_lens,
                width=self.time_width,
                axis=_TIME_AXIS,
            )

        if self.freq_masks > 0:
            x = self._apply_masks(
                x=x,
                num_masks=self.freq_masks,
                x_lens=x_lens,
                width=self.freq_width,
                axis=_FREQ_AXIS,
            )

        return x

    def _apply_masks(
        self,
        x: torch.Tensor,
        num_masks: int,
        x_lens: torch.Tensor,
        width: int | float,
        axis: int,
    ) -> torch.Tensor:
        """
        This function applies SpecAugment masks to the input tensor `x` along
        the specified axis. The masks are independently generated for each
        batch item and applied in parallel.

        For full details see original implementation:

        https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/asr/parts/submodules/spectr_augment.py

        Args:
            x: (Batch, Freq, Time) tensor of spectrograms
            x_lens: (Batch,) tensor of lengths of the spectrograms
            num_masks: Number of masks to apply.
            axis: Axis along which to apply the masks, either FREQ_AXIS or TIME_AXIS.
            width: Maximum width of a mask either a number/fraction of time-steps if
                axis is TIME_AXIS, or number of frequency bands if axis is FREQ_AXIS.
        """

        if axis not in (_FREQ_AXIS, _TIME_AXIS):
            raise ValueError(
                f"Axis can be ({_FREQ_AXIS}) or ({_TIME_AXIS}). Received: {axis=}"
            )

        assert not (
            isinstance(width, float) and axis == _FREQ_AXIS
        ), "Float width supported only with time axis."

        batch_size = x.shape[0]
        axis_length = x.shape[axis]

        # Help the type checker
        width: int | float | torch.Tensor = width

        # If width is float then it is transformed into a tensor
        if axis == _TIME_AXIS and isinstance(width, float):
            width = torch.clamp(width * x_lens, max=axis_length).unsqueeze(1)

        # Generate [0-1) random numbers and then scale the tensors.
        # Use float32 dtype for begin/end mask markers before they are quantized to long.
        mask_width = width * torch.rand(
            (batch_size, num_masks),
            device=x.device,
            dtype=torch.float32,
        )
        mask_width = mask_width.long()
        mask_start = torch.rand(
            (batch_size, num_masks), device=x.device, dtype=torch.float32
        )

        if axis == _TIME_AXIS:
            # length can only be used for the time axis
            mask_start = mask_start * (x_lens.unsqueeze(1) - mask_width)
        else:
            mask_start = mask_start * (axis_length - mask_width)

        mask_start = mask_start.long()
        mask_end = mask_start + mask_width

        # Create mask values using vectorized indexing
        indices = torch.arange(axis_length, device=x.device)
        # Create a mask_tensor with all the indices.
        # The mask_tensor shape is (batch_size, num_masks, axis_length).
        mask_tensor = (indices >= mask_start.unsqueeze(-1)) & (
            indices < mask_end.unsqueeze(-1)
        )

        # Reduce masks to one mask
        mask_tensor = mask_tensor.any(dim=1)

        # Create a final mask that aligns with the full tensor
        mask = torch.zeros_like(x, dtype=torch.bool)
        if axis == _TIME_AXIS:
            mask_ranges = mask_tensor[:, None, :]
        else:  # axis == self.FREQ_AXIS
            mask_ranges = mask_tensor[:, :, None]
        mask[:, :, :] = mask_ranges

        # Apply the mask value
        return x.masked_fill(mask=mask, value=0.0)
