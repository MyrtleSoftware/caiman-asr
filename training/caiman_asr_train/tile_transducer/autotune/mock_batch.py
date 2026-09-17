from dataclasses import dataclass

import torch
from beartype import beartype

JOINT_LATTICE_CAP = 131_072
MOCK_U = 64
PROBE_CELLS = 1024


@beartype
@dataclass(frozen=True)
class TuneShape:
    cells: int

    def __post_init__(self) -> None:
        assert (
            self.cells > 0 and self.cells % MOCK_U == 0
        ), f"cells={self.cells} must be a positive multiple of {MOCK_U}"

    def __str__(self) -> str:
        return f"C={self.cells} (U={MOCK_U})"


@beartype
def tune_shape() -> TuneShape:
    return TuneShape(cells=JOINT_LATTICE_CAP)


@beartype
def probe_shape() -> TuneShape:
    return TuneShape(cells=PROBE_CELLS)


@beartype
def _label_row(V: int, device: torch.device) -> torch.Tensor:
    return torch.randint(0, V - 1, (1, MOCK_U - 1), device=device, dtype=torch.int32)


@beartype
def _label_cell(Label: torch.Tensor, cells: int) -> torch.Tensor:
    u = torch.arange(MOCK_U, device=Label.device).clamp(max=Label.shape[1] - 1)
    return Label[0, u].repeat(cells // MOCK_U).int()


@beartype
def _joint_inputs(
    shape: TuneShape, H: int, V: int, device: torch.device, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    A = torch.relu(torch.randn(shape.cells, H, device=device, dtype=dtype))
    W = torch.randn(H, V, device=device, dtype=dtype) * (H**-0.5)
    return A, W, _label_row(V, device)


@beartype
def fwd_batch(
    shape: TuneShape,
    H: int,
    V: int,
    device: torch.device,
    seed: int = 0,
    dtype: torch.dtype = torch.float16,
    accum_dtype: torch.dtype = torch.float32,
) -> dict:
    torch.manual_seed(seed)
    cells = shape.cells
    A, W, Label = _joint_inputs(shape, H, V, device, dtype)
    return dict(
        A=A,
        W=W,
        Bias=torch.randn(V, device=device, dtype=dtype),
        Label=Label,
        cell_idx=torch.arange(cells, device=device, dtype=torch.int64),
        label_cell=_label_cell(Label, cells),
        total=cells,
        denom=torch.zeros(cells, device=device, dtype=accum_dtype),
        lbx=torch.zeros(cells, device=device, dtype=accum_dtype),
        lex=torch.zeros(cells, device=device, dtype=accum_dtype),
        X=torch.zeros(cells, V, device=device, dtype=dtype),
    )


@beartype
def bwd_batch(
    shape: TuneShape,
    H: int,
    V: int,
    device: torch.device,
    seed: int = 0,
    dtype: torch.dtype = torch.float16,
    accum_dtype: torch.dtype = torch.float32,
) -> dict:
    torch.manual_seed(seed)
    cells = shape.cells
    A, W, Label = _joint_inputs(shape, H, V, device, dtype)
    return dict(
        A=A,
        W=W,
        total=cells,
        c_dense=torch.randn(cells, device=device, dtype=accum_dtype),
        c_blank=torch.randn(cells, device=device, dtype=accum_dtype),
        c_label=torch.randn(cells, device=device, dtype=accum_dtype),
        denom=torch.full((cells,), 10.0, device=device, dtype=accum_dtype),
        label_cell=_label_cell(Label, cells),
        X=torch.randn(cells, V, device=device, dtype=dtype),
    )
