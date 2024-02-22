# CUDA/C++ extensions in Pytorch

## Compilation and architecture

Our CUDA/C++ extensions are built in this separate python module/library such that they are only built once when the docker container is built. This cannot be done as part of the main `setup.py` as editable installations produce build artifacts in the source tree which are overwritten when the docker container mounts the source tree. If you would like an editable installation of `rnnt_ext` for development you can run `python -m pip install -e lib/` from inside the docker container (you will need to do this every time you restart the container). Furthermore, if you are developing the C++/CUDA source you will need to run the above command every time you edit a `.cu`, `.cpp`, `.h`, `.hpp` or `.cuh` file.

Our extensions share common code via header files in `lib/csrc/myrtle` each extension module is contained within a single source file in `lib/csrc` (like python 1 file = 1 module). You can add a new extension by modifying `lib/setup.py` and adding a new source file to `lib/csrc`.

### IDE/Editor setup

If you want your language server to understand the compilation flags and includes of the C++ code (highly recommended) run:
```bash
bear python lib/setup.py build
```
to generate a `compile_commands.json` file that  can then be used by yor language server to provide IDE functionality.


## Notes on Torch extensions

Pytorch provides a C++ tensor API that is very similar to the python API. You can expose a C++ function that accepts/returns pytorch tensors to python with pybind (see [lstm.cu](csrc/lstm.cu)).

If you are implementing a function that should work with autograd you will need to provide the forward and backward passes, these are joined together in python-land with the `torch.autograd.Function` class (see [lstm.py](src/rnnt_ext/custom_lstm/lstm.py)).


### Useful docs

- [Torch's reference on CUDA extensions](https://pytorch.org/tutorials/advanced/cpp_extension.html)
- [CuBLAS docs](https://docs.nvidia.com/cuda/cublas/index.html)
- [Understanding torch's autograd](https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html)
- [Autograd's API](https://pytorch.org/docs/stable/autograd.html)

### Performance notes

All the standard CUDA performance guidelines apply (fuse kernel invocations, maximize occupance, make the matrix multiplications as large as possible, etc.). Additionally, some torch specific pitfalls were encountered:
- Calling into C++ from inside a loop in python can easily become the bottleneck for small batch sizes.
- Every torch operation in the C++ API has dynamic dispatch (often several layers of dynamic dispatch), which becomes a bottleneck when dispatching small kernels in a tight loop. To illustrate, a loop calling `torch::matmul` requires the matrices to be unexpectedly large before the multiplication becomes the bottleneck (as opposed to the virtual function calls). Furthermore, this also applies to indexing!
- Torch's batched matrix multiply does not do the performant thing for contiguous ND @ 2D (e.g. view ND as 2D).

### Gotchas

- CuBLAS is fortran order (column major) but pytorch is C order (row major).

## Notes on the LSTM implementation

This folder contains a CUDA extension implementing a custom LSTM. Most of the LSTM blogs/references online are very difficult to follow, use unusual notation and are filled with errors.
[This paper](https://doi.org/10.1109/TNNLS.2016.2582924) is the only correct and complete reference Myrtle has found (there is an older preprint on arXiv).
