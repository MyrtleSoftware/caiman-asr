from pathlib import Path

from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


def build_ext(name):
    """
    Must be in setup.py due to __file__
    """

    def localize(r_path):
        return str(Path(__file__).resolve().parent.joinpath(r_path))

    return CUDAExtension(
        name=f"rnnt_ext.cuda.{name}",
        sources=[localize(f"csrc/{name}.cu")],
        include_dirs=[localize("csrc")],  # Ninja requires absolute paths
        extra_compile_args={
            "cxx": ["-O3", "-march=native"],
            "nvcc": ["-O3", "--use_fast_math"],
        },
    )


modules = list(map(build_ext, ["logsumexp", "transducer_loss", "lstm"]))

setup(
    name="rnnt_ext",
    version="0.1.0",
    description="Myrtle.ai's CAIMAN-ASR training module extensions",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    ext_modules=modules,
    cmdclass={"build_ext": BuildExtension},
)
