import ast
import re
import os
import subprocess
import setuptools

from pathlib import Path
import paddle
from paddle.utils.cpp_extension import BuildExtension, CUDAExtension

current_dir = os.path.dirname(os.path.realpath(__file__))


def get_package_version():
    with open(Path(current_dir) / "ultra_ep" / "__init__.py", "r") as f:
        version_match = re.search(r"^__version__\s*=\s*(.*)$", f.read(), re.MULTILINE)
    public_version = ast.literal_eval(version_match.group(1))

    # noinspection PyBroadException
    try:
        status_cmd = ["git", "status", "--porcelain"]
        status_output = subprocess.check_output(status_cmd).decode("ascii").strip()
        if status_output:
            print(
                f"Warning: Git working directory is not clean. Uncommitted changes:\n{status_output}"
            )

        cmd = ["git", "rev-parse", "--short", "HEAD"]
        revision = "+" + subprocess.check_output(cmd).decode("ascii").rstrip()
    except:
        revision = "+local"
    return f"{public_version}{revision}"


def find_cpp_cuda_sources(root_dir="csrc"):
    return [
        "csrc/python_api.cpp",
        "csrc/kernels/reroute.cu",
        "csrc/kernels/placement.cu",
        "csrc/kernels/legacy/placement_cpu.cpp",
    ]


if __name__ == "__main__":
    cxx_flags = [
        "-O3",
        "-Wno-deprecated-declarations",
        "-Wno-unused-variable",
        "-Wno-sign-compare",
        "-Wno-reorder",
        "-Wno-attributes",
    ]
    nvcc_flags = ["-O3", "-Xcompiler", "-O3", "-rdc=true"]

    sources = find_cpp_cuda_sources("csrc")
    include_dirs = ["csrc/"]

    nvcc_flags.extend(["--ptxas-options=--register-usage-level=10"])
    # Compilation workaround for CUDA 13.x
    if paddle.version.cuda and paddle.version.cuda.startswith("13."):
        include_dirs.extend(["/usr/local/cuda/include/cccl/"])

    # Disable aggressive PTX instructions
    if int(os.getenv("DISABLE_AGGRESSIVE_PTX_INSTRS", "1")):
        cxx_flags.append("-DDISABLE_AGGRESSIVE_PTX_INSTRS")
        nvcc_flags.append("-DDISABLE_AGGRESSIVE_PTX_INSTRS")

    # Put them together
    extra_compile_args = {
        "cxx": cxx_flags,
        "nvcc": nvcc_flags,
        "nvcc_dlink": ["-dlink"],
    }
    # Summary
    print("Build summary:")
    print(f" > Sources: {sources}")
    print(f" > Includes: {include_dirs}")
    print(f" > Compilation flags: {extra_compile_args}")
    print(f' > Arch list: {os.getenv("PADDLE_CUDA_ARCH_LIST")}')
    print()

    setuptools.setup(
        name="ultra_ep",
        version=get_package_version(),
        packages=setuptools.find_packages(include=["ultra_ep", "ultra_ep.*"]),
        install_requires=["torch>=2.9", "nvidia-ml-py>=12.570"],
        ext_modules=[
            CUDAExtension(
                name="ultra_ep._C",
                include_dirs=include_dirs,
                sources=sources,
                extra_compile_args=extra_compile_args,
            )
        ],
        cmdclass={
            "build_ext": BuildExtension,
        },
    )
