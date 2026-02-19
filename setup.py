import os
import re
import subprocess
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

_src_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dqtorch')

nvcc_flags = [
    '-O3', '-std=c++17',
    '-U__CUDA_NO_HALF_OPERATORS__', '-U__CUDA_NO_HALF_CONVERSIONS__', '-U__CUDA_NO_HALF2_OPERATORS__',
]


def _cuda_version_tuple():
    """Return CUDA toolkit version as (major, minor), or None if unavailable."""
    try:
        result = subprocess.run(
            ['nvcc', '--version'],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None

    match = re.search(r"release\s+(\d+)\.(\d+)", result.stdout + result.stderr)
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def _arch_gencode_flags():
    """
    Force native SASS builds for A100 (sm_80) and H100 (sm_90) to avoid
    runtime PTX JIT/toolchain mismatches.
    """
    flags = [
        '-gencode=arch=compute_80,code=sm_80',
    ]

    version = _cuda_version_tuple()
    # sm_90 requires CUDA >= 11.8
    if version is not None and (version[0] > 11 or (version[0] == 11 and version[1] >= 8)):
        flags.append('-gencode=arch=compute_90,code=sm_90')
    else:
        print(
            "WARNING: CUDA toolkit does not report >= 11.8; building without sm_90. "
            "H100 support requires CUDA 11.8+."
        )
    return flags


nvcc_flags += _arch_gencode_flags()

if os.name == "posix":
    c_flags = ['-O3', '-std=c++17']
elif os.name == "nt":
    c_flags = ['/O2', '/std:c++17']

    # find cl.exe
    def find_cl_path():
        import glob
        for edition in ["Enterprise", "Professional", "BuildTools", "Community"]:
            paths = sorted(glob.glob(r"C:\\Program Files (x86)\\Microsoft Visual Studio\\*\\%s\\VC\\Tools\\MSVC\\*\\bin\\Hostx64\\x64" % edition), reverse=True)
            if paths:
                return paths[0]

    # If cl.exe is not on path, try to find it.
    if os.system("where cl.exe >nul 2>nul") != 0:
        cl_path = find_cl_path()
        if cl_path is None:
            raise RuntimeError("Could not locate a supported Microsoft Visual C++ installation")
        os.environ["PATH"] += ";" + cl_path

LIB_NAME = 'dqtorch'
setup(
    name=LIB_NAME, # package name, import this to use python API
    description='A faster pytorch libraray for (dual) quaternion batched operations.',
    license="MIT",
    author="Chaoyang Wang",
    python_requires=">=3.6",
    install_requires=["torch>=1.12"],
    packages=[LIB_NAME],
    ext_modules=[
        CUDAExtension(
            name='_quaternion_cuda', # extension name, import this to use CUDA API
            sources=[os.path.join(_src_path, 'src', f) for f in [
                'quaternion.cu',
                'bindings.cpp',
            ]],
            extra_compile_args={
                'cxx': c_flags,
                'nvcc': nvcc_flags,
            }
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension,
    }
)