import glob
import os
import platform
import re
import warnings
from setuptools import find_packages, setup

from torch.utils.cpp_extension import BuildExtension
cmd_class = {'build_ext': BuildExtension}


def parse_requirements(fname='requirements/runtime.txt', with_version=True):
    """Parse the package dependencies listed in a requirements file but strips
    specific versioning information.

    Args:
        fname (str): path to requirements file
        with_version (bool, default=False): if True include version specs

    Returns:
        List[str]: list of requirements items

    CommandLine:
        python -c "import setup; print(setup.parse_requirements())"
    """
    import sys
    from os.path import exists
    require_fpath = fname

    def parse_line(line):
        """Parse information from a line in a requirements text file."""
        if line.startswith('-r '):
            # Allow specifying requirements in other files
            target = line.split(' ')[1]
            for info in parse_require_file(target):
                yield info
        else:
            info = {'line': line}
            if line.startswith('-e '):
                info['package'] = line.split('#egg=')[1]
            else:
                # Remove versioning from the package
                pat = '(' + '|'.join(['>=', '==', '>']) + ')'
                parts = re.split(pat, line, maxsplit=1)
                parts = [p.strip() for p in parts]

                info['package'] = parts[0]
                if len(parts) > 1:
                    op, rest = parts[1:]
                    if ';' in rest:
                        # Handle platform specific dependencies
                        # http://setuptools.readthedocs.io/en/latest/setuptools.html#declaring-platform-specific-dependencies
                        version, platform_deps = map(str.strip,
                                                     rest.split(';'))
                        info['platform_deps'] = platform_deps
                    else:
                        version = rest  # NOQA
                    info['version'] = (op, version)
            yield info

    def parse_require_file(fpath):
        with open(fpath, 'r') as f:
            for line in f.readlines():
                line = line.strip()
                if line and not line.startswith('#'):
                    for info in parse_line(line):
                        yield info

    def gen_packages_items():
        if exists(require_fpath):
            for info in parse_require_file(require_fpath):
                parts = [info['package']]
                if with_version and 'version' in info:
                    parts.extend(info['version'])
                if not sys.version.startswith('3.4'):
                    # apparently package_deps are broken in 3.4
                    platform_deps = info.get('platform_deps')
                    if platform_deps is not None:
                        parts.append(';' + platform_deps)
                item = ''.join(parts)
                yield item

    packages = list(gen_packages_items())
    return packages


install_requires = parse_requirements()


def get_extensions():
    extensions = []

    ext_name = 'cuda_example._ext'
    from torch.utils.cpp_extension import CUDAExtension

    # prevent ninja from using too many resources
    try:
        import psutil
        num_cpu = len(psutil.Process().cpu_affinity())
        cpu_use = max(4, num_cpu - 1)
    except (ModuleNotFoundError, AttributeError):
        cpu_use = 4

    os.environ.setdefault('MAX_JOBS', str(cpu_use))
    define_macros = []

    # Before PyTorch1.8.0, when compiling CUDA code, `cxx` is a
    # required key passed to PyTorch. Even if there is no flag passed
    # to cxx, users also need to pass an empty list to PyTorch.
    # Since PyTorch1.8.0, it has a default value so users do not need
    # to pass an empty list anymore.
    # More details at https://github.com/pytorch/pytorch/pull/45956
    extra_compile_args = {'cxx': []}

    # Since the PR (https://github.com/open-mmlab/mmcv/pull/1463) uses
    # c++14 features, the argument ['std=c++14'] must be added here.
    # However, in the windows environment, some standard libraries
    # will depend on c++17 or higher. In fact, for the windows
    # environment, the compiler will choose the appropriate compiler
    # to compile those cpp files, so there is no need to add the
    # argument
    if platform.system() != 'Windows':
        extra_compile_args['cxx'] = ['-std=c++14']

    include_dirs = []

    define_macros += [('MMCV_WITH_CUDA', None)]
    cuda_args = os.getenv('MMCV_CUDA_ARGS')
    extra_compile_args['nvcc'] = [cuda_args] if cuda_args else []
    op_files = glob.glob('./cuda_example/pytorch/*.cpp') + \
        glob.glob('./cuda_example/pytorch/cpu/*.cpp') + \
        glob.glob('./cuda_example/pytorch/cuda/*.cu') + \
        glob.glob('./cuda_example/pytorch/cuda/*.cpp')
    extension = CUDAExtension
    include_dirs.append(os.path.abspath('./cuda_example/common'))
    include_dirs.append(os.path.abspath('./cuda_example/common/cuda'))

    # Since the PR (https://github.com/open-mmlab/mmcv/pull/1463) uses
    # c++14 features, the argument ['std=c++14'] must be added here.
    # However, in the windows environment, some standard libraries
    # will depend on c++17 or higher. In fact, for the windows
    # environment, the compiler will choose the appropriate compiler
    # to compile those cpp files, so there is no need to add the
    # argument
    if 'nvcc' in extra_compile_args and platform.system() != 'Windows':
        extra_compile_args['nvcc'] += ['-std=c++14']

    ext_ops = extension(
        name=ext_name,
        sources=op_files,
        include_dirs=include_dirs,
        define_macros=define_macros,
        extra_compile_args=extra_compile_args)
    extensions.append(ext_ops)

    return extensions


setup(
    name='torch-cuda-project',
    version="V0.0.1",
    description='Inherit from OpenMMLab',
    keywords='Practice for torch cuda op',
    packages=find_packages(),
    include_package_data=True,
    classifiers=[
        'Development Status :: 4 - Beta',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Topic :: Utilities',
    ],
    url='https://github.com/www516717402',
    author='WJY',
    author_email='www516717402@gmail.com',
    install_requires=install_requires,
    extras_require={
        'all': parse_requirements('requirements.txt'),
        'tests': parse_requirements('requirements/test.txt'),
        'build': parse_requirements('requirements/build.txt'),
        'optional': parse_requirements('requirements/optional.txt'),
    },
    ext_modules=get_extensions(),
    cmdclass=cmd_class,
    zip_safe=False)
