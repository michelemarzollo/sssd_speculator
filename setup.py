# Copyright (c) Huawei Technologies Co., Ltd. 2020-2020. All rights reserved.

import os
import sys
import subprocess
import setuptools
import shutil
import pybind11

from setuptools import Extension
from setuptools.command.build_ext import build_ext


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=''):
        super().__init__(name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    def run(self):
        # Find the absolute path of the cmake executable
        cmake_executable = shutil.which('cmake')

        if cmake_executable is None:
            raise RuntimeError("CMake must be installed to build the following extensions: " +
                               ", ".join(e.name for e in self.extensions))

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))

        # pybind11 include dir
        include_path = pybind11.get_include()
        # See comments from Caleb and MrCrHaM (works also for conda)
        # https://stackoverflow.com/questions/63254584/how-to-make-cmake-find-pybind11
        base_path, _ = include_path.rsplit('/include', 1)
        modified_path = base_path + '/share/cmake/pybind11'

        cmake_args = [
            f'-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}',
            f'-DPYTHON_EXECUTABLE={sys.executable}',
            f'-DPYBIND11_INCLUDE_DIR={modified_path}',
        ]

        cfg = 'Debug' if self.debug else 'Release'
        cmake_args += [f'-DCMAKE_BUILD_TYPE={cfg}']

        cxx_flags = '-O3'

        cmake_args += [f'-DCMAKE_CXX_FLAGS={cxx_flags}']

        build_args = ['--config', cfg, '--', '-j4']

        build_tmp = os.path.join(os.getcwd(), 'build')
        if self.build_temp is not None:
            build_tmp = self.build_temp

        if not os.path.exists(build_tmp):
            os.makedirs(build_tmp)

        # Run CMake configuration
        subprocess.check_call([shutil.which('cmake'), ext.sourcedir] + cmake_args, cwd=build_tmp)
        # Build the extension
        subprocess.check_call([shutil.which('cmake'), '--build', '.'] + build_args, cwd=build_tmp)


setuptools.setup(
    name='sssd_speculator',
    author='Huawei Technologies Co., Ltd.',
    author_email='michele.marzollo@huawei.com',
    url='https://github.com/michelemarzollo/sssd_speculator',
    license='Proprietary',
    packages=["sssd_speculator"],
    ext_modules=[
        CMakeExtension(
            "sssd_speculator.sssd_speculator",
            sourcedir="sssd_speculator",
        )
    ],
    cmdclass=dict(build_ext=CMakeBuild),
)
