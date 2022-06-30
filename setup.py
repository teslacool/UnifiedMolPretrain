from setuptools import setup, find_packages
import sys

assert sys.version_info.major == 3 and sys.version_info.minor >= 6
setup(
    name="pretrain3d",
    packages=[package for package in find_packages()],
    install_requires=["torch", "torch_geometric"],
    description="3D molecule pre-training",
    author="anonymous",
)
