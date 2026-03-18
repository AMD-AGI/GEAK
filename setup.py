# Modifications Copyright(C)[2025] Advanced Micro Devices, Inc. All rights reserved.
# https://github.com/algorithmicsuperintelligence/openevolve  - Apache License 2.0

from setuptools import setup, find_packages

setup(
    name="openevolve",
    version="0.0.11",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "anthropic",
    ],
)
