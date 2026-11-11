#!/bin/bash
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
set -euo pipefail

export AUDITWHEEL_POLICY=manylinux_2_28
export AUDITWHEEL_ARCH=x86_64
export CIBW_ARCHS=x86_64
export CIBW_PLATFORM=linux

python -m pip install auditwheel 'pytest-asyncio>=1.3' 'pytest-mock>=3.15.1' 'pytest-timeout>=2.4'
yum install -y fribidi
cd /build/pillow
# Reuse upstream's complete, patched native dependency build, not system codecs.
# Upstream multibuild expects some optional environment variables to be unset.
bash .github/workflows/wheels-dependencies.sh
python -m pip wheel --no-deps -C raqm=enable -C raqm=vendor -C fribidi=vendor -C imagequant=disable \
    --wheel-dir /unrepaired .
python -m auditwheel repair --wheel-dir /wheels /unrepaired/*.whl
python .github/generate-sbom.py
python .github/embed-sbom.py /wheels .
python -m pip install --find-links=/wheels 'pillow[tests]==13.0.0+rai.py310.1' 'numpy<1.24'
python selftest.py
python /build/test_security_backports.py pillow

cd /build/filelock
# GitHub archives have no .git directory: label the actual patched source explicitly.
SETUPTOOLS_SCM_PRETEND_VERSION=4.2.1+rai.py310.1 python -m pip wheel --no-deps --wheel-dir /wheels .
python -m pip install --find-links=/wheels 'filelock==4.2.1+rai.py310.1'
python /build/test_security_backports.py filelock
