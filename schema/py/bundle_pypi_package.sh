#!/bin/bash
# Copyright 2026 The ODML Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This script bundles the litert-lm Python package into a PyPI-ready wheel.
# It performs the following steps:
# 1. Builds Protobuf and FlatBuffer bindings using Bazel.
# 2. Stages the source files and generated bindings in a temporary directory.
# 3. Rewrites absolute imports in generated Protobuf files to match the package structure.
# 4. Creates a virtual environment and builds the wheel using 'uv'.
# 5. Verifies the built wheel by installing it and running help commands.

# Ensure script stops on error
set -e

WORKSPACE_ROOT=$(bazel info workspace)
echo "Workspace Root: ${WORKSPACE_ROOT}"
STAGING_DIR="/tmp/litertlm_builder"

# Build Proto and FlatBuffer bindings
bazel build //runtime/proto:all
bazel build //schema/core:litertlm_header_schema_py

# Create a temporary staging directory
echo "Setting up staging directory: ${STAGING_DIR}"

rm -rf "${STAGING_DIR}"
mkdir -p "${STAGING_DIR}"

# Copy the Python package source files from schema/py
# This assumes pyproject.toml is in ${WORKSPACE_ROOT}/schema/py
echo "Copying schema/py contents..."
mkdir -p "${STAGING_DIR}/litert_lm_builder"
mkdir -p "${STAGING_DIR}/litert_lm_builder/schema/core"
mkdir -p "${STAGING_DIR}/litert_lm_builder/runtime/proto"

# Create necessary __init__.py files for sub-packages
touch "${STAGING_DIR}/litert_lm_builder/schema/__init__.py"
touch "${STAGING_DIR}/litert_lm_builder/schema/core/__init__.py"
touch "${STAGING_DIR}/litert_lm_builder/runtime/__init__.py"
touch "${STAGING_DIR}/litert_lm_builder/runtime/proto/__init__.py"

# Copy python files to litert_lm_builder directory
cp "${WORKSPACE_ROOT}/schema/py/"*.py "${STAGING_DIR}/litert_lm_builder/"

# Copy pyproject.toml to the root of staging
cp "${WORKSPACE_ROOT}/schema/py/pyproject.toml" "${STAGING_DIR}/"

# Copy the generated Protobuf Python files
echo "Copying Protobuf bindings..."
cp -f "${WORKSPACE_ROOT}/bazel-bin/runtime/proto/"*_pb2.py "${STAGING_DIR}/litert_lm_builder/runtime/proto/"

# Rewrite absolute imports in generated Protobuf files
echo "Rewriting absolute imports in Protobuf files..."
find "${STAGING_DIR}/litert_lm_builder/runtime/proto" -name "*_pb2.py" -exec sed -i 's/from runtime.proto/from litert_lm_builder.runtime.proto/g' {} \;

# Copy the generated FlatBuffer Python file
echo "Copying FlatBuffer bindings..."
find "${WORKSPACE_ROOT}/bazel-bin/schema/core" -name "*.py" -exec cp -f {} "${STAGING_DIR}/litert_lm_builder/schema/core/" \;

# Rewrite imports in all Python files in the package to use the new layout
echo "Rewriting imports in all Python files..."
find "${STAGING_DIR}/litert_lm_builder" -name "*.py" -exec sed -i -e 's/from litert_lm\.schema\.py/from litert_lm_builder/g' -e 's/from litert_lm\./from litert_lm_builder\./g' {} \;

cd "${STAGING_DIR}"

# Create a fresh, isolated virtual environment inside staging!
python3 -m venv .venv
source .venv/bin/activate

# Install uv and build tools into this fresh environment
python3 -m pip install uv
uv pip install --upgrade pip
uv pip install setuptools wheel

# Build without isolation (uses the setuptools we just installed!)
uv build --no-build-isolation

# Install the wheel we just built to verify it works!
uv pip install dist/litert_lm_builder-*.whl

# Verification checks using the new tools
litertlm-builder --help
litertlm-peek --help

# Python API check (verifies __init__.py)
python3 -c "import litert_lm_builder; print(litert_lm_builder.LitertLmFileBuilder)"
