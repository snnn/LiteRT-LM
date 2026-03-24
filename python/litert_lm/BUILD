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

load(
    "@org_tensorflow//tensorflow:pytype.default.bzl",
    "pytype_strict_library",
)
# [Google-internal load of `cc_binary`]

load(":wheel_tags.bzl", "current_python_wheel_tags")
load("@rules_python//python:defs.bzl", "py_test")

package(default_visibility = ["//visibility:public"])

pytype_strict_library(
    name = "interfaces",
    srcs = ["interfaces.py"],
)

pytype_strict_library(
    name = "tools",
    srcs = ["tools.py"],
)

cc_binary(
    name = "litert_lm_ext.so",
    srcs = ["litert_lm.cc"],
    copts = [
        "-fexceptions",
        "-fvisibility=hidden",
    ],
    features = ["-use_header_modules"],
    linkopts = select({
        "@platforms//os:macos": ["-Wl,-undefined,dynamic_lookup"],
        "//conditions:default": [
            "-Wl,-Bsymbolic",
            "-Wl,-rpath,'$$ORIGIN'",
        ],
    }),
    linkshared = True,
    tags = [
        "noandroid",
        "noios",
        "nowindows",
    ],
    deps = [
        "@com_google_absl//absl/base:log_severity",
        "@com_google_absl//absl/log:globals",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
        "@nanobind",
        "@nanobind_json",
        "@litert//litert/c/internal:litert_logging",
        "//runtime/conversation",
        "//runtime/core:engine_impl",
        "//runtime/engine:engine_factory",
        "//runtime/engine:engine_interface",
        "//runtime/engine:engine_settings",
        "//runtime/executor:litert_compiled_model_executor_utils",
        "@rules_python//python/cc:current_py_cc_headers",
        "@litert//tflite:minimal_logging",
        "@litert//tflite/core/c:private_c_api_types",
    ],
)

pytype_strict_library(
    name = "litert_lm",
    srcs = [
        "__init__.py",
        "litert_lm_ext.py",
    ],
    data = [":litert_lm_ext.so"] + select({
        "@platforms//os:macos": [":copy_gemma_model_constraint_provider_dylib"],
        "@platforms//os:windows": [], # the cc_binary rule already properly provisions this DLL into the package directory on Windows.
        "//conditions:default": [":copy_gemma_model_constraint_provider_so"],
    }),
    deps = [
        ":interfaces",
        ":tools",
    ],
)

py_test(
    name = "engine_test",
    srcs = ["engine_test.py"],
    data = ["//runtime/testdata"],
    tags = ["nowindows"],
    deps = [
        ":litert_lm",
        "@absl_py//absl/flags",
        "@absl_py//absl/testing:absltest",
    ],
    imports = [".."],
)

py_test(
    name = "tool_test",
    srcs = ["tool_test.py"],
    tags = ["nowindows"],
    deps = [
        ":litert_lm",
        "@absl_py//absl/testing:absltest",
        "@absl_py//absl/testing:parameterized",
    ],
    imports = [".."],
)

load("@rules_python//python:packaging.bzl", "py_package", "py_wheel")
load("@bazel_skylib//rules:copy_file.bzl", "copy_file")
load("//:version.bzl", "VERSION")

config_setting(
    name = "is_nightly",
    define_values = {"PYPI_NIGHTLY": "1"},
)

genrule(
    name = "copy_gemma_model_constraint_provider_so",
    srcs = select({
        "//build_config:linux_x86_64": ["//prebuilt/linux_x86_64:libGemmaModelConstraintProvider.so"],
        "//build_config:linux_arm64": ["//prebuilt/linux_arm64:libGemmaModelConstraintProvider.so"],
        "//conditions:default": [],
    }),
    outs = ["libGemmaModelConstraintProvider.so"],
    cmd = "if [ -n \"$(SRCS)\" ]; then cp $< $@; else touch $@; fi",
    tags = ["manual"],
)

genrule(
    name = "copy_gemma_model_constraint_provider_dylib",
    srcs = select({
        "@platforms//os:macos": ["//prebuilt/macos_arm64:libGemmaModelConstraintProvider.dylib"],
        "//conditions:default": [],
    }),
    outs = ["libGemmaModelConstraintProvider.dylib"],
    cmd = "if [ -n \"$(SRCS)\" ]; then cp $< $@; else touch $@; fi",
    tags = ["manual"],
)

current_python_wheel_tags(
    name = "current_python_wheel_tags",
)

py_package(
    name = "litert_lm_pkg",
    # Explicitly specify packages to avoid including transitively linked shared
    # libraries (like duplicate .so files in _solib_*/...) in the wheel.
    packages = ["python.litert_lm"],
    deps = [":litert_lm"],
)

# To build normal wheel:
# $ PYTHON_VERSION=3.10
# $ bazel build \
#     --repo_env=HERMETIC_PYTHON_VERSION=${PYTHON_VERSION} \
#     --@rules_python//python/config_settings:python_version=${PYTHON_VERSION} \
#     -c opt //python/litert_lm:litert_lm_wheel
#
# To build nightly wheel:
# $ PYTHON_VERSION=3.10
# $ DATE=$(date +'%Y%m%d')
# $ bazel build \
#     --repo_env=HERMETIC_PYTHON_VERSION=${PYTHON_VERSION} \
#     --@rules_python//python/config_settings:python_version=${PYTHON_VERSION} \
#     --define=PYPI_NIGHTLY=1 \
#     --define=PYPI_NIGHTLY_DATE=${DATE} \
#     -c opt //python/litert_lm:litert_lm_wheel
py_wheel(
    name = "litert_lm_wheel",
    abi = "$(ABI)",
    distribution = select({
        ":is_nightly": "litert-lm-nightly",
        "//conditions:default": "litert-lm",
    }),
    platform = select({
        "//build_config:linux_x86_64": "manylinux_2_27_x86_64",
        "//build_config:linux_arm64": "manylinux_2_27_aarch64",
        "@platforms//os:macos": "macosx_12_0_arm64",
        "@platforms//os:windows": "win_amd64",
        "//conditions:default": "any",
    }),
    python_tag = select({
        "@rules_python//python/config_settings:is_python_3.10": "cp310",
        "@rules_python//python/config_settings:is_python_3.11": "cp311",
        "@rules_python//python/config_settings:is_python_3.12": "cp312",
        "@rules_python//python/config_settings:is_python_3.13": "cp313",
        "//conditions:default": "py3",
    }),
    summary = "LiteRT-LM inference library",
    license = "Apache-2.0",
    description_file = "README.md",
    homepage = "https://github.com/google-ai-edge/LiteRT-LM",
    classifiers = [
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: POSIX :: Linux",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: MacOS :: MacOS X",
    ],
    strip_path_prefixes = ["python"],
    toolchains = [":current_python_wheel_tags"],
    version = select({
        ":is_nightly": VERSION + ".dev$(PYPI_NIGHTLY_DATE)",
        "//conditions:default": VERSION,
    }),
    deps = [":litert_lm_pkg"],
)
