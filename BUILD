# Copyright 2025 The ODML Authors.
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

# If litert_lm_link_capi_so is not defined, each target links LiteRT C API either statically or
# dynamically based on its own requirements.
config_setting(
    name = "litert_lm_link_capi_so",
    values = {"define": "litert_lm_link_capi_so=true"},
)

config_setting(
    name = "litert_lm_link_capi_static",
    values = {"define": "litert_lm_link_capi_so=false"},
)
