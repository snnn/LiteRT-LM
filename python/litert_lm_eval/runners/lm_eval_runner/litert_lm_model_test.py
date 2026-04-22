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

"""Tests for LiteRT-LM lm-eval backend behavior."""

import sys
import types
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized


def _register_model(*_aliases):
  return lambda cls: cls


class _TemplateLM:

  def __init__(self):
    self.cache_hook = mock.MagicMock()


mock_lm_eval = mock.MagicMock()
sys.modules["lm_eval"] = mock_lm_eval
sys.modules["lm_eval.utils"] = types.SimpleNamespace(
    get_rolling_token_windows=lambda **_: [],
    make_disjoint_window=lambda x: x,
)
sys.modules["lm_eval.api"] = mock.MagicMock()
sys.modules["lm_eval.api.model"] = types.SimpleNamespace(TemplateLM=_TemplateLM)
sys.modules["lm_eval.api.registry"] = types.SimpleNamespace(
    register_model=_register_model
)
sys.modules["lm_eval.models"] = mock.MagicMock()
sys.modules["lm_eval.models.utils"] = types.SimpleNamespace(
    normalize_gen_kwargs=lambda kwargs, default_max_gen_toks: {
        "max_gen_toks": kwargs.get("max_gen_toks", default_max_gen_toks),
        "until": kwargs.get("until", []),
        "do_sample": kwargs.get("do_sample", False),
        "temperature": kwargs.get("temperature", 0.0),
    }
)

mock_litert_lm = types.SimpleNamespace(
    Backend=types.SimpleNamespace(CPU="cpu", GPU="gpu"),
    Engine=mock.MagicMock(),
    GenerateConfig=lambda max_output_tokens: types.SimpleNamespace(
        max_output_tokens=max_output_tokens
    ),
)
sys.modules["litert_lm"] = mock_litert_lm

from litert_lm_eval.runners.lm_eval_runner import litert_lm_model  # pylint: disable=g-import-not-at-top


class LitertLmModelTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    mock_litert_lm.Engine.reset_mock()

  def test_generate_until_stops_at_earliest_sequence(self):
    model = litert_lm_model.LitertLmModelRunner(pretrained="dummy_path")

    mock_session = mock.MagicMock()
    mock_decode_responses = mock.MagicMock()
    mock_decode_responses.texts = [
        "The answer is 42.\n\nUser: What is next? Question:"
    ]
    mock_session.run_decode.return_value = mock_decode_responses
    model.engine.create_session.return_value.__enter__.return_value = mock_session
    model.engine.create_session.return_value.__exit__.return_value = None

    class MockRequest:

      def __init__(self):
        self.args = (
            "context prompt",
            {"until": ["User:", "\n\n", "Question:"]},
        )

    res = model.generate_until([MockRequest()], disable_tqdm=True)
    self.assertEqual(["The answer is 42."], res)

  def test_loglikelihood_tokens_groups_shared_context(self):
    model = litert_lm_model.LitertLmModelRunner(pretrained="dummy_path")
    model.cache_hook = mock.MagicMock()

    mock_session = mock.MagicMock()
    scoring = mock.MagicMock()
    scoring.scores = [-1.5, -3.0, -0.5]
    scoring.greedy_token_ids = [[20], [99], [20]]
    mock_session.run_token_scoring.return_value = scoring
    model.engine.create_session.return_value.__enter__.return_value = mock_session
    model.engine.create_session.return_value.__exit__.return_value = None

    requests = [
      (("hello", " world"), [10], [20]),
      (("hello", " everyone"), [10], [30]),
      (("different", " world"), [11], [20]),
    ]

    res = model._loglikelihood_tokens(requests, disable_tqdm=True)

    self.assertEqual([(-1.5, True), (-3.0, False), (-0.5, True)], res)
    self.assertEqual(2, mock_session.run_prefill_token_ids.call_count)
    self.assertEqual(2, mock_session.run_token_scoring.call_count)
    mock_session.run_prefill_token_ids.assert_any_call([10])
    mock_session.run_prefill_token_ids.assert_any_call([11])
    mock_session.run_token_scoring.assert_any_call([[20], [30]])
    mock_session.run_token_scoring.assert_any_call([[20]])

  def test_loglikelihood_tokens_requires_raw_prompt_mode(self):
    model = litert_lm_model.LitertLmModelRunner(
        pretrained="dummy_path", prompt_mode="bundle"
    )
    with self.assertRaises(NotImplementedError):
      model._loglikelihood_tokens([], disable_tqdm=True)


if __name__ == "__main__":
  absltest.main()
