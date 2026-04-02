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

"""Unit tests for the main litert-lm CLI."""

from absl.testing import absltest
from click.testing import CliRunner

from litert_lm_cli import main


class MainTest(absltest.TestCase):

  def test_help_shorthand(self):
    runner = CliRunner()
    result_help = runner.invoke(main.cli, ["--help"])
    result_h = runner.invoke(main.cli, ["-h"])
    self.assertEqual(result_help.exit_code, 0)
    self.assertEqual(result_h.exit_code, 0)
    self.assertEqual(result_help.output, result_h.output)

  def test_subcommand_help_shorthand(self):
    runner = CliRunner()
    result_help = runner.invoke(main.cli, ["list", "--help"])
    result_h = runner.invoke(main.cli, ["list", "-h"])
    self.assertEqual(result_help.exit_code, 0)
    self.assertEqual(result_h.exit_code, 0)
    self.assertEqual(result_help.output, result_h.output)


if __name__ == "__main__":
  absltest.main()
