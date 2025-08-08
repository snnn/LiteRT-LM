# Copyright 2025 The ODML Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import io
import os

from absl.testing import absltest

from litert_lm.python.tools import litertlm_peek


class LitertlmPeekPyTest(absltest.TestCase):

  def test_process_litertlm_file(self):
    """Tests the process_litertlm_file function directly."""
    test_data_path = os.path.join(
        os.environ.get("TEST_SRCDIR", ""),
        "litert_lm/schema/testdata/test_tok_tfl_llm.litertlm",
    )

    # Use an in-memory stream to capture the output.
    output_stream = io.StringIO()

    # Call the function directly.
    litertlm_peek.peek_litertlm_file(test_data_path, None, output_stream)

    # Get the output and perform assertions.
    stdout = output_stream.getvalue()
    self.assertNotEmpty(stdout)

    # Assert that the output contains expected strings.
    self.assertIn("LiteRT-LM Version:", stdout)
    self.assertIn("System Metadata", stdout)
    self.assertIn("Sections", stdout)
    self.assertIn("SP_Tokenizer", stdout)
    self.assertIn("TFLiteModel", stdout)
    self.assertIn("LlmMetadataProto", stdout)
    self.assertIn("<<<<<<<< start of LlmMetadata", stdout)

  def test_process_litertlm_file_hf_tokenizer(self):
    """Tests the process_litertlm_file function directly."""
    test_data_path = os.path.join(
        os.environ.get("TEST_SRCDIR", ""),
        "litert_lm/schema/testdata/test_hf_tokenizer.litertlm",
    )

    # Use an in-memory stream to capture the output.
    output_stream = io.StringIO()

    # Call the function directly.
    litertlm_peek.peek_litertlm_file(test_data_path, None, output_stream)

    # Get the output and perform assertions.
    stdout = output_stream.getvalue()
    self.assertNotEmpty(stdout)

    # Assert that the output contains expected strings.
    self.assertIn("LiteRT-LM Version:", stdout)
    self.assertIn("System Metadata", stdout)
    self.assertIn("Sections", stdout)
    self.assertIn("HF_Tokenizer_Zlib", stdout)

  def test_process_litertlm_file_tokenizer_tflite(self):
    """Tests the process_litertlm_file function directly."""
    test_data_path = os.path.join(
        os.environ.get("TEST_SRCDIR", ""),
        "litert_lm/schema/testdata/test_tokenizer_tflite.litertlm",
    )

    # Use an in-memory stream to capture the output.
    output_stream = io.StringIO()

    # Call the function directly.
    litertlm_peek.peek_litertlm_file(test_data_path, None, output_stream)

    # Get the output and perform assertions.
    stdout = output_stream.getvalue()
    self.assertNotEmpty(stdout)

    # Assert that the output contains expected strings.
    self.assertIn("LiteRT-LM Version:", stdout)
    self.assertIn("System Metadata", stdout)
    self.assertIn("Sections", stdout)
    self.assertIn("SP_Tokenizer", stdout)
    self.assertIn("TFLiteModel", stdout)

  def test_process_litertlm_file_with_dump_files(self):
    """Tests the process_litertlm_file function with dump_files_dir."""
    test_data_path = os.path.join(
        os.environ.get("TEST_SRCDIR", ""),
        "litert_lm/schema/testdata/test_tok_tfl_llm.litertlm",
    )

    # Create a temporary directory for dumping files.
    dump_dir = self.create_tempdir().full_path

    # Use an in-memory stream to capture the output.
    output_stream = io.StringIO()

    # Call the function directly with dump_files_dir.
    litertlm_peek.peek_litertlm_file(test_data_path, dump_dir, output_stream)

    # Get the output and perform assertions.
    stdout = output_stream.getvalue()
    self.assertNotEmpty(stdout)

    # Assert that the output contains expected strings.
    self.assertIn("LiteRT-LM Version:", stdout)
    self.assertIn("System Metadata", stdout)
    self.assertIn("Sections", stdout)
    self.assertIn("SP_Tokenizer", stdout)
    self.assertIn("TFLiteModel", stdout)
    self.assertIn("LlmMetadataProto", stdout)
    self.assertIn("<<<<<<<< start of LlmMetadata", stdout)

    # Check if files were dumped in the specified directory.
    self.assertTrue(
        os.path.exists(os.path.join(dump_dir, "Section0_SP_Tokenizer.spiece"))
    )
    self.assertTrue(
        os.path.exists(os.path.join(dump_dir, "Section1_TFLiteModel.tflite"))
    )
    self.assertTrue(
        os.path.exists(os.path.join(dump_dir, "LlmMetadataProto.pbtext"))
    )


if __name__ == "__main__":
  absltest.main()
