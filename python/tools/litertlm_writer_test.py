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
from litert_lm.python.tools import litertlm_writer


class LitertlmWriterPyTest(absltest.TestCase):

  def test_litertlm_write(self):
    # Create dummy input files.
    tokenizer_path = self.create_tempfile(
        "tokenizer.spiece", "Dummy tokenizer content"
    ).full_path
    model_path = self.create_tempfile(
        "model.tflite", "Dummy model content"
    ).full_path
    output_path = os.path.join(self.create_tempdir(), "output.litertlm")

    input_files = [tokenizer_path, model_path]
    metadata_str = "tokenizer:lang=en;tflite:quantized=true,size=1024"

    # Call the library function.
    litertlm_writer.litertlm_write(output_path, input_files, metadata_str)

    # Verify the output file exists.
    self.assertTrue(os.path.exists(output_path))
    self.assertGreater(os.path.getsize(output_path), 0)

    # Use the peek library to inspect the contents.
    output_stream = io.StringIO()
    litertlm_peek.peek_litertlm_file(output_path, None, output_stream)
    peek_output = output_stream.getvalue()

    # Assert that the output contains expected strings.
    self.assertIn("LiteRT-LM Version: 1.3.0", peek_output)
    self.assertIn("SP_Tokenizer", peek_output)
    self.assertIn("TFLiteModel", peek_output)
    self.assertIn("Key: lang, Value (String): en", peek_output)
    self.assertIn("Key: quantized, Value (Bool): True", peek_output)
    # These following offset can verify the padding is correct.
    self.assertIn("Begin Offset: 16384", peek_output)
    self.assertIn("Begin Offset: 32768", peek_output)

  def test_litertlm_write_comprehensive(self):
    """Tests creation with all supported file types."""
    tokenizer_path = self.create_tempfile(
        "tokenizer.spiece", "Dummy tokenizer content"
    ).full_path
    hf_tokenizer_path = self.create_tempfile(
        "tokenizer.json", '{"version": "1.0"}'
    ).full_path
    model_path = self.create_tempfile(
        "model.tflite", "Dummy model content"
    ).full_path
    llm_metadata_path = self.create_tempfile(
        "llm_metadata.pbtext", 'start_token: { token_str: "<start>" }'
    ).full_path
    binary_data_path = self.create_tempfile(
        "data.bin", "Dummy binary data"
    ).full_path
    output_path = os.path.join(self.create_tempdir(), "output.litertlm")

    input_files = [
        tokenizer_path,
        hf_tokenizer_path,
        model_path,
        llm_metadata_path,
        binary_data_path,
    ]
    metadata_str = (
        "tokenizer:lang=en,version=1.2;"
        "hf_tokenizer_zlib:config=base;"
        "tflite:quantized=false,size=2048;"
        "llm_metadata:author=tester,temp=0.9;"
        "binary_data:type=test"
    )

    litertlm_writer.litertlm_write(output_path, input_files, metadata_str)
    self.assertTrue(os.path.exists(output_path))

    output_stream = io.StringIO()
    litertlm_peek.peek_litertlm_file(output_path, None, output_stream)
    peek_output = output_stream.getvalue()

    self.assertIn("SP_Tokenizer", peek_output)
    self.assertIn("HF_Tokenizer_Zlib", peek_output)
    self.assertIn("TFLiteModel", peek_output)
    self.assertIn("LlmMetadataProto", peek_output)
    self.assertIn("GenericBinaryData", peek_output)
    self.assertIn("Key: lang, Value (String): en", peek_output)
    self.assertIn("Key: temp, Value (Double): 0.9", peek_output)
    self.assertIn("Key: size, Value (Int64): 2048", peek_output)

  def test_litertlm_write_multiple_tflite(self):
    """Tests creation with all supported file types."""
    tokenizer_path = self.create_tempfile(
        "tokenizer.spiece", "Dummy tokenizer content"
    ).full_path
    hf_tokenizer_path = self.create_tempfile(
        "tokenizer.json", '{"version": "1.0"}'
    ).full_path
    model_path = self.create_tempfile(
        "model.tflite", "Dummy model content"
    ).full_path
    model_path_2 = self.create_tempfile(
        "model_2.tflite", "Dummy model content 2"
    ).full_path
    llm_metadata_path = self.create_tempfile(
        "llm_metadata.pbtext", 'start_token: { token_str: "<start>" }'
    ).full_path
    output_path = os.path.join(self.create_tempdir(), "output.litertlm")

    input_files = [
        tokenizer_path,
        hf_tokenizer_path,
        model_path,
        model_path_2,
        llm_metadata_path,
    ]
    metadata_str = (
        "tokenizer:lang=en,version=1.2;"
        "hf_tokenizer_zlib:config=base;"
        "tflite:model_type=PREFILL_DECODE;"
        "tflite:model_type=EMBEDDER;"
        "llm_metadata:author=tester,temp=0.9;"
        "binary_data:type=test"
    )

    litertlm_writer.litertlm_write(output_path, input_files, metadata_str)
    self.assertTrue(os.path.exists(output_path))

    output_stream = io.StringIO()
    litertlm_peek.peek_litertlm_file(output_path, None, output_stream)
    peek_output = output_stream.getvalue()
    self.assertIn("SP_Tokenizer", peek_output)
    self.assertIn("HF_Tokenizer_Zlib", peek_output)
    self.assertIn("TFLiteModel", peek_output)
    self.assertIn("LlmMetadataProto", peek_output)
    self.assertIn("Key: lang, Value (String): en", peek_output)
    self.assertIn("Key: temp, Value (Double): 0.9", peek_output)
    self.assertEqual(
        peek_output.count("Key: model_type, Value (String): PREFILL_DECODE"), 1
    )
    self.assertEqual(
        peek_output.count("Key: model_type, Value (String): EMBEDDER"), 1
    )

  def test_no_metadata(self):
    """Tests writing a file with no section metadata."""
    tokenizer_path = self.create_tempfile(
        "tokenizer.spiece", "Dummy tokenizer content"
    ).full_path
    output_path = os.path.join(self.create_tempdir(), "output.litertlm")

    litertlm_writer.litertlm_write(output_path, [tokenizer_path], "")
    self.assertTrue(os.path.exists(output_path))

    output_stream = io.StringIO()
    litertlm_peek.peek_litertlm_file(output_path, None, output_stream)
    peek_output = output_stream.getvalue()

    self.assertIn("SP_Tokenizer", peek_output)

  def test_metadata_on_some_sections(self):
    """Tests providing metadata for only a subset of sections."""
    tokenizer_path = self.create_tempfile(
        "tokenizer.spiece", "Dummy tokenizer content"
    ).full_path
    model_path = self.create_tempfile(
        "model.tflite", "Dummy model content"
    ).full_path
    output_path = os.path.join(self.create_tempdir(), "output.litertlm")

    metadata_str = "tokenizer:lang=fr"
    litertlm_writer.litertlm_write(
        output_path, [tokenizer_path, model_path], metadata_str
    )
    self.assertTrue(os.path.exists(output_path))

    output_stream = io.StringIO()
    litertlm_peek.peek_litertlm_file(output_path, None, output_stream)
    peek_output = output_stream.getvalue()

    self.assertIn("Key: lang, Value (String): fr", peek_output)
    # Ensure no metadata is associated with the tflite section
    tflite_section = peek_output.split("Section 1:")[1]
    self.assertIn("<None>", tflite_section)
    self.assertNotIn("Key:", tflite_section)

  def test_empty_input_files(self):
    """Tests that an error is raised for empty input file list."""
    with self.assertRaisesRegex(ValueError, "At least one input file"):
      litertlm_writer.litertlm_write("output.litertlm", [], "")

  def test_non_existent_input_file(self):
    """Tests that an error is raised for a non-existent input file."""
    with self.assertRaises(FileNotFoundError):
      output_path = os.path.join(self.create_tempdir(), "output.litertlm")
      litertlm_writer.litertlm_write(
          output_path, ["non_existent_file.tflite"], ""
      )

  def test_invalid_metadata_format_missing_colon(self):
    """Tests error handling for malformed metadata string."""
    with self.assertRaisesRegex(ValueError, "Invalid section metadata format"):
      litertlm_writer.parse_metadata_string("tokenizerkey=val")

  def test_invalid_metadata_format_bad_kv_pair(self):
    """Tests error handling for malformed key-value pair."""
    with self.assertRaisesRegex(ValueError, "Invalid key-value pair"):
      litertlm_writer.parse_metadata_string("tokenizer:key_no_equals")


if __name__ == "__main__":
  absltest.main()
