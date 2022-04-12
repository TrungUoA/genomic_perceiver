# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tokenizer implementation mapping strings to their UTF-8 bytes."""

from typing import Union
import numpy as np
import tensorflow as tf


class DNATokenizer:
  """Tokenizes string to utf-8 bytes."""

  def __init__(self, vocab_file):
    self._num_reserved_tokens = 6  # PAD, BOS, EOS, MASK, CLS, SEP
    self._vocabs = np.array( [ line.strip() for line in open(vocab_file) ] )

  def to_string(self, inputs: np.ndarray) -> str:
    return self._vocabs[ inputs.argmax(axis=-1) ]

  def to_int(self, inputs: Union[list, np.ndarray]) -> np.ndarray:
    if isinstance(inputs, list):
      inputs = np.array(inputs)
    encoded = np.where( inputs[:, None] == dna_tokenizer._vocabs[None, :] )[1]

    return encoded #tf.cast( encoded, tf.int32 ) #.astype(np.int32)
    # ### somehow the numpy_function always see the casted results as tf.int64?!

  @property
  def vocab_size(self) -> int:
    return 4102

  @property
  def pad_token(self) -> int:
    return 0

  @property
  def bos_token(self) -> int:
    return 1

  @property
  def eos_token(self) -> int:
    return 2

  @property
  def mask_token(self) -> int:
    return 3

  @property
  def cls_token(self) -> int:
    return 4

  @property
  def sep_token(self) -> int:
    return 5

dna_tokenizer = DNATokenizer(vocab_file="tokenization/vocab_6mer.txt")