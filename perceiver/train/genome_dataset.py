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

""" Sample virus genome dataset collected from ncbi
"""

import enum
from typing import Any, Generator, Mapping, Optional, Sequence, Text, Tuple

import jax
import numpy as np
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds
from snp_input import *
#import tensorflow_probability as tfp

#from perceiver.train import autoaugment

#Batch = Mapping[Text, np.ndarray]
Batch = Tuple[np.ndarray, np.int32]
AUTOTUNE = tf.data.experimental.AUTOTUNE
SEED = 2022
tf.random.set_seed(SEED)

#INPUT_DIM = 224  # The number of pixels in the image resize.

KMER = 6
NUM_CLASSES = 3

def one_hot(seq, label):
  """
  Converts the label to categorical.
  Arguments ~
      seq: Tensor of Shape () - Simply for outputting
      label: Tensor of Shape (32,) for casting and converting to categorical
  Returns the image (as it was inputted) and the label converted to a categorical vector
  """
  # Casts to an Int and performs one-hot ops
  label = tf.one_hot(tf.cast(label, tf.int32), NUM_CLASSES)
  # Recasts it to Float32
  label = tf.cast(label, tf.float32)
  return seq, label

from textwrap import wrap

def read_fasta_file(filename, label):
    rawfile = open(filename, mode="r")
    instances = []
    new_instance = [ "", label ]

    for line in rawfile:
        if line[0] == '>':
            if len( new_instance[0] ) > 0:
                if len( new_instance[0] ) % KMER != 0:
                    new_instance[0] += "A" * ( KMER - len( new_instance[0] ) % KMER )
                new_instance[0] = wrap(new_instance[0], KMER)
                new_instance[0] = ' '.join( new_instance[0] )
                instances.append( new_instance )
            new_instance = [ "", label ]
        else:
            new_instance[0] += line.strip()

    return instances

def read_sample_genomes(seed):
    alpha_samples = read_fasta_file("data/coronavirus/alpha.fna", label=0)
    mers_samples = read_fasta_file("data/coronavirus/mers.fna", label=1)
    covid_samples = read_fasta_file("data/coronavirus/SARS-Cov-2.fasta", label=2)

    viral_squences = np.array(alpha_samples + mers_samples + covid_samples)
    # this shuffling shouldnot be done on a large dataset
    np.random.seed(seed)
    np.random.shuffle(viral_squences)

    all_viruses = tf.data.Dataset.from_tensor_slices( (viral_squences[:, 0], viral_squences[:, 1]) )
    return all_viruses

def max_length(data):
    max_length = 0
    #data_size = tf.data.experimental.cardinality(data)
    for gene, label in data:  # only take first element of dataset
        numpy_gene = gene.numpy()
        #counts[ labels.index(numpy_label) ] += 1
        if max_length < len(numpy_gene):
            max_length = len(numpy_gene)

    assert ( max_length - KMER ) % 7 == 0
    return int( (max_length - KMER)/7 + 1 )

#all = read_sample_genomes(SEED)
MAX_SEQ_LEN = 65803 #max_length(all)

from perceiver.dna_tokenizer import dna_tokenizer
import functools

# use decorator to input default max_len=MAX_SEQ_LEN
def kmerlist_padding(max_len):
    def wrapper_converter(func):
        @functools.wraps(func)
        def wrapper(gene_str):
            seq = func(gene_str)
            padded_seq = np.concatenate( [seq, np.repeat( dna_tokenizer.pad_token, (max_len - seq.shape[0]) ).astype(np.int32)] )
            return padded_seq
        return wrapper

    return wrapper_converter

@kmerlist_padding(max_len=MAX_SEQ_LEN)
def string_to_kmerlist(gene_str):
    gene_seq = gene_str.decode("utf-8").split(sep=' ')
    return dna_tokenizer.to_int(gene_seq)

def tokenizing_input(gene_str, label_str):
    gene = tf.numpy_function(func=string_to_kmerlist, inp=[gene_str], Tout=tf.int32)
    label = tf.numpy_function(func=lambda x: tf.cast(int(x.decode("utf-8")), tf.int32), inp=[label_str], Tout=tf.int32)

    return ( gene, label )    # convert label to int

#all = all.map(tokenizing_input) #commented to use UKBB data
#all = all.map(one_hot)  # done in class Experiment

class Split(enum.Enum):
  """ImageNet dataset split."""
  TRAIN = 1
  TRAIN_AND_VALID = 2
  VALID = 3
  TEST = 4

  @classmethod
  def from_string(cls, name: Text) -> 'Split':
    return {'TRAIN': Split.TRAIN, 'TRAIN_AND_VALID': Split.TRAIN_AND_VALID,
            'VALID': Split.VALID, 'VALIDATION': Split.VALID,
            'TEST': Split.TEST}[name.upper()]

  @property
  def num_examples(self):
    return {Split.TRAIN_AND_VALID: 536, Split.TRAIN: 469,
            Split.VALID: 67, Split.TEST: 134}[self]       # 70% train, 10% valid, 20% test


def load(
    split: str,#Split,
    *,
    is_training: bool,
    # batch_dims should be:
    # [device_count, per_device_batch_size] or [total_batch_size]
    batch_dims: Sequence[int],
    # The shape to which images are resized.
    #im_dim: int = INPUT_DIM,
    threadpool_size: int = 48,
    max_intra_op_parallelism: int = 1,
) -> Generator[Batch, None, None]:
  """Loads the given split of the dataset."""
  total_batch_size = np.prod(batch_dims)
  train, test, tokenizer, _, _ = get_data(batch_size=total_batch_size, save_pickle=True)

  # select a subset of the dataset defined by the "split"
  if split == "train":
      ds = tf.data.Dataset.from_tensor_slices((train[0], train[1]))
  elif split == "test":
      ds = tf.data.Dataset.from_tensor_slices((test[0], test[1]))

  options = tf.data.Options()
  options.threading.private_threadpool_size = threadpool_size
  options.threading.max_intra_op_parallelism = (
      max_intra_op_parallelism)
  options.experimental_optimization.map_parallelization = True
  if is_training:
    options.experimental_deterministic = False
  ds = ds.with_options(options)

  if is_training:
    if jax.process_count() > 1:
      # Only cache if we are reading a subset of the dataset.
      ds = ds.cache()
    ds = ds.repeat()
    ds = ds.shuffle(buffer_size=10 * total_batch_size, seed=SEED)

  else:
    if test[0].shape[0] % total_batch_size != 0:
      raise ValueError(f'Test/valid must be divisible by {total_batch_size}')

  for batch_size in reversed(batch_dims):
    ds = ds.batch(batch_size)

  ds = ds.prefetch(AUTOTUNE)

  yield from tfds.as_numpy(ds)

def _shard(
    split: Split, shard_index: int, num_shards: int) -> Tuple[int, int]:
  """Returns [start, end) for the given shard index."""
  assert shard_index < num_shards
  arange = np.arange(split.num_examples)
  shard_range = np.array_split(arange, num_shards)[shard_index]
  start, end = shard_range[0], (shard_range[-1] + 1)
  if split == Split.TRAIN:
    # Note that our TRAIN=TFDS_TRAIN[10000:] and VALID=TFDS_TRAIN[:10000].
    offset = Split.VALID.num_examples
    start += offset
    end += offset
  return start, end
