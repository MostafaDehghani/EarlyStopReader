from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from six.moves import range  # pylint: disable=redefined-builtin
from tensor2tensor.data_generators import generator_utils as utils
from tensor2tensor.data_generators import text_encoder
from tensor2tensor.utils import metrics
from tensor2tensor.utils import registry
from tensor2tensor.data_generators import algorithmic


@registry.register_problem
class AlgorithmicSortedStringMatching(algorithmic.AlgorithmicProblem):
  """Problem spec for checking if a symbol has occurred in a sorted string.
  """

  @property
  def num_symbols(self):
    return 10

  @property
  def train_length(self):
    return 100

  @property
  def dev_length(self):
    return self.train_length * 2

  def generate_data(self, data_dir, _, task_id=-1):

    def generator_eos(nbr_symbols, max_length, nbr_cases):
      """Shift by NUM_RESERVED_IDS and append EOS token."""
      for case in self.generator(nbr_symbols, max_length, nbr_cases):
        new_case = {}
        for feature in case:
          if feature not in ["targets"]:
            new_case[feature] = [
                i + text_encoder.NUM_RESERVED_TOKENS for i in case[feature]
            ] + [text_encoder.EOS_ID]
          else:
            new_case[feature] = case[feature]
        yield new_case

    utils.generate_dataset_and_shuffle(
        generator_eos(self.num_symbols, self.train_length, self.train_size),
        self.training_filepaths(data_dir, self.num_shards, shuffled=True),
        generator_eos(self.num_symbols, self.dev_length, self.dev_size),
        self.dev_filepaths(data_dir, 1, shuffled=True),
        shuffle=False)

  def generator(self, nbr_symbols, max_length, nbr_cases):
    """Generating for sorted string matching task on sequence of symbols.

    The length of the sequence is drawn uniformly at random from [1, max_length]
    and then symbols are drawn (uniquely w/ or w/o replacement) uniformly at
    random from [0, nbr_symbols) until nbr_cases sequences have been produced.
    Then the input sequence is sorted. One of the symbols is chosen randomly as
    the target sub-string and finally the target is set to be the number of
    occurances of the target symbol in the input sequence.

    Args:
      nbr_symbols: number of symbols to use in each sequence.
      max_length: integer, maximum length of sequences to generate.
      nbr_cases: the number of cases to generate.

    Yields:
      A dictionary {"inputs": input-list, "targets": target-list} where
      target-list is input-list sorted.
    """
    for _ in range(nbr_cases):
      # Sample the sequence length.
      input_string_length = np.random.randint(max_length) + 1
      input_string = list(np.random.randint(nbr_symbols, size=input_string_length))
      input_string = list(sorted(input_string))
      input_symbole_index = np.random.randint(input_string_length)
      input_symbole = input_string[input_symbole_index]
      keep_prob = np.random.random_sample()

      if keep_prob < 0.5:
        targets = [1]
      else:
        input_string = list(filter(lambda s: s != input_symbole, input_string))
        targets = [0]


      inputs  =  [int(inp) for inp in ([input_symbole] + input_string)]
      yield {"inputs": inputs, "targets": targets}

  def eval_metrics(self):
    return [metrics.Metrics.ACC]

  def hparams(self, defaults, unused_model_hparams):
    p = defaults
    vocab_size = self.num_symbols + text_encoder.NUM_RESERVED_TOKENS
    p.input_modality = {"inputs": (registry.Modalities.SYMBOL, vocab_size)}
    p.target_modality = (registry.Modalities.CLASS_LABEL, 2)
