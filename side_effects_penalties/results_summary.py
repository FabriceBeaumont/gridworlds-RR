# Copyright 2019 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Plot results for different side effects penalties.

Loads csv result files generated by `run_experiment' and outputs a summary data
frame in a csv file to be used for plotting by plot_results.ipynb.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
from absl import app
from absl import flags
import pandas as pd

from file_loading import load_files


FLAGS = flags.FLAGS

"""Ignore for now. 
Probably only used to read and visualize the (best) stored results.
"""

if __name__ == '__main__':  # Avoid defining flags when used as a library.
  flags.DEFINE_string('path', '', 'File path.')
  flags.DEFINE_string('input_suffix', '',
                      'Filename suffix to use when loading data files.')
  flags.DEFINE_string('output_suffix', '',
                      'Filename suffix to use when saving files.')
  flags.DEFINE_bool('bar_plot', True,
                    'Make a data frame for a bar plot (True) ' +
                    'or learning curves (False)')
  flags.DEFINE_string('env_name', 'box', 'Environment name.')
  flags.DEFINE_bool('noops', True, 'Whether the environment includes noops.')
  flags.DEFINE_list('beta_list', [0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0],
                    'List of beta values.')
  flags.DEFINE_list('seed_list', [1], 'List of random seeds.')
  flags.DEFINE_bool('compare_penalties', True,
                    'Compare different penalties using the best beta value ' +
                    'for each penalty (True), or compare different beta values '
                    + 'for the same penalty (False).')
  flags.DEFINE_enum('dev_measure', 'rel_reach',
                    ['none', 'reach', 'rel_reach', 'att_util'],
                    'Deviation measure (used if compare_penalties=False).')
  flags.DEFINE_enum('dev_fun', 'truncation', ['truncation', 'absolute'],
                    'Summary function for the deviation measure ' +
                    '(used if compare_penalties=False)')
  flags.DEFINE_float('value_discount', 0.99,
                     'Discount factor for deviation measure value function ' +
                     '(used if compare_penalties=False)')


def beta_choice(baseline, dev_measure, dev_fun, value_discount, env_name,
                beta_list, seed_list, noops=False, path='', suffix=''):
  """Choose beta value that gives the highest final performance."""
  if dev_measure == 'none':
    return 0.1
  perf_max = float('-inf')
  best_beta = 0.0
  for beta in beta_list:
    df = load_files(baseline=baseline, dev_measure=dev_measure,
                    dev_fun=dev_fun, value_discount=value_discount, beta=beta,
                    env_name=env_name, noops=noops, path=path, suffix=suffix,
                    seed_list=seed_list)
    if df.empty:
      perf = float('-inf')
    else:
      perf = df['performance_smooth'].mean()
    if perf > perf_max:
      perf_max = perf
      best_beta = beta
  return best_beta


def penalty_label(dev_measure, dev_fun, value_discount):
  """Penalty label specifying design choices."""
  dev_measure_labels = {
      'none': 'None', 'rel_reach': 'RR', 'att_util': 'AU', 'reach': 'UR'}
  label = dev_measure_labels[dev_measure]
  disc_lab = 'u' if value_discount == 1.0 else 'd'
  dev_lab = ''
  if dev_measure in ['rel_reach', 'att_util']:
    dev_lab = 't' if dev_fun == 'truncation' else 'a'
  if dev_measure != 'none':
    label = label + '(' + disc_lab + dev_lab + ')'
  return label


def make_summary_data_frame(
    env_name, beta_list, seed_list, final=True, baseline=None, dev_measure=None,
    dev_fun=None, value_discount=None, noops=False, compare_penalties=True,
    path='', input_suffix='', output_suffix=''):
  """Make summary dataframe from multiple csv result files and output to csv."""
  # For each of the penalty parameters (baseline, dev_measure, dev_fun, and
  # value_discount), compare a list of multiple values if the parameter is None,
  # or use the provided parameter value if it is not None
  baseline_list = ['start', 'inaction', 'stepwise', 'step_noroll']
  if dev_measure is not None:
    dev_measure_list = [dev_measure]
  else:
    dev_measure_list = ['none', 'reach', 'rel_reach', 'att_util']
  dataframes = []
  for dev_measure in dev_measure_list:
    # These deviation measures don't have a deviation function:
    if dev_measure in ['reach', 'none']:
      dev_fun_list = ['none']
    elif dev_fun is not None:
      dev_fun_list = [dev_fun]
    else:
      dev_fun_list = ['truncation', 'absolute']
    # These deviation measures must be discounted:
    if dev_measure in ['none', 'att_util']:
      value_discount_list = [0.99]
    elif value_discount is not None:
      value_discount_list = [value_discount]
    else:
      value_discount_list = [0.99, 1.0]
    for baseline in baseline_list:
      for vd in value_discount_list:
        for devf in dev_fun_list:
          # Choose the best beta for this set of penalty parameters if
          # compare_penalties=True, or compare all betas otherwise
          if compare_penalties:
            beta = beta_choice(
                baseline=baseline, dev_measure=dev_measure, dev_fun=devf,
                value_discount=vd, env_name=env_name, noops=noops,
                beta_list=beta_list, seed_list=seed_list, path=path,
                suffix=input_suffix)
            betas = [beta]
          else:
            betas = beta_list
          for beta in betas:
            label = penalty_label(
                dev_measure=dev_measure, dev_fun=devf, value_discount=vd)
            df_part = load_files(
                baseline=baseline, dev_measure=dev_measure, dev_fun=devf,
                value_discount=vd, beta=beta, env_name=env_name,
                noops=noops, path=path, suffix=input_suffix, final=final,
                seed_list=seed_list)
            df_part = df_part.assign(
                baseline=baseline, dev_measure=dev_measure, dev_fun=devf,
                value_discount=vd, beta=beta, env_name=env_name, label=label)
            dataframes.append(df_part)
  df = pd.concat(dataframes, sort=False)
  # Output summary data frame
  final_str = '_final' if final else ''
  if compare_penalties:
    filename = ('df_summary_penalties_' + env_name + final_str +
                output_suffix + '.csv')
  else:
    filename = ('df_summary_betas_' + env_name + '_' + dev_measure + '_' +
                dev_fun + '_' + str(value_discount) + final_str + output_suffix
                + '.csv')
  f = os.path.join(path, filename)
  df.to_csv(f)
  return df


def main(unused_argv):
  compare_penalties = FLAGS.compare_penalties
  dev_measure = None if compare_penalties else FLAGS.dev_measure
  dev_fun = None if compare_penalties else FLAGS.dev_fun
  value_discount = None if compare_penalties else FLAGS.value_discount
  make_summary_data_frame(
      compare_penalties=compare_penalties, env_name=FLAGS.env_name,
      noops=FLAGS.noops, final=FLAGS.bar_plot, dev_measure=dev_measure,
      value_discount=value_discount, dev_fun=dev_fun, path=FLAGS.path,
      input_suffix=FLAGS.input_suffix, output_suffix=FLAGS.output_suffix,
      beta_list=FLAGS.beta_list, seed_list=FLAGS.seed_list)


if __name__ == '__main__':
  app.run(main)