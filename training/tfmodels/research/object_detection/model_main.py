# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Binary to run train and evaluation on object detection model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import functools
print = functools.partial(print, flush=True)
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]="3"
from absl import flags
from pathlib import Path
import tensorflow as tf
from tensorflow.python.platform import tf_logging as logging

from object_detection import model_hparams
from object_detection import model_lib
from best_checkpoint_copier import BestCheckpointCopier

flags.DEFINE_string(
    'model_dir', None, 'Path to output model directory '
    'where event and checkpoint files will be written.')
flags.DEFINE_string('pipeline_config_path', None, 'Path to pipeline config '
                    'file.')
flags.DEFINE_integer('num_train_steps', None, 'Number of train steps.')
flags.DEFINE_boolean('eval_training_data', False,
                     'If training data should be evaluated for this job. Note '
                     'that one call only use this in eval-only mode, and '
                     '`checkpoint_dir` must be supplied.')
flags.DEFINE_integer('sample_1_of_n_eval_examples', 1, 'Will sample one of '
                     'every n eval input examples, where n is provided.')
flags.DEFINE_integer('sample_1_of_n_eval_on_train_examples', 5, 'Will sample '
                     'one of every n train input examples for evaluation, '
                     'where n is provided. This is only used if '
                     '`eval_training_data` is True.')
flags.DEFINE_string(
    'hparams_overrides', None, 'Hyperparameter overrides, '
    'represented as a string containing comma-separated '
    'hparam_name=value pairs.')
flags.DEFINE_string(
    'checkpoint_dir', None, 'Path to directory holding a checkpoint.  If '
    '`checkpoint_dir` is provided, this binary operates in eval-only mode, '
    'writing resulting metrics to `model_dir`.')
flags.DEFINE_string(
    'checkpoint_path', None, 'Path to a specific checkpoint file.  If '
    '`checkpoint_path` is provided, this binary operates in eval-only mode, '
    'writing resulting metrics to `model_dir`.')
flags.DEFINE_boolean(
    'run_once', False, 'If running in eval-only mode, whether to run just '
    'one round of eval vs running continuously (default).'
)
flags.DEFINE_integer(
    "save_checkpoints_steps", 500, "Save checkpoints every"
)
flags.DEFINE_integer(
    "eval_throttle_secs", 600, """Do not re-evaluate unless the last evaluation was
        started at least this many seconds ago. Of course, evaluation does not
        occur if no new checkpoints are available, hence, this is the minimum."""
)
FLAGS = flags.FLAGS


def main(unused_argv):
  flags.mark_flag_as_required('model_dir')
  flags.mark_flag_as_required('pipeline_config_path')
  config = tf.estimator.RunConfig(
      model_dir=FLAGS.model_dir
      , save_checkpoints_secs=None
      , save_checkpoints_steps=FLAGS.save_checkpoints_steps
  )

  train_and_eval_dict = model_lib.create_estimator_and_inputs(
      run_config=config,
      hparams=model_hparams.create_hparams(FLAGS.hparams_overrides),
      pipeline_config_path=FLAGS.pipeline_config_path,
      train_steps=FLAGS.num_train_steps,
      sample_1_of_n_eval_examples=FLAGS.sample_1_of_n_eval_examples,
      sample_1_of_n_eval_on_train_examples=(
          FLAGS.sample_1_of_n_eval_on_train_examples))
  estimator = train_and_eval_dict['estimator']
  train_input_fn = train_and_eval_dict['train_input_fn']
  eval_input_fns = train_and_eval_dict['eval_input_fns']
  eval_on_train_input_fn = train_and_eval_dict['eval_on_train_input_fn']
  predict_input_fn = train_and_eval_dict['predict_input_fn']
  train_steps = train_and_eval_dict['train_steps']

  if FLAGS.checkpoint_dir or FLAGS.checkpoint_path:
    if FLAGS.eval_training_data:
      name = 'training_data'
      input_fn = eval_on_train_input_fn
    else:
      name = 'validation_data'
      # The first eval input will be evaluated.
      input_fn = eval_input_fns[0]
    if FLAGS.run_once:
      if FLAGS.checkpoint_path:
        ckpt_path = Path(FLAGS.checkpoint_path).resolve()
        print("Using checkpoint: ", str(ckpt_path), flush=True)
        logging.info(f"USING CHECKPOINT: {str(ckpt_path)}")
        estimator.evaluate(input_fn,
                              steps=None,
                              checkpoint_path=str(ckpt_path))
      elif FLAGS.checkpoint_dir:
        print("USING CHECKPOINT_DIR: ", FLAGS.checkpoint_dir, flush=True)
        logging.info(f"USING CHECKPOINT_DIR: {FLAGS.checkpoint_dir}")
        estimator.evaluate(input_fn,
                              steps=None,
                              checkpoint_path=tf.train.latest_checkpoint(
                                  FLAGS.checkpoint_dir))
    else:
      model_lib.continuous_eval(estimator, FLAGS.checkpoint_dir, input_fn,
                                train_steps, name)
  else:
    train_spec, eval_specs = model_lib.create_train_and_eval_specs(
        train_input_fn,
        eval_input_fns,
        eval_on_train_input_fn,
        predict_input_fn,
        train_steps,
        eval_on_train_data=False,
        eval_throttle_secs=FLAGS.eval_throttle_secs)

    # Currently only a single Eval Spec is allowed.
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_specs[0])


if __name__ == '__main__':
  tf.app.run()
