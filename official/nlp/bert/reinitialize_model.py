# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Rewrite BERT model CLS pooler weights."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags
import tensorflow as tf

from official.nlp.bert import bert_models
from official.nlp.bert import configs as bert_configs

FLAGS = flags.FLAGS


def rewrite_weights(orig_ckpt, orig_config, output_ckpt, pooler_initialization):
  """Remove vestigial pooler weights."""
  # read original checkpoint
  print(f"building model from config: [{orig_config}] ...")
  bert_config = bert_configs.BertConfig.from_json_file(orig_config)
  m = bert_models.get_transformer_encoder(
      bert_config=bert_config,
      sequence_length=1,
      output_range=1)

  print("...successfully built model.")

  print(f"\nloading model from prefix: [{orig_ckpt}] ...")
  checkpoint = tf.train.Checkpoint(model=m)
  checkpoint.restore(orig_ckpt).assert_existing_objects_matched()
  print("...successfully loaded model.")

  orig_pooler_weights, orig_pooler_bias = m.pooler_layer.weights

  print("\nupdating weights...")

  # update pooler bias
  print("  ...pooler bias with zeros.")
  new_pooler_bias = tf.constant(0.,
                                dtype=orig_pooler_bias.dtype,
                                shape=orig_pooler_bias.shape)

  # update pooler weights
  pooler_shape = orig_pooler_weights.shape
  pooler_dtype = orig_pooler_weights.dtype
  if pooler_initialization == "identity":
    print("  ...pooler weights with identity.")
    new_pooler_weights = tf.eye(pooler_shape[0], dtype=pooler_dtype)
  elif pooler_initialization == "truncated_normal":
    stddev = bert_config.initializer_range
    print("  ...pooler weights with truncated_normal "
          "(stddev={}).".format(stddev))
    new_pooler_weights = tf.random.truncated_normal(
        shape=pooler_shape, mean=0., stddev=stddev, dtype=pooler_dtype)
  else:
    raise ValueError(pooler_initialization)

  m.pooler_layer.set_weights([new_pooler_weights, new_pooler_bias])
  print("...weights updated!")

  print("\nsaving checkpoint...")
  new_checkpoint = tf.train.Checkpoint(model=m)
  # access save_counter so it is created before saving the checkpoint.
  new_checkpoint.save_counter  # pylint: disable=pointless-statement
  new_checkpoint.write(output_ckpt)
  print("... saved!")

  print(f"\nsurgery successful! new model at: [{output_ckpt}]")


def main(_):
  return rewrite_weights(FLAGS.ckpt,
                         FLAGS.bert_config_file,
                         FLAGS.output_path,
                         FLAGS.pooler_initialization)


if __name__ == "__main__":
  flags.DEFINE_string("ckpt", None,
                      "Path to BERT checkpoint.")
  flags.DEFINE_string("output_path", None,
                      "Path to output the modified checkpoint.")
  flags.DEFINE_string("bert_config_file", None,
                      "Path to bert_config.")
  flags.DEFINE_enum("pooler_initialization", "identity",
                    ["identity", "truncated_normal"],
                    "How to overwrite the pooler layer.")
  app.run(main)
