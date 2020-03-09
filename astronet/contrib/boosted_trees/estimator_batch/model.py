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
"""GTFlow Model definitions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy

from astronet.contrib import learn
from astronet.contrib.boosted_trees.estimator_batch import estimator_utils
from astronet.contrib.boosted_trees.estimator_batch import trainer_hooks
from astronet.contrib.boosted_trees.python.ops import model_ops
from astronet.contrib.boosted_trees.python.training.functions import gbdt_batch
from tensorflow.core.protobuf import config_pb2
from tensorflow.python.framework import ops
from tensorflow.python.ops import state_ops
from tensorflow.python.training import training_util
from google.protobuf import text_format
from astronet.contrib.boosted_trees.proto import tree_config_pb2


class ModelBuilderOutputType(object):
  MODEL_FN_OPS = 0
  ESTIMATOR_SPEC = 1


def model_builder(features,
                  labels,
                  mode,
                  params,
                  config,
                  output_type=ModelBuilderOutputType.MODEL_FN_OPS):
  """Multi-machine batch gradient descent tree model.

  Args:
    features: `Tensor` or `dict` of `Tensor` objects.
    labels: Labels used to train on.
    mode: Mode we are in. (TRAIN/EVAL/INFER)
    params: A dict of hyperparameters.
      The following hyperparameters are expected:
      * head: A `Head` instance.
      * learner_config: A config for the learner.
      * feature_columns: An iterable containing all the feature columns used by
          the model.
      * examples_per_layer: Number of examples to accumulate before growing a
          layer. It can also be a function that computes the number of examples
          based on the depth of the layer that's being built.
      * weight_column_name: The name of weight column.
      * center_bias: Whether a separate tree should be created for first fitting
          the bias.
      * override_global_step_value: If after the training is done, global step
        value must be reset to this value. This is particularly useful for hyper
        parameter tuning, which can't recognize early stopping due to the number
        of trees. If None, no override of global step will happen.
    config: `RunConfig` of the estimator.
    output_type: Whether to return ModelFnOps (old interface) or EstimatorSpec
      (new interface).

  Returns:
    A `ModelFnOps` object.
  Raises:
    ValueError: if inputs are not valid.
  """
  head = params["head"]
  learner_config = params["learner_config"]
  examples_per_layer = params["examples_per_layer"]
  feature_columns = params["feature_columns"]
  weight_column_name = params["weight_column_name"]
  num_trees = params["num_trees"]
  use_core_libs = params["use_core_libs"]
  logits_modifier_function = params["logits_modifier_function"]
  output_leaf_index = params["output_leaf_index"]
  override_global_step_value = params.get("override_global_step_value", None)
  num_quantiles = params["num_quantiles"]

  if features is None:
    raise ValueError("At least one feature must be specified.")

  if config is None:
    raise ValueError("Missing estimator RunConfig.")
  if config.session_config is not None:
    session_config = config.session_config
    session_config.allow_soft_placement = True
  else:
    session_config = config_pb2.ConfigProto(allow_soft_placement=True)
  config = config.replace(session_config=session_config)

  center_bias = params["center_bias"]

  if isinstance(features, ops.Tensor):
    features = {features.name: features}

  # Make a shallow copy of features to ensure downstream usage
  # is unaffected by modifications in the model function.
  training_features = copy.copy(features)
  training_features.pop(weight_column_name, None)
  global_step = training_util.get_global_step()

  initial_ensemble = ""
  if learner_config.each_tree_start.nodes:
    if learner_config.each_tree_start_num_layers <= 0:
      raise ValueError("You must provide each_tree_start_num_layers.")
    num_layers = learner_config.each_tree_start_num_layers
    initial_ensemble = """
             trees { %s }
             tree_weights: 0.1
             tree_metadata {
              num_tree_weight_updates: 1
              num_layers_grown: %d
              is_finalized: false
             }
             """ % (text_format.MessageToString(
                 learner_config.each_tree_start), num_layers)
    tree_ensemble_proto = tree_config_pb2.DecisionTreeEnsembleConfig()
    text_format.Merge(initial_ensemble, tree_ensemble_proto)
    initial_ensemble = tree_ensemble_proto.SerializeToString()

  with ops.device(global_step.device):
    ensemble_handle = model_ops.tree_ensemble_variable(
        stamp_token=0,
        tree_ensemble_config=initial_ensemble,  # Initialize the ensemble.
        name="ensemble_model")

  # Create GBDT model.
  gbdt_model = gbdt_batch.GradientBoostedDecisionTreeModel(
      is_chief=config.is_chief,
      num_ps_replicas=config.num_ps_replicas,
      ensemble_handle=ensemble_handle,
      center_bias=center_bias,
      examples_per_layer=examples_per_layer,
      learner_config=learner_config,
      feature_columns=feature_columns,
      logits_dimension=head.logits_dimension,
      features=training_features,
      use_core_columns=use_core_libs,
      output_leaf_index=output_leaf_index,
      num_quantiles=num_quantiles)
  with ops.name_scope("gbdt", "gbdt_optimizer"):
    predictions_dict = gbdt_model.predict(mode)
    logits = predictions_dict["predictions"]
    if logits_modifier_function:
      logits = logits_modifier_function(logits, features, mode)

    def _train_op_fn(loss):
      """Returns the op to optimize the loss."""
      update_op = gbdt_model.train(loss, predictions_dict, labels)
      with ops.control_dependencies(
          [update_op]), (ops.colocate_with(global_step)):
        update_op = state_ops.assign_add(global_step, 1).op
        return update_op

  create_estimator_spec_op = getattr(head, "create_estimator_spec", None)

  training_hooks = []
  if num_trees:
    if center_bias:
      num_trees += 1

    finalized_trees, attempted_trees = gbdt_model.get_number_of_trees_tensor()
    training_hooks.append(
        trainer_hooks.StopAfterNTrees(num_trees, attempted_trees,
                                      finalized_trees,
                                      override_global_step_value))

  if output_type == ModelBuilderOutputType.MODEL_FN_OPS:
    if use_core_libs and callable(create_estimator_spec_op):
      model_fn_ops = head.create_estimator_spec(
          features=features,
          mode=mode,
          labels=labels,
          train_op_fn=_train_op_fn,
          logits=logits)
      model_fn_ops = estimator_utils.estimator_spec_to_model_fn_ops(
          model_fn_ops)
    else:
      model_fn_ops = head.create_model_fn_ops(
          features=features,
          mode=mode,
          labels=labels,
          train_op_fn=_train_op_fn,
          logits=logits)

    if output_leaf_index and gbdt_batch.LEAF_INDEX in predictions_dict:
      model_fn_ops.predictions[gbdt_batch.LEAF_INDEX] = predictions_dict[
          gbdt_batch.LEAF_INDEX]

    model_fn_ops.training_hooks.extend(training_hooks)
    return model_fn_ops
  elif output_type == ModelBuilderOutputType.ESTIMATOR_SPEC:
    assert callable(create_estimator_spec_op)
    estimator_spec = head.create_estimator_spec(
        features=features,
        mode=mode,
        labels=labels,
        train_op_fn=_train_op_fn,
        logits=logits)

    if output_leaf_index and gbdt_batch.LEAF_INDEX in predictions_dict:
      estimator_spec.predictions[gbdt_batch.LEAF_INDEX] = predictions_dict[
          gbdt_batch.LEAF_INDEX]

    estimator_spec = estimator_spec._replace(
        training_hooks=training_hooks + list(estimator_spec.training_hooks))
    return estimator_spec

  return model_fn_ops


def ranking_model_builder(features,
                          labels,
                          mode,
                          params,
                          config,
                          output_type=ModelBuilderOutputType.MODEL_FN_OPS):
  """Multi-machine batch gradient descent tree model for ranking.

  Args:
    features: `Tensor` or `dict` of `Tensor` objects.
    labels: Labels used to train on.
    mode: Mode we are in. (TRAIN/EVAL/INFER)
    params: A dict of hyperparameters.
      The following hyperparameters are expected:
      * head: A `Head` instance.
      * learner_config: A config for the learner.
      * feature_columns: An iterable containing all the feature columns used by
          the model.
      * examples_per_layer: Number of examples to accumulate before growing a
          layer. It can also be a function that computes the number of examples
          based on the depth of the layer that's being built.
      * weight_column_name: The name of weight column.
      * center_bias: Whether a separate tree should be created for first fitting
          the bias.
      * ranking_model_pair_keys (Optional): Keys to distinguish between features
        for left and right part of the training pairs for ranking. For example,
        for an Example with features "a.f1" and "b.f1", the keys would be
        ("a", "b").
      * override_global_step_value: If after the training is done, global step
        value must be reset to this value. This is particularly useful for hyper
        parameter tuning, which can't recognize early stopping due to the number
        of trees. If None, no override of global step will happen.
    config: `RunConfig` of the estimator.
    output_type: Whether to return ModelFnOps (old interface) or EstimatorSpec
      (new interface).


  Returns:
    A `ModelFnOps` object.
  Raises:
    ValueError: if inputs are not valid.
  """
  head = params["head"]
  learner_config = params["learner_config"]
  examples_per_layer = params["examples_per_layer"]
  feature_columns = params["feature_columns"]
  weight_column_name = params["weight_column_name"]
  num_trees = params["num_trees"]
  use_core_libs = params["use_core_libs"]
  logits_modifier_function = params["logits_modifier_function"]
  output_leaf_index = params["output_leaf_index"]
  ranking_model_pair_keys = params["ranking_model_pair_keys"]
  override_global_step_value = params.get("override_global_step_value", None)
  num_quantiles = params["num_quantiles"]

  if features is None:
    raise ValueError("At least one feature must be specified.")

  if config is None:
    raise ValueError("Missing estimator RunConfig.")

  center_bias = params["center_bias"]

  if isinstance(features, ops.Tensor):
    features = {features.name: features}

  # Make a shallow copy of features to ensure downstream usage
  # is unaffected by modifications in the model function.
  training_features = copy.copy(features)
  training_features.pop(weight_column_name, None)
  global_step = training_util.get_global_step()
  with ops.device(global_step.device):
    ensemble_handle = model_ops.tree_ensemble_variable(
        stamp_token=0,
        tree_ensemble_config="",  # Initialize an empty ensemble.
        name="ensemble_model")

  # Extract the features.
  if mode == learn.ModeKeys.TRAIN or mode == learn.ModeKeys.EVAL:
    # For ranking pairwise training, we extract two sets of features.
    if len(ranking_model_pair_keys) != 2:
      raise ValueError("You must provide keys for ranking.")
    left_pair_key = ranking_model_pair_keys[0]
    right_pair_key = ranking_model_pair_keys[1]
    if left_pair_key is None or right_pair_key is None:
      raise ValueError("Both pair keys should be provided for ranking.")

    features_1 = {}
    features_2 = {}
    for name in training_features:
      feature = training_features[name]
      new_name = name[2:]
      if name.startswith(left_pair_key + "."):
        features_1[new_name] = feature
      else:
        assert name.startswith(right_pair_key + ".")
        features_2[new_name] = feature

    main_features = features_1
    supplementary_features = features_2
  else:
    # For non-ranking or inference ranking, we have only 1 set of features.
    main_features = training_features

  # Create GBDT model.
  gbdt_model_main = gbdt_batch.GradientBoostedDecisionTreeModel(
      is_chief=config.is_chief,
      num_ps_replicas=config.num_ps_replicas,
      ensemble_handle=ensemble_handle,
      center_bias=center_bias,
      examples_per_layer=examples_per_layer,
      learner_config=learner_config,
      feature_columns=feature_columns,
      logits_dimension=head.logits_dimension,
      features=main_features,
      use_core_columns=use_core_libs,
      output_leaf_index=output_leaf_index,
      num_quantiles=num_quantiles)

  with ops.name_scope("gbdt", "gbdt_optimizer"):
    # Logits for inference.
    if mode == learn.ModeKeys.INFER:
      predictions_dict = gbdt_model_main.predict(mode)
      logits = predictions_dict[gbdt_batch.PREDICTIONS]
      if logits_modifier_function:
        logits = logits_modifier_function(logits, features, mode)
    else:
      gbdt_model_supplementary = gbdt_batch.GradientBoostedDecisionTreeModel(
          is_chief=config.is_chief,
          num_ps_replicas=config.num_ps_replicas,
          ensemble_handle=ensemble_handle,
          center_bias=center_bias,
          examples_per_layer=examples_per_layer,
          learner_config=learner_config,
          feature_columns=feature_columns,
          logits_dimension=head.logits_dimension,
          features=supplementary_features,
          use_core_columns=use_core_libs,
          output_leaf_index=output_leaf_index)

      # Logits for train and eval.
      if not supplementary_features:
        raise ValueError("Features for ranking must be specified.")

      predictions_dict_1 = gbdt_model_main.predict(mode)
      predictions_1 = predictions_dict_1[gbdt_batch.PREDICTIONS]

      predictions_dict_2 = gbdt_model_supplementary.predict(mode)
      predictions_2 = predictions_dict_2[gbdt_batch.PREDICTIONS]

      logits = predictions_1 - predictions_2
      if logits_modifier_function:
        logits = logits_modifier_function(logits, features, mode)

      predictions_dict = predictions_dict_1
      predictions_dict[gbdt_batch.PREDICTIONS] = logits

    def _train_op_fn(loss):
      """Returns the op to optimize the loss."""
      update_op = gbdt_model_main.train(loss, predictions_dict, labels)
      with ops.control_dependencies(
          [update_op]), (ops.colocate_with(global_step)):
        update_op = state_ops.assign_add(global_step, 1).op
        return update_op

  create_estimator_spec_op = getattr(head, "create_estimator_spec", None)

  training_hooks = []
  if num_trees:
    if center_bias:
      num_trees += 1

    finalized_trees, attempted_trees = (
        gbdt_model_main.get_number_of_trees_tensor())
    training_hooks.append(
        trainer_hooks.StopAfterNTrees(num_trees, attempted_trees,
                                      finalized_trees,
                                      override_global_step_value))

  if output_type == ModelBuilderOutputType.MODEL_FN_OPS:
    if use_core_libs and callable(create_estimator_spec_op):
      model_fn_ops = head.create_estimator_spec(
          features=features,
          mode=mode,
          labels=labels,
          train_op_fn=_train_op_fn,
          logits=logits)
      model_fn_ops = estimator_utils.estimator_spec_to_model_fn_ops(
          model_fn_ops)
    else:
      model_fn_ops = head.create_model_fn_ops(
          features=features,
          mode=mode,
          labels=labels,
          train_op_fn=_train_op_fn,
          logits=logits)

    if output_leaf_index and gbdt_batch.LEAF_INDEX in predictions_dict:
      model_fn_ops.predictions[gbdt_batch.LEAF_INDEX] = predictions_dict[
          gbdt_batch.LEAF_INDEX]

    model_fn_ops.training_hooks.extend(training_hooks)
    return model_fn_ops

  elif output_type == ModelBuilderOutputType.ESTIMATOR_SPEC:
    assert callable(create_estimator_spec_op)
    estimator_spec = head.create_estimator_spec(
        features=features,
        mode=mode,
        labels=labels,
        train_op_fn=_train_op_fn,
        logits=logits)

    estimator_spec = estimator_spec._replace(
        training_hooks=training_hooks + list(estimator_spec.training_hooks))
    return estimator_spec

  return model_fn_ops
