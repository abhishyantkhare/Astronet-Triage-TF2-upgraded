# pylint: disable=g-import-not-at-top
# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
"""contrib module containing volatile or experimental code."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

# Add projects here, they will show up under tf.contrib.
from astronet.contrib import batching
from astronet.contrib import bayesflow
from astronet.contrib import cloud
from astronet.contrib import cluster_resolver
from astronet.contrib import coder
from astronet.contrib import compiler
from astronet.contrib import copy_graph
from astronet.contrib import crf
from astronet.contrib import cudnn_rnn
from astronet.contrib import data
from astronet.contrib import deprecated
from astronet.contrib import distribute
from astronet.contrib import distributions
from astronet.contrib import estimator
from astronet.contrib import factorization
from astronet.contrib import feature_column
from astronet.contrib import framework
from astronet.contrib import gan
from astronet.contrib import graph_editor
from astronet.contrib import grid_rnn
from astronet.contrib import image
from astronet.contrib import input_pipeline
from astronet.contrib import integrate
from astronet.contrib import keras
from astronet.contrib import kernel_methods
from astronet.contrib import kfac
from astronet.contrib import labeled_tensor
from astronet.contrib import layers
from astronet.contrib import learn
from astronet.contrib import legacy_seq2seq
from astronet.contrib import linalg
from astronet.contrib import linear_optimizer
from astronet.contrib import lookup
from astronet.contrib import losses
from astronet.contrib import memory_stats
from astronet.contrib import metrics
from astronet.contrib import model_pruning
from astronet.contrib import nccl
from astronet.contrib import nn
from astronet.contrib import opt
from astronet.contrib import periodic_resample
from astronet.contrib import predictor
from astronet.contrib import proto
from astronet.contrib import quantization
from astronet.contrib import quantize
from astronet.contrib import recurrent
from astronet.contrib import reduce_slice_ops
from astronet.contrib import resampler
from astronet.contrib import rnn
from astronet.contrib import rpc
from astronet.contrib import saved_model
from astronet.contrib import seq2seq
from astronet.contrib import signal
from astronet.contrib import slim
from astronet.contrib import solvers
from astronet.contrib import sparsemax
from astronet.contrib import staging
from astronet.contrib import stat_summarizer
from astronet.contrib import stateless
from astronet.contrib import tensor_forest
from astronet.contrib import tensorboard
from astronet.contrib import testing
from astronet.contrib import tfprof
from astronet.contrib import timeseries
from astronet.contrib import tpu
from astronet.contrib import training
from astronet.contrib import util
from astronet.contrib.eager.python import tfe as eager
if os.name != "nt":
  from astronet.contrib.lite.python import lite
from astronet.contrib.optimizer_v2 import optimizer_v2_symbols as optimizer_v2
from astronet.contrib.receptive_field import receptive_field_api as receptive_field
from astronet.contrib.remote_fused_graph import pylib as remote_fused_graph
from astronet.contrib.specs import python as specs
from astronet.contrib.summary import summary

from tensorflow.python.util.lazy_loader import LazyLoader
ffmpeg = LazyLoader("ffmpeg", globals(),
                    "astronet.contrib.ffmpeg")
del os
del LazyLoader

del absolute_import
del division
del print_function
