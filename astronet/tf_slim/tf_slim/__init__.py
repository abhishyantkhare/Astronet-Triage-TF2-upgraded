# coding=utf-8
# coding=utf-8
# Copyright 2016 The TF-Slim Authors. All Rights Reserved.
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
"""Slim is an interface to TF functions, examples and models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# pylint:disable=g-bad-import-order,wildcard-import

from astronet.tf_slim.tf_slim.layers import *
from astronet.tf_slim.tf_slim.ops.variables import *


from astronet.tf_slim.tf_slim.ops import arg_scope as arg_scope_lib

# Move explicit import after wild-card imports to prevent accidental
# overwrites (such as summaries). Also use import x syntax, to directly
# access file, rather than previously imported symbols.

from astronet.tf_slim.tf_slim.data import data_decoder
from astronet.tf_slim.tf_slim.data import data_provider
from astronet.tf_slim.tf_slim.data import dataset
from astronet.tf_slim.tf_slim.data import dataset_data_provider
from astronet.tf_slim.tf_slim.data import parallel_reader
from astronet.tf_slim.tf_slim.data import prefetch_queue
from astronet.tf_slim.tf_slim.data import tfexample_decoder

import astronet.tf_slim.tf_slim.evaluation as evaluation
import astronet.tf_slim.tf_slim.learning as learning
import astronet.tf_slim.tf_slim.losses as losses
import astronet.tf_slim.tf_slim.metrics as metrics

import astronet.tf_slim.tf_slim.model_analyzer as model_analyzer
import astronet.tf_slim.tf_slim.queues as queues
import astronet.tf_slim.tf_slim.summaries as summaries

from astronet.tf_slim.tf_slim.training import evaluation as train_eval
from astronet.tf_slim.tf_slim.training import training as training

arg_scope = arg_scope_lib.arg_scope
add_arg_scope = arg_scope_lib.add_arg_scope
current_arg_scope = arg_scope_lib.current_arg_scope
has_arg_scope = arg_scope_lib.has_arg_scope
arg_scoped_arguments = arg_scope_lib.arg_scoped_arguments
arg_scope_func_key = arg_scope_lib.arg_scope_func_key
