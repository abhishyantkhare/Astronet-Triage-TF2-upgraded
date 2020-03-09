# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""Slim is an interface to contrib functions, examples and models.

TODO(nsilberman): flesh out documentation.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# pylint: disable=unused-import,line-too-long,g-importing-member,wildcard-import
# TODO(jart): Delete non-slim imports
from astronet.contrib import losses
from astronet.contrib import metrics
from astronet.contrib.framework.python.ops.arg_scope import *
from astronet.contrib.framework.python.ops.variables import *
from astronet.contrib.layers.python.layers import *
from astronet.contrib.layers.python.layers.initializers import *
from astronet.contrib.layers.python.layers.regularizers import *
from astronet.contrib.slim.python.slim import evaluation
from astronet.contrib.slim.python.slim import learning
from astronet.contrib.slim.python.slim import model_analyzer
from astronet.contrib.slim.python.slim import queues
from astronet.contrib.slim.python.slim import summaries
from astronet.contrib.slim.python.slim.data import data_decoder
from astronet.contrib.slim.python.slim.data import data_provider
from astronet.contrib.slim.python.slim.data import dataset
from astronet.contrib.slim.python.slim.data import dataset_data_provider
from astronet.contrib.slim.python.slim.data import parallel_reader
from astronet.contrib.slim.python.slim.data import prefetch_queue
from astronet.contrib.slim.python.slim.data import tfexample_decoder
from tensorflow.python.util.all_util import make_all
# pylint: enable=unused-import,line-too-long,g-importing-member,wildcard-import

__all__ = make_all(__name__)
