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

"""High level API for learning with TensorFlow (deprecated).

This module and all its submodules are deprecated. See
[contrib/learn/README.md](https://www.tensorflow.org/code/astronet.contrib/learn/README.md)
for migration instructions.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# pylint: disable=wildcard-import
from astronet.contrib.learn.python.learn import basic_session_run_hooks
from astronet.contrib.learn.python.learn import datasets
from astronet.contrib.learn.python.learn import estimators
from astronet.contrib.learn.python.learn import graph_actions
from astronet.contrib.learn.python.learn import learn_io as io
from astronet.contrib.learn.python.learn import models
from astronet.contrib.learn.python.learn import monitors
from astronet.contrib.learn.python.learn import ops
from astronet.contrib.learn.python.learn import preprocessing
from astronet.contrib.learn.python.learn import utils
from astronet.contrib.learn.python.learn.estimators import *
from astronet.contrib.learn.python.learn.evaluable import Evaluable
from astronet.contrib.learn.python.learn.experiment import Experiment
from astronet.contrib.learn.python.learn.export_strategy import ExportStrategy
from astronet.contrib.learn.python.learn.graph_actions import evaluate
from astronet.contrib.learn.python.learn.graph_actions import infer
from astronet.contrib.learn.python.learn.graph_actions import run_feeds
from astronet.contrib.learn.python.learn.graph_actions import run_n
from astronet.contrib.learn.python.learn.graph_actions import train
from astronet.contrib.learn.python.learn.learn_io import *
from astronet.contrib.learn.python.learn.metric_spec import MetricSpec
from astronet.contrib.learn.python.learn.monitors import NanLossDuringTrainingError
from astronet.contrib.learn.python.learn.trainable import Trainable
from astronet.contrib.learn.python.learn.utils import *
# pylint: enable=wildcard-import
