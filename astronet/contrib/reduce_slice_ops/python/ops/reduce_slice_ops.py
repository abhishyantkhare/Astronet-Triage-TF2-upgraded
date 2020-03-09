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
"""Python wrapper for the reduce slice operators."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from astronet.contrib.reduce_slice_ops.ops import gen_reduce_slice_ops
from astronet.contrib.util import loader
from tensorflow.python.platform import resource_loader


_reduce_slice_ops = loader.load_op_library(
    resource_loader.get_path_to_datafile("_reduce_slice_ops.so"))


reduce_slice_sum = gen_reduce_slice_ops.reduce_slice_sum
reduce_slice_prod = gen_reduce_slice_ops.reduce_slice_prod
reduce_slice_max = gen_reduce_slice_ops.reduce_slice_max
reduce_slice_min = gen_reduce_slice_ops.reduce_slice_min