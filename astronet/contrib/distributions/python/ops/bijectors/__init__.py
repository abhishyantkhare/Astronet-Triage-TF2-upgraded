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
"""Bijector Ops.

Use [tfp.bijectors](/probability/api_docs/python/tfp/bijectors) instead.

@@AbsoluteValue
@@Affine
@@AffineLinearOperator
@@AffineScalar
@@Bijector
@@BatchNormalization
@@Chain
@@CholeskyOuterProduct
@@ConditionalBijector
@@Exp
@@FillTriangular
@@Gumbel
@@Identity
@@Inline
@@Invert
@@Kumaraswamy
@@MaskedAutoregressiveFlow
@@MatrixInverseTriL
@@Ordered
@@Permute
@@PowerTransform
@@RealNVP
@@Reshape
@@ScaleTriL
@@Sigmoid
@@SinhArcsinh
@@SoftmaxCentered
@@Softplus
@@Softsign
@@Square
@@TransformDiagonal
@@Weibull

@@masked_autoregressive_default_template
@@masked_dense
@@real_nvp_default_template
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# pylint: disable=unused-import,wildcard-import,line-too-long,g-importing-member

from astronet.contrib.distributions.python.ops.bijectors.absolute_value import *
from astronet.contrib.distributions.python.ops.bijectors.affine import *
from astronet.contrib.distributions.python.ops.bijectors.affine_linear_operator import *
from astronet.contrib.distributions.python.ops.bijectors.affine_scalar import *
from astronet.contrib.distributions.python.ops.bijectors.batch_normalization import *
from astronet.contrib.distributions.python.ops.bijectors.chain import *
from astronet.contrib.distributions.python.ops.bijectors.cholesky_outer_product import *
from astronet.contrib.distributions.python.ops.bijectors.conditional_bijector import *
from astronet.contrib.distributions.python.ops.bijectors.exp import *
from astronet.contrib.distributions.python.ops.bijectors.fill_triangular import *
from astronet.contrib.distributions.python.ops.bijectors.gumbel import *
from astronet.contrib.distributions.python.ops.bijectors.inline import *
from astronet.contrib.distributions.python.ops.bijectors.invert import *
from astronet.contrib.distributions.python.ops.bijectors.kumaraswamy import *
from astronet.contrib.distributions.python.ops.bijectors.masked_autoregressive import *
from astronet.contrib.distributions.python.ops.bijectors.matrix_inverse_tril import *
from astronet.contrib.distributions.python.ops.bijectors.ordered import *
from astronet.contrib.distributions.python.ops.bijectors.permute import *
from astronet.contrib.distributions.python.ops.bijectors.power_transform import *
from astronet.contrib.distributions.python.ops.bijectors.real_nvp import *
from astronet.contrib.distributions.python.ops.bijectors.reshape import *
from astronet.contrib.distributions.python.ops.bijectors.scale_tril import *
from astronet.contrib.distributions.python.ops.bijectors.sigmoid import *
from astronet.contrib.distributions.python.ops.bijectors.sinh_arcsinh import *
from astronet.contrib.distributions.python.ops.bijectors.softmax_centered import *
from astronet.contrib.distributions.python.ops.bijectors.softplus import *
from astronet.contrib.distributions.python.ops.bijectors.softsign import *
from astronet.contrib.distributions.python.ops.bijectors.square import *
from astronet.contrib.distributions.python.ops.bijectors.transform_diagonal import *
from tensorflow.python.ops.distributions.bijector import *
from tensorflow.python.ops.distributions.identity_bijector import Identity

# pylint: enable=unused-import,wildcard-import,line-too-long,g-importing-member

from tensorflow.python.util.all_util import remove_undocumented

remove_undocumented(__name__)
