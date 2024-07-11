#  Copyright 2023 Google LLC
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
"""Function for building a walk operator for the THC hamiltonian."""
import numpy as np
from numpy.typing import NDArray

from qualtran.bloqs.block_encoding.lcu_block_encoding import LCUBlockEncoding
from qualtran.bloqs.chemistry.thc import PrepareTHC, SelectTHC
from qualtran.bloqs.qubitization.qubitization_walk_operator import QubitizationWalkOperator


def get_walk_operator_for_thc_ham(
    tpq: NDArray,
    eta: NDArray,
    zeta: NDArray,
    num_bits_state_prep: int,
    num_bits_theta: int,
    kr1: int = 1,
    kr2: int = 1,
) -> QubitizationWalkOperator:
    """Build a QubitizationWalkOperator for the THC hamiltonian.

    Args:
        tpq: Modified one-body hamiltonian.
        eta: THC leaf tensors.
        zeta: THC central tensor.
        num_bits_state_prep: The number of bits for the state prepared during alias sampling.
        num_bits_theta: Number of bits of precision for the rotations. Called
            $\beth$ in the reference.
        kr1: block sizes for QROM erasure for outputting rotation angles. See Eq 34.
        kr2: block sizes for QROM erasure for outputting rotation angles. This
            is for the second QROM (eq 35)

    Returns:
        walk_op: Walk operator for THC hamiltonian.
    """
    t_l, _ = np.linalg.eigh(tpq)
    prep = PrepareTHC.from_hamiltonian_coeffs(t_l, zeta, num_bits_state_prep)
    overlap = eta.dot(eta.T)
    norm_fac = np.diag(np.diag(overlap))
    zeta_normalized = norm_fac.dot(zeta).dot(norm_fac)  # Eq. 11 & 12
    lambda_t = np.sum(np.abs(t_l))  # Eq. 19
    lambda_z = 0.5 * np.sum(np.abs(zeta_normalized))  # Eq. 20
    num_mu = zeta.shape[-1]
    num_spin_orb = 2 * tpq.shape[-1]
    sel = SelectTHC(num_mu, num_spin_orb, num_bits_theta, prep.keep_bitsize, kr1=kr1, kr2=kr2)
    block_encoding = LCUBlockEncoding(
        alpha=lambda_t + lambda_z, epsilon=1e-3, select=sel, prepare=prep
    )
    walk_op = QubitizationWalkOperator(block_encoding=block_encoding)
    return walk_op
