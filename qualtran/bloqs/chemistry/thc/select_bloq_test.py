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

from typing import Sequence

import cirq
import numpy as np
import pytest
import scipy.linalg
from openfermion.ops import FermionOperator
from openfermion.transforms import jordan_wigner, qubit_operator_to_pauli_sum

from qualtran.bloqs.chemistry.thc.select_bloq import _givens_network, _thc_sel, find_givens_angles


def test_thc_rotations(bloq_autotester):
    bloq_autotester(_givens_network)


def test_thc_select(bloq_autotester):
    bloq_autotester(_thc_sel)


@pytest.mark.parametrize("theta", 2 * np.pi * np.random.random(10))
def test_interleaved_cliffords(theta):
    a, b = cirq.LineQubit.range(2)
    XY = cirq.Circuit([cirq.X(a), cirq.Y(b)])
    UXY = cirq.unitary(XY)
    RXY_ref = scipy.linalg.expm(-1j * theta * UXY / 2)
    C0 = [cirq.H(a), cirq.S(b) ** -1, cirq.H(b), cirq.CNOT(a, b)]
    C1 = [cirq.S(a) ** -1, cirq.H(a), cirq.H(b), cirq.CNOT(a, b)]
    RXY = cirq.unitary(cirq.Circuit([C0, cirq.Rz(rads=theta)(b), cirq.inverse(C0)]))
    assert np.allclose(RXY, RXY_ref)
    YX = cirq.Circuit([cirq.Y(a), cirq.X(b)])
    UYX = cirq.unitary(YX)
    RYX_ref = scipy.linalg.expm(1j * theta * UYX / 2)
    RYX = cirq.unitary(cirq.Circuit([C1, cirq.Rz(rads=-theta)(b), cirq.inverse(C1)]))
    assert np.allclose(RYX, RYX_ref)


def test_givens_unitary():
    num_orb = 2
    mat = np.random.random((num_orb, num_orb))
    mat = 0.5 * (mat + mat.T)
    unitary, _ = np.linalg.qr(mat)
    assert np.allclose(unitary.T @ unitary, np.eye(num_orb))
    thetas = find_givens_angles(unitary)
    qubits = cirq.LineQubit.range(num_orb)

    def majoranas_as_mats(p, qubits=None):
        a_p = FermionOperator(f'{p}')
        a_p_dag = FermionOperator(f'{p}^')
        maj_0 = a_p + a_p_dag
        maj_1 = -1j * (a_p - a_p_dag)
        return (
            qubit_operator_to_pauli_sum(jordan_wigner(maj_0), qubits=qubits).matrix(),
            qubit_operator_to_pauli_sum(jordan_wigner(maj_1), qubits=qubits).matrix(),
        )

    gamma_0_0, gamma_1_0 = majoranas_as_mats(0, qubits=qubits)
    for p in range(num_orb - 1):
        gamma_0_p, gamma_1_p = majoranas_as_mats(p, qubits=qubits)
        gamma_0_pp1, gamma_1_pp1 = majoranas_as_mats(p + 1, qubits=qubits)
        exp_g = scipy.linalg.expm(thetas[p] * (gamma_0_p * gamma_0_pp1))
        lhs = exp_g.conj().T @ gamma_0 @ exp_g
        rhs = gamma_0_p
