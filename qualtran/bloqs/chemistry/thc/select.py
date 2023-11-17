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
"""SELECT for the molecular tensor hypercontraction (THC) hamiltonian"""
from functools import cached_property
from typing import Dict, Set, Tuple, TYPE_CHECKING

import cirq
import numpy as np
from attrs import frozen

from qualtran import Bloq, BloqBuilder, Register, SelectionRegister, Signature, SoquetT
from qualtran.bloqs.basic_gates import CSwap, Toffoli, XGate
from qualtran.bloqs.chemistry.df.select import ApplyControlledZs
from qualtran.bloqs.select_and_prepare import SelectOracle

if TYPE_CHECKING:
    from qualtran.resource_counting import BloqCountT, SympySymbolAllocator


@frozen
class Rotation(Bloq):

    bitsize: int

    def signature(self) -> Signature:
        return Signature.build(theta=self.bitsize)

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> Set['BloqCountT']:
        rot_cost = self.bitsize - 2
        return {(Toffoli(), rot_cost)}


@frozen
class QROM(Bloq):

    datasizes: Tuple[int, ...]
    bitsize: int
    adjoint: bool = False
    kr: int = 1

    def signature(self) -> Signature:
        return Signature.build(data=self.bitsize)

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> Set['BloqCountT']:
        # from listings on page 17 of Ref. [1]
        if self.adjoint:
            toff_cost_qrom = (
                sum(int(np.ceil(d / ((i+1) * self.kr))) for i, d in enumerate(self.datasizes)) + self.kr
            )
        else:
            toff_cost_qrom = sum(self.datasizes) - 2
        return {(Toffoli(), toff_cost_qrom)}


@frozen
class THCRotations(Bloq):
    """Bloq for rotating into THC basis through Givens rotation network.

    This is accounting for In-data:rot and In-R in Fig. 7 of the THC paper (Ref.
    1). In practice this bloq is made up of a QROM load of the angles followed
    by controlled rotations. Equivalently it can be built from a modified
    version of the ProgrammableRotationGateArray from implemented in qualtran
    from Ref. 2. This is a placeholder waiting for an actual implementation.
    See https://github.com/quantumlib/Qualtran/issues/386.

    Args:
        num_mu: THC auxiliary index dimension $M$
        num_spin_orb: number of spin orbitals $N$
        num_bits_theta: Number of bits of precision for the rotations. Called
            $\beth$ in the reference.
        kr1: block sizes for QROM erasure for outputting rotation angles. See Eq 34.
        kr2: block sizes for QROM erasure for outputting rotation angles. This
            is for the second QROM (eq 35)
        two_body_only: Whether to only apply the two body Hamiltonian. This reduces the QROM size.
        adjoint: Whether to dagger this bloq or not.

    References:
        [Even more efficient quantum computations of chemistry through
            tensor hypercontraction](https://arxiv.org/pdf/2011.03494.pdf) Fig. 7.
        [Quantum computing enhanced computational catalysis](https://arxiv.org/abs/2007.14460).
            Burg, Low et. al. 2021. Eq. 73
    """

    num_mu: int
    num_spin_orb: int
    num_bits_theta: int
    two_body_only: bool = False
    adjoint: bool = False
    kr1: int = 1
    kr2: int = 1

    @cached_property
    def signature(self) -> Signature:
        return Signature(
            [
                Register("nu_eq_mp1", bitsize=1),
                Register("data", bitsize=self.num_bits_theta),
                Register("sel", bitsize=self.num_mu.bit_length()),
                Register("trg", bitsize=self.num_spin_orb // 2),
            ]
        )

    def pretty_name(self) -> str:
        dag = '†' if self.adjoint else ''
        return f"In_mu-R{dag}"

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> Set['BloqCountT']:
        # from listings on page 17 of Ref. [1]
        datasizes = [self.num_mu, self.num_spin_orb // 2]
        if self.two_body_only:
            datasizes[1] = 0
        return {
            (
                QROM(tuple(datasizes), self.num_bits_theta, adjoint=self.adjoint, kr=self.kr1 if self.two_body_only else self.kr2),
                1
            ),
            (Rotation(self.num_bits_theta), self.num_spin_orb),
        }


@frozen
class SelectTHC(SelectOracle):
    r"""SELECT for THC Hamiltonian.

    Args:
        num_mu: THC auxiliary index dimension $M$
        num_spin_orb: number of spin orbitals $N$
        num_bits_theta: Number of bits of precision for the rotations. Called
            $\beth$ in the reference.
        kr1: block sizes for QROM erasure for outputting rotation angles. See Eq 34.
        kr2: block sizes for QROM erasure for outputting rotation angles. This
            is for the second QROM (eq 35)

    Registers:
        succ: success flag qubit from uniform state preparation
        nu_eq_mp1: flag for if $nu = M+1$
        mu: $\mu$ register.
        nu: $\nu$ register.
        theta: sign register.
        plus_mn: Flag controlling swaps between mu and nu. Note that as per the
            Reference, the swaps are NOT performed as part of SELECT as they're
            acounted for during Prepare.
        plus_a / plus_b: plus state for controlled swaps on spins.
        sys_a / sys_b : System registers for (a)lpha/(b)eta orbitals.

    References:
        [Even more efficient quantum computations of chemistry through
            tensor hypercontraction](https://arxiv.org/pdf/2011.03494.pdf) Fig. 7.
    """

    num_mu: int
    num_spin_orb: int
    num_bits_theta: int
    kr1: int = 1
    kr2: int = 1

    @cached_property
    def control_registers(self) -> Tuple[Register, ...]:
        return ()

    @cached_property
    def selection_registers(self) -> Tuple[SelectionRegister, ...]:
        return (
            SelectionRegister("succ", bitsize=1),
            SelectionRegister("nu_eq_mp1", bitsize=1),
            SelectionRegister(
                "mu", bitsize=(self.num_mu).bit_length(), iteration_length=self.num_mu + 1
            ),
            SelectionRegister(
                "nu", bitsize=(self.num_mu).bit_length(), iteration_length=self.num_mu + 1
            ),
            SelectionRegister("plus_mn", bitsize=1),
            SelectionRegister("plus_a", bitsize=1),
            SelectionRegister("plus_b", bitsize=1),
        )

    @cached_property
    def target_registers(self) -> Tuple[Register, ...]:
        return (
            Register("sys_a", bitsize=self.num_spin_orb // 2),
            Register("sys_b", bitsize=self.num_spin_orb // 2),
        )

    def build_composite_bloq(
        self,
        bb: 'BloqBuilder',
        succ: SoquetT,
        nu_eq_mp1: SoquetT,
        mu: SoquetT,
        nu: SoquetT,
        plus_mn: SoquetT,
        plus_a: SoquetT,
        plus_b: SoquetT,
        sys_a: SoquetT,
        sys_b: SoquetT,
    ) -> Dict[str, 'SoquetT']:

        plus_a, sys_a, sys_b = bb.add(CSwap(self.num_spin_orb // 2), ctrl=plus_a, x=sys_a, y=sys_b)

        # Rotations
        data = bb.allocate(self.num_bits_theta)
        nu_eq_mp1, data, mu, sys_a = bb.add(
            THCRotations(
                num_mu=self.num_mu,
                num_spin_orb=self.num_spin_orb,
                num_bits_theta=self.num_bits_theta,
                kr1=self.kr1,
                kr2=self.kr2,
            ),
            nu_eq_mp1=nu_eq_mp1,
            data=data,
            sel=mu,
            trg=sys_a,
        )
        # Controlled Z_0
        (succ,), (sys_a, sys_b) = bb.add(
            ApplyControlledZs(1, self.num_spin_orb // 2), ctrls=(succ,), system=(sys_a, sys_b)
        )
        # Undo rotations
        nu_eq_mp1, data, mu, sys_a = bb.add(
            THCRotations(
                num_mu=self.num_mu,
                num_spin_orb=self.num_spin_orb,
                num_bits_theta=self.num_bits_theta,
                kr1=self.kr1,
                kr2=self.kr2,
                adjoint=True,
            ),
            nu_eq_mp1=nu_eq_mp1,
            data=data,
            sel=mu,
            trg=sys_a,
        )

        # Clean up
        plus_a, sys_a, sys_b = bb.add(CSwap(self.num_spin_orb // 2), ctrl=plus_a, x=sys_a, y=sys_b)

        plus_mn = bb.add(XGate(), q=plus_mn)
        # should be a negative control
        nu_eq_mp1, mu, nu = bb.add(CSwap(self.num_mu.bit_length()), ctrl=nu_eq_mp1, x=mu, y=nu)

        # Swap spins
        # Should be a negative control..
        nu_eq_mp1, plus_a, plus_b = bb.add(CSwap(1), ctrl=nu_eq_mp1, x=plus_a, y=plus_b)

        # System register spin swaps
        plus_b, sys_a, sys_b = bb.add(CSwap(self.num_spin_orb // 2), ctrl=plus_b, x=sys_a, y=sys_b)

        # Rotations
        nu_eq_mp1, data, mu, sys_a = bb.add(
            THCRotations(
                num_mu=self.num_mu,
                num_spin_orb=self.num_spin_orb,
                num_bits_theta=self.num_bits_theta,
                kr1=self.kr1,
                kr2=self.kr2,
                two_body_only=True,
            ),
            nu_eq_mp1=nu_eq_mp1,
            data=data,
            sel=mu,
            trg=sys_a,
        )
        # Controlled Z_0
        (succ, nu_eq_mp1), (sys_a, sys_b) = bb.add(
            ApplyControlledZs(2, self.num_spin_orb // 2),
            ctrls=(succ, nu_eq_mp1),
            system=(sys_a, sys_b),
        )
        # Undo rotations
        nu_eq_mp1, data, mu, sys_a = bb.add(
            THCRotations(
                num_mu=self.num_mu,
                num_spin_orb=self.num_spin_orb,
                num_bits_theta=self.num_bits_theta,
                kr1=self.kr1,
                kr2=self.kr2,
                two_body_only=True,
                adjoint=True,
            ),
            nu_eq_mp1=nu_eq_mp1,
            data=data,
            sel=mu,
            trg=sys_a,
        )

        # Clean up
        plus_b, sys_a, sys_b = bb.add(CSwap(self.num_spin_orb // 2), ctrl=plus_b, x=sys_a, y=sys_b)

        # Undo the mu-nu swaps
        bb.free(data)
        return {
            'succ': succ,
            'nu_eq_mp1': nu_eq_mp1,
            'mu': mu,
            'nu': nu,
            'plus_mn': plus_mn,
            'plus_a': plus_a,
            'plus_b': plus_b,
            'sys_a': sys_a,
            'sys_b': sys_b,
        }
