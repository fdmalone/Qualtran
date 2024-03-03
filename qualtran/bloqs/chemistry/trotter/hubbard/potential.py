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
"""Bloqs implementing unitary evolution under the one-body hopping Hamiltonian."""

from functools import cached_property
from typing import Set, TYPE_CHECKING, Union

import sympy
from attrs import frozen

from qualtran import Bloq, bloq_example, BloqDocSpec, QAny, Register, Signature
from qualtran.bloqs.basic_gates import Rz

if TYPE_CHECKING:
    from qualtran.resource_counting import BloqCountT, SympySymbolAllocator


@frozen
class Potential(Bloq):
    r"""Bloq implementing the hubbard U part of the hamiltonian.

    Args:
        length: Lattice length L.

    Registers:
        system: The system register of size 2 `length`.

    References:
        [Early fault-tolerant simulations of the Hubbard model](
            https://iopscience.iop.org/article/10.1088/2058-9565/ac3110/meta)
    """

    length: Union[int, sympy.Expr]
    angle: Union[float, sympy.Expr]
    eps: Union[float, sympy.Expr] = 1e-9

    @cached_property
    def signature(self) -> Signature:
        return Signature([Register('system', QAny(self.length), shape=(2,))])

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> Set['BloqCountT']:
        return (Rz(angle=self.angle, eps=self.eps), self.length**2 // 2)


@bloq_example
def _potential() -> Potential:
    length = 8
    angle = 0.5
    kinetic_energy = Potential(length, angle)
    return kinetic_energy


_POTENTIAL_DOC = BloqDocSpec(
    bloq_cls=Potential,
    import_line='from qualtran.bloqs.chemistry.trotter.hubbard.potential import Potential',
    examples=(_potential,),
)
