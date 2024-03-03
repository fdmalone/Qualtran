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
"""Bloq for building Trotterized unitary"""

from functools import cached_property
from typing import Iterable, Set, TYPE_CHECKING, Union

import attrs
import sympy

from qualtran import Bloq, bloq_example, BloqDocSpec, QAny, QBit, Register, Signature

if TYPE_CHECKING:
    from qualtran.resource_counting import BloqCountT, SympySymbolAllocator


@attrs.frozen
def TrotterizedUnitary(Bloq):
    """Implement arbitrary trotterized unitary given any Trotter splitting of the Hamiltonian."""

    bloqs: Iterable[Bloq]
    indices: Iterable[int]
    coeffs: Iterable[Union[float, sympy.Expr]]
    timestep: Union[float, sympy.Expr]

    def __attrs_post_init__(self):
        ref_sig = self.bloqs[0]
        for bloq in self.bloq[1:]:
            assert bloq.signature == ref_sig

    @cached_property
    def signature(self) -> Signature:
        return self.bloqs[0].signature

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> Set['BloqCountT']:
        return {
            (attrs.evolve(self.bloqs[i], angle=self.timestep * c), 1) for i, c in zip(self.indices, self.coeffs)
        }
