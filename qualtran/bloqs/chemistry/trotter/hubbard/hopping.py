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
from typing import Set

from attrs import frozen

from qualtran import (
    Bloq,
    bloq_example,
    BloqDocSpec,
    QAny,
    Register,
    Signature,
)
if TYPE_CHECKING:
    from qualtran.resource_counting import BloqCountT, SympySymbolAllocator


@frozen
class HoppingPlaquette(Bloq):

@frozen
class HoppingTile(Bloq):
    r"""Bloq implementing a "tile" of the one-body hopping unitary.

    Args:
        length: Lattice length L.

    Registers:
        system: The system register of size 2 `length`.

    References:
        [Early fault-tolerant simulations of the Hubbard model](
            https://iopscience.iop.org/article/10.1088/2058-9565/ac3110/meta)
    """

    length: int

    @cached_property
    def signature(self) -> Signature:
        return Signature([Register('system', QAny(self.length), shape=(2,))])

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> Set['BloqCountT']:


@bloq_example
def _kinetic_energy() -> KineticEnergy:
    nelec = 12
    ngrid_x = 2 * 8 + 1
    kinetic_energy = KineticEnergy(nelec, ngrid_x)
    return kinetic_energy


_KINETIC_ENERGY = BloqDocSpec(
    bloq_cls=KineticEnergy,
    import_line='from qualtran.bloqs.chemistry.trotter.grid_ham.kinetic import KineticEnergy',
    examples=(_kinetic_energy,),
)
