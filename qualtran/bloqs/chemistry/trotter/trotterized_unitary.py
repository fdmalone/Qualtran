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
"""Bloq for building a Trotterized unitary"""

from functools import cached_property
from typing import Dict, Iterable, Union

import attrs
import sympy

from qualtran import Bloq, bloq_example, BloqBuilder, BloqDocSpec, Signature, Soquet, SoquetT


@attrs.frozen
class TrotterizedUnitary(Bloq):
    r"""Implement arbitrary trotterized unitary given any Trotter splitting of the Hamiltonian.

    Given an arbitrary splitting of the Hamiltonian into $m$ terms

    $$
        H = \sum_j=1^m H_j
    $$

    then the unitary $e^{i t H}$ can be approximately implemented via Trotterization as

    $$
        U \approx \prod_k=1^l \prod_j e^{i t c_k H_j}
    $$

    where $c_j^k$ are some coefficients.

    Args:
        bloqs: A tuple of bloqs of length `m` which implement the unitaries for
            each term in the Hamiltonian.
        indices: A tuple of length `l` which specifies which bloq to apply when
            forming the unitary as a product of unitaries.
        coeffs: The coefficients `c_k` which appear in the expression for the unitary.
        timestep: The timestep `t`.

    Registers:
        system: The system register to which to apply the unitary.
    """

    bloqs: Iterable[Bloq]
    indices: Iterable[int]
    coeffs: Iterable[Union[float, sympy.Expr]]
    timestep: Union[float, sympy.Expr]

    def __attrs_post_init__(self):
        ref_sig = self.bloqs[0].signature
        for bloq in self.bloqs[1:]:
            assert bloq.signature == ref_sig

    @cached_property
    def signature(self) -> Signature:
        return self.bloqs[0].signature

    def build_composite_bloq(self, bb: 'BloqBuilder', **soqs: SoquetT) -> Dict[str, 'Soquet']:
        for i, c in zip(self.indices, self.coeffs):
            soqs |= bb.add_d(attrs.evolve(self.bloqs[i], angle=2 * c * self.timestep), **soqs)
        return soqs


@bloq_example
def _trott_unitary() -> TrotterizedUnitary:
    from qualtran.bloqs.for_testing.ising import IsingXUnitary, IsingZZUnitary

    nsites = 3
    j_zz = 2
    gamma_x = 0.1
    dt = 0.01
    indices = (0, 1, 0)
    coeffs = (0.5 * gamma_x, j_zz, 0.5 * gamma_x)
    zz_bloq = IsingZZUnitary(nsites=nsites, angle=2 * dt * j_zz)
    x_bloq = IsingXUnitary(nsites=nsites, angle=0.5 * 2 * dt * gamma_x)
    trott_unitary = TrotterizedUnitary(
        bloqs=(x_bloq, zz_bloq), indices=indices, coeffs=coeffs, timestep=dt
    )
    return trott_unitary


_TROTT_UNITARY_DOC = BloqDocSpec(
    bloq_cls=TrotterizedUnitary,
    import_line=(
        'from qualtran.bloqs.for_testing.ising import IsingXUnitary, IsingZZUnitary\n'
        'from qualtran.bloqs.chemistry.trotter.trotterized_unitary import TrotterizedUnitary'
    ),
    examples=(_trott_unitary,),
)
