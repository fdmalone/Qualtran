from typing import Union, Sequence, Iterable
import cirq


class LessThanGate(cirq.ArithmeticGate):
    """Applies U_a|x>|z> = |x> |z ^ (x < a)>"""

    def __init__(self, input_register: Sequence[int], val: int) -> None:
        self._input_register = input_register
        self._val = val
        self._target_register = [2]

    def registers(self) -> Sequence[Union[int, Sequence[int]]]:
        return self._input_register, self._val, self._target_register

    def with_registers(
        self, *new_registers: Union[int, Sequence[int]]
    ) -> "LessThanGate":
        return LessThanGate(new_registers[0], new_registers[1])

    def apply(
        self, input_val, max_val, target_register_val
    ) -> Union[int, Iterable[int]]:
        return input_val, max_val, target_register_val ^ (input_val < max_val)

    def __repr__(self) -> str:
        return f"cirq_qubitization.LessThanGate({self._input_register, self._val})"


class LessThanEqualGate(cirq.ArithmeticGate):
    """Applies U|x>|y>|z> = |x>|y> |z ^ (x <= y)>"""

    def __init__(
        self, first_input_register: Sequence[int], second_input_register: Sequence[int]
    ) -> None:
        self._first_input_register = first_input_register  # |x>
        self._second_input_register = second_input_register  # |y>
        self._target_register = [2]  # |z>

    def registers(self) -> Sequence[Union[int, Sequence[int]]]:
        return (
            self._first_input_register,
            self._second_input_register,
            self._target_register,
        )

    def with_registers(
        self, *new_registers: Union[int, Sequence[int]]
    ) -> "LessThanEqualGate":
        return LessThanEqualGate(new_registers[0], new_registers[1])

    def apply(
        self, first_input_val, second_input_val, target_register_val
    ) -> Union[int, int, Iterable[int]]:
        return (
            first_input_val,
            second_input_val,
            target_register_val ^ (first_input_val <= second_input_val),
        )

    def __repr__(self) -> str:
        return f"cirq_qubitization.LessThanEqualGate({self._first_input_register, self._second_input_register})"
