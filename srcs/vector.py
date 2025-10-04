
from collections.abc import Callable, Iterable

class Vector:
	def __init__(self, values: Iterable[float] | None = None) -> None:
		self._values = [float(v) for v in values] if values else []

	def _zip_values(self, other: 'Vector') -> Iterable[tuple[float, float]]:
		if len(self) != len(other):
			raise ValueError('Vectors must have the same length')
		return zip(self._values, other._values)

	def clone(self) -> 'Vector':
		return Vector(self._values)

	def dot(self, other: 'Vector') -> float:
		return sum(a * b for a, b in self._zip_values(other))

	def map(self, func: Callable[[float], float]) -> 'Vector':
		return Vector(func(v) for v in self._values)

	def square(self) -> 'Vector':
		return Vector(value * value for value in self._values)

	def sum(self) -> float:
		return sum(self._values)

	def to_list(self) -> list[float]:
		return list(self._values)

	def __add__(self, other: 'Vector') -> 'Vector':
		return Vector(a + b for a, b in self._zip_values(other))

	def __sub__(self, other: 'Vector') -> 'Vector':
		return Vector(a - b for a, b in self._zip_values(other))

	def __mul__(self, scalar: float) -> 'Vector':
		return Vector(a * scalar for a in self._values)

	def __rmul__(self, scalar: float) -> 'Vector':
		return self.__mul__(scalar)

	def __truediv__(self, scalar: float) -> 'Vector':
		return Vector(a / scalar for a in self._values)

	def __getitem__(self, index: int) -> float:
		return self._values[index]

	def __iter__(self):
		return iter(self._values)

	def __len__(self) -> int:
		return len(self._values)

	def __repr__(self) -> str:
		return f"Vector({self._values!r})"
