import math
from . import Vector

class FeatureScaler:
	def __init__(self, mean_value: float, std_value: float) -> None:
		self._mean = float(mean_value)
		std = float(std_value)
		self._std = std if std > 0 else 1.0

	@classmethod
	def from_vector(cls, values: Vector) -> 'FeatureScaler':
		length = len(values)
		if length == 0:
			raise ValueError('Values must not be empty')
		mean_value = values.sum() / length
		centered = values.map(lambda value: value - mean_value)
		variance = centered.dot(centered) / length
		std_value = math.sqrt(variance)
		return cls(mean_value, std_value)

	@property
	def mean(self) -> float:
		return self._mean

	@property
	def std(self) -> float:
		return self._std

	def scale(self, value: float) -> float:
		return (float(value) - self._mean) / self._std

	def unscale(self, value: float) -> float:
		return float(value) * self._std + self._mean

	def scale_vector(self, values: Vector) -> Vector:
		return values.map(self.scale)

	def unscale_vector(self, values: Vector) -> Vector:
		return values.map(self.unscale)
