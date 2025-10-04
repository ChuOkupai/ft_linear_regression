from . import Vector

class LinearRegressionModel:
	def __init__(self, thetas: Vector | None = None) -> None:
		self._theta = self._build_thetas(thetas)

	def _build_thetas(self, values: Vector | None) -> Vector:
		if values is None:
			return Vector([0.0, 0.0])
		if len(values) != 2:
			raise ValueError('Theta must contain exactly two values')
		return values.clone()

	@property
	def parameters(self) -> Vector:
		return self._theta.clone()

	def predict(self, feature: float) -> float:
		return self._theta[0] + self._theta[1] * feature

	def update(self, values: Vector) -> None:
		self._theta = self._build_thetas(values)
