from . import Dataset, FeatureScaler, LinearRegressionModel, Vector
from tqdm import tqdm

class GradientDescentTrainer:
	def __init__(self, model: LinearRegressionModel, learning_rate: float) -> None:
		if learning_rate <= 0:
			raise ValueError('Learning rate must be positive')
		self._model = model
		self._learning_rate = float(learning_rate)

	def train(self, dataset: Dataset, epochs: int) -> Vector:
		if epochs <= 0:
			raise ValueError('Epochs must be positive')
		scaler = FeatureScaler.from_vector(dataset.features)
		scaled_features = scaler.scale_vector(dataset.features)
		for _ in tqdm(range(epochs), unit='epoch'):
			errors = scaled_features.map(self._model.predict) - dataset.targets
			gradients = Vector([errors.sum(), errors.dot(scaled_features)]) / len(scaled_features)
			tmp_theta = self._model.parameters - gradients * self._learning_rate
			self._model.update(tmp_theta)
		thetas, std, mean = self._model.parameters, scaler.std, scaler.mean
		return Vector([thetas[0] - (thetas[1] / std) * mean, thetas[1] / std])
