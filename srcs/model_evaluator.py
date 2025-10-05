import math
from . import Dataset, Vector

class ModelEvaluator:
	def _predictions(self, dataset: Dataset, parameters: Vector) -> Vector:
		if len(parameters) != 2:
			raise ValueError('Parameters vector must have exactly two elements (intercept and slope)')
		intercept, slope = parameters[0], parameters[1]
		return dataset.features.map(lambda x: intercept + slope * x)

	def evaluate(self, dataset: Dataset, parameters: Vector) -> dict[str, float]:
		n = dataset.size
		if n == 0:
			raise ValueError('Dataset must contain at least one sample')

		residuals = dataset.targets - self._predictions(dataset, parameters)
		sse = residuals.square().sum()
		mse = sse / n
		rmse = math.sqrt(mse)
		mae = residuals.map(abs).sum() / n

		# R2: 1 - SSE / TSS, guarding for constant target values
		y = dataset.targets
		mean_y = y.sum() / n
		tss = (y - Vector([mean_y] * n)).square().sum()
		if tss == 0:
			r2 = 1.0 if sse == 0 else 0.0
		else:
			r2 = 1.0 - (sse / tss)

		return {
			'MSE': mse,
			'RMSE': rmse,
			'MAE': mae,
			'R2': r2,
		}
