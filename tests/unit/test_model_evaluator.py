import unittest
from typing import cast
from srcs import Dataset, Vector, ModelEvaluator

class TestModelEvaluator(unittest.TestCase):
	def test_perfect_fit_metrics(self) -> None:
		dataset = Dataset([0.0, 1.0, 2.0, 3.0], [1.0, 3.0, 5.0, 7.0], 'x', 'y')
		params = Vector([1.0, 2.0])
		metrics = ModelEvaluator().evaluate(dataset, params)
		self.assertAlmostEqual(metrics['MSE'], 0.0, places=10)
		self.assertAlmostEqual(metrics['RMSE'], 0.0, places=10)
		self.assertAlmostEqual(metrics['MAE'], 0.0, places=10)
		self.assertAlmostEqual(metrics['R2'], 1.0, places=10)

	def test_constant_targets_r2_definition(self) -> None:
		dataset = Dataset([1.0, 2.0, 3.0], [2.0, 2.0, 2.0], 'x', 'y')
		params = Vector([2.0, 0.0])
		metrics = ModelEvaluator().evaluate(dataset, params)
		self.assertAlmostEqual(metrics['R2'], 1.0, places=10)
		metrics_off = ModelEvaluator().evaluate(dataset, Vector([1.9, 0.0]))
		self.assertAlmostEqual(metrics_off['R2'], 0.0, places=10)

	def test_evaluate_raises_on_wrong_parameter_length(self) -> None:
		dataset = Dataset([0.0, 1.0], [0.0, 1.0], 'x', 'y')
		with self.assertRaises(ValueError):
			ModelEvaluator().evaluate(dataset, Vector([1.0]))

	def test_evaluate_raises_on_empty_dataset_like(self) -> None:
		class EmptyDataset:
			size = 0
			targets = None

		with self.assertRaises(ValueError):
			ModelEvaluator().evaluate(cast(Dataset, EmptyDataset()), Vector([0.0, 1.0]))

if __name__ == '__main__':
	unittest.main()
