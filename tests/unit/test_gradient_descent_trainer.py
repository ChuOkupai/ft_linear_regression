import unittest
from srcs import Dataset, GradientDescentTrainer, LinearRegressionModel

class TestGradientDescentTrainer(unittest.TestCase):
	def test_training_converges_to_expected_parameters(self) -> None:
		dataset = Dataset([0.0, 1.0, 2.0, 3.0], [3.0, 5.0, 7.0, 9.0], 'x', 'y')
		model = LinearRegressionModel()
		trainer = GradientDescentTrainer(model, 0.01)
		parameters = trainer.train(dataset, 1500)
		self.assertAlmostEqual(parameters[0], 3.0, delta=0.1)
		self.assertAlmostEqual(parameters[1], 2.0, delta=0.1)

	def test_invalid_learning_rate_raises(self) -> None:
		model = LinearRegressionModel()
		with self.assertRaises(ValueError):
			GradientDescentTrainer(model, 0.0)

	def test_invalid_epoch_count_raises(self) -> None:
		dataset = Dataset([1.0, 2.0], [2.0, 4.0], 'x', 'y')
		model = LinearRegressionModel()
		trainer = GradientDescentTrainer(model, 0.01)
		with self.assertRaises(ValueError):
			trainer.train(dataset, 0)

	def test_training_handles_constant_feature(self) -> None:
		dataset = Dataset([5.0, 5.0, 5.0, 5.0], [10.0, 10.0, 10.0, 10.0], 'x', 'y')
		model = LinearRegressionModel()
		trainer = GradientDescentTrainer(model, 0.1)
		parameters = trainer.train(dataset, 100)
		self.assertAlmostEqual(parameters[0], 10.0, delta=0.1)
		self.assertAlmostEqual(parameters[1], 0.0, delta=0.1)

if __name__ == '__main__':
	unittest.main()
