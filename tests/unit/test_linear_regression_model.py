import unittest
from srcs import LinearRegressionModel, Vector

class TestLinearRegressionModel(unittest.TestCase):
	def test_default_parameters_are_zero(self) -> None:
		model = LinearRegressionModel()
		self.assertEqual(model.parameters.to_list(), [0.0, 0.0])

	def test_predict_uses_theta(self) -> None:
		model = LinearRegressionModel(Vector([1.5, 2.0]))
		prediction = model.predict(3.0)
		self.assertAlmostEqual(prediction, 7.5)

	def test_predict_many_matches_individual_calls(self) -> None:
		model = LinearRegressionModel(Vector([0.5, 1.0]))
		features = Vector([1.0, 2.0, 3.0])
		batch = features.map(model.predict)
		self.assertEqual(batch.to_list(), [1.5, 2.5, 3.5])

	def test_predict_many_with_list_iterable(self) -> None:
		model = LinearRegressionModel(Vector([0.5, 1.0]))
		batch = Vector([1.0, 2.0, 3.0]).map(model.predict)
		self.assertEqual(batch.to_list(), [1.5, 2.5, 3.5])

	def test_update_validates_length(self) -> None:
		model = LinearRegressionModel()
		with self.assertRaises(ValueError):
			model.update(Vector([1.0]))

	def test_parameters_returns_clone(self) -> None:
		model = LinearRegressionModel(Vector([1.0, 2.0]))
		a = model.parameters
		b = model.parameters
		self.assertIsNot(a, b)
		self.assertEqual(a.to_list(), b.to_list())

if __name__ == '__main__':
	unittest.main()
