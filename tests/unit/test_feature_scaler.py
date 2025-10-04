import unittest
from srcs import FeatureScaler, Vector

class TestFeatureScaler(unittest.TestCase):
	def test_from_vector_computes_statistics(self) -> None:
		source = Vector([1.0, 2.0, 3.0])
		scaler = FeatureScaler.from_vector(source)
		self.assertAlmostEqual(scaler.mean, 2.0, places=6)
		self.assertAlmostEqual(scaler.std, 0.8164965809, places=6)

	def test_scale_and_unscale(self) -> None:
		source = Vector([0.0, 2.0, 4.0])
		scaler = FeatureScaler.from_vector(source)
		scaled = scaler.scale(6.0)
		expected = (6.0 - scaler.mean) / scaler.std
		self.assertAlmostEqual(scaled, expected, places=6)
		self.assertAlmostEqual(scaler.unscale(scaled), 6.0, places=6)

	def test_scale_vector_handles_constant_values(self) -> None:
		source = Vector([5.0, 5.0, 5.0])
		scaler = FeatureScaler.from_vector(source)
		scaled_vector = scaler.scale_vector(source)
		self.assertEqual(scaler.std, 1.0)
		self.assertEqual(scaled_vector.to_list(), [0.0, 0.0, 0.0])

	def test_from_vector_raises_on_empty(self) -> None:
		with self.assertRaises(ValueError):
			FeatureScaler.from_vector(Vector([]))

	def test_unscale_vector_roundtrip(self) -> None:
		source = Vector([1.0, 2.0, 3.0])
		scaler = FeatureScaler.from_vector(source)
		scaled = scaler.scale_vector(source)
		unscaled = scaler.unscale_vector(scaled)
		self.assertEqual(unscaled.to_list(), source.to_list())

if __name__ == '__main__':
	unittest.main()
