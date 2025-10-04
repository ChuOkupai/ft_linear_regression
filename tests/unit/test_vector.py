import unittest
from srcs import Vector

class TestVector(unittest.TestCase):
	def test_init_empty(self):
		v = Vector()
		self.assertEqual(len(v), 0)
		self.assertEqual(v.to_list(), [])

	def test_init_with_values(self):
		v = Vector([1, 2.5, 3])
		self.assertEqual(v.to_list(), [1.0, 2.5, 3.0])

	def test__zip_values_same_length(self):
		v1 = Vector([1, 2])
		v2 = Vector([3, 4])
		pairs = list(v1._zip_values(v2))
		self.assertEqual(pairs, [(1.0, 3.0), (2.0, 4.0)])

	def test__zip_values_length_mismatch_raises(self):
		v1 = Vector([1])
		v2 = Vector([1, 2])
		with self.assertRaises(ValueError):
			list(v1._zip_values(v2))

	def test_clone_independence(self):
		v = Vector([1, 2, 3])
		c = v.clone()
		self.assertIsNot(v, c)
		self.assertEqual(v.to_list(), c.to_list())
		lst = c.to_list()
		lst.append(4)
		self.assertEqual(v.to_list(), [1.0, 2.0, 3.0])

	def test_dot_product(self):
		v1 = Vector([1, 2, 3])
		v2 = Vector([4, 5, 6])
		self.assertEqual(v1.dot(v2), 32.0)

	def test_map_function(self):
		v = Vector([1, 2, 3])
		w = v.map(lambda x: x + 0.5)
		self.assertEqual(w.to_list(), [1.5, 2.5, 3.5])

	def test_square(self):
		v = Vector([1, -2, 3])
		self.assertEqual(v.square().to_list(), [1.0, 4.0, 9.0])

	def test_sum(self):
		v = Vector([1, 2, 3.5])
		self.assertEqual(v.sum(), 6.5)

	def test_to_list_copy(self):
		v = Vector([1, 2])
		lst = v.to_list()
		lst.append(3)
		self.assertEqual(v.to_list(), [1.0, 2.0])

	def test_add(self):
		v1 = Vector([1, 2])
		v2 = Vector([3, 4])
		self.assertEqual((v1 + v2).to_list(), [4.0, 6.0])

	def test_add_length_mismatch_raises(self):
		with self.assertRaises(ValueError):
			_ = Vector([1]) + Vector([1, 2])

	def test_sub(self):
		v1 = Vector([5, 7])
		v2 = Vector([3, 4])
		self.assertEqual((v1 - v2).to_list(), [2.0, 3.0])

	def test_sub_length_mismatch_raises(self):
		with self.assertRaises(ValueError):
			_ = Vector([1, 2]) - Vector([1])

	def test_mul_scalar_left(self):
		v = Vector([1, -2, 0.5])
		self.assertEqual((v * 2).to_list(), [2.0, -4.0, 1.0])
		self.assertEqual((v * 0).to_list(), [0.0, 0.0, 0.0])

	def test_rmul_scalar_right(self):
		v = Vector([1, -2, 0.5])
		self.assertEqual((3 * v).to_list(), [3.0, -6.0, 1.5])

	def test_truediv_scalar(self):
		v = Vector([2, -4, 1])
		self.assertEqual((v / 2).to_list(), [1.0, -2.0, 0.5])

	def test_truediv_by_zero_raises(self):
		v = Vector([1, 2])
		with self.assertRaises(ZeroDivisionError):
			_ = v / 0

	def test_getitem_indexing_and_negative(self):
		v = Vector([10, 20, 30])
		self.assertEqual(v[0], 10.0)
		self.assertEqual(v[2], 30.0)
		self.assertEqual(v[-1], 30.0)
		with self.assertRaises(IndexError):
			_ = v[3]

	def test_iter(self):
		v = Vector([1, 2, 3])
		self.assertEqual(list(iter(v)), [1.0, 2.0, 3.0])
		self.assertEqual(list(v), [1.0, 2.0, 3.0])

	def test_len(self):
		self.assertEqual(len(Vector()), 0)
		self.assertEqual(len(Vector([1, 2, 3])), 3)

	def test_repr(self):
		v = Vector([1, 2])
		self.assertEqual(repr(v), "Vector([1.0, 2.0])")

if __name__ == '__main__':
	unittest.main()
