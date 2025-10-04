import unittest
from srcs import Dataset

class TestLinearRegressionDataset(unittest.TestCase):
	def test_size_matches_input(self) -> None:
		dataset = Dataset([1.0, 2.0, 3.0], [2.0, 4.0, 6.0], 'feature', 'target')
		self.assertEqual(dataset.size, 3)

	def test_raises_on_mismatched_sizes(self) -> None:
		with self.assertRaises(ValueError):
			Dataset([1.0, 2.0], [1.0], 'feature', 'target')

	def test_raises_on_empty_input(self) -> None:
		with self.assertRaises(ValueError):
			Dataset([], [], 'feature', 'target')

	def test_from_csv_reads_file(self) -> None:
		from tempfile import TemporaryDirectory
		from pathlib import Path
		with TemporaryDirectory() as td:
			p = Path(td) / 'data.csv'
			p.write_text('feature,target\n1.0,2.0\n3.0,6.0\n', encoding='utf-8')
			dataset = Dataset.from_csv(p)
			self.assertEqual(dataset.feature_name, 'feature')
			self.assertEqual(dataset.target_name, 'target')
			self.assertEqual(dataset.size, 2)
			self.assertEqual(dataset.features.to_list(), [1.0, 3.0])
			self.assertEqual(dataset.targets.to_list(), [2.0, 6.0])

	def test_from_csv_raises_on_insufficient_columns(self) -> None:
		from tempfile import TemporaryDirectory
		from pathlib import Path
		with TemporaryDirectory() as td:
			p = Path(td) / 'data.csv'
			p.write_text('only\n1\n', encoding='utf-8')
			with self.assertRaises(ValueError):
				Dataset.from_csv(p)

	def test_from_csv_requires_both_feature_and_target(self) -> None:
		from tempfile import TemporaryDirectory
		from pathlib import Path
		with TemporaryDirectory() as td:
			p = Path(td) / 'data.csv'
			p.write_text('feature,target\n1.0,2.0\n', encoding='utf-8')
			with self.assertRaises(ValueError):
				Dataset.from_csv(p, feature='feature')

	def test_from_csv_raises_on_missing_column_names_when_explicit(self) -> None:
		from tempfile import TemporaryDirectory
		from pathlib import Path
		with TemporaryDirectory() as td:
			p = Path(td) / 'data.csv'
			p.write_text('f1,f2\n1.0,2.0\n', encoding='utf-8')
			with self.assertRaises(ValueError):
				Dataset.from_csv(p, feature='missing', target='f2')

	def test_from_csv_raises_on_same_feature_and_target_names(self) -> None:
		from tempfile import TemporaryDirectory
		from pathlib import Path
		with TemporaryDirectory() as td:
			p = Path(td) / 'data.csv'
			p.write_text('x,y\n1.0,2.0\n', encoding='utf-8')
			with self.assertRaises(ValueError):
				Dataset.from_csv(p, feature='x', target='x')

	def test_from_csv_accepts_explicit_column_names(self) -> None:
		from tempfile import TemporaryDirectory
		from pathlib import Path
		with TemporaryDirectory() as td:
			p = Path(td) / 'data.csv'
			p.write_text('a,b,c\n1.0,2.0,3.0\n4.0,5.0,6.0\n', encoding='utf-8')
			dataset = Dataset.from_csv(p, feature='b', target='c')
			self.assertEqual(dataset.feature_name, 'b')
			self.assertEqual(dataset.target_name, 'c')
			self.assertEqual(dataset.features.to_list(), [2.0, 5.0])
			self.assertEqual(dataset.targets.to_list(), [3.0, 6.0])

if __name__ == '__main__':
	unittest.main()
