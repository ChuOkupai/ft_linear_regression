import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from srcs import ModelConfiguration, Vector

class TestModelConfiguration(unittest.TestCase):
	def test_from_file_reads_values(self):
		with TemporaryDirectory() as td:
			p = Path(td) / 'model.json'
			data = {
				'feature_name': 'size',
				'target_name': 'price',
				'thetas': [0, 1.5, 2]
			}
			p.write_text(json.dumps(data), encoding='utf-8')
			cfg = ModelConfiguration.from_file(p)
			self.assertEqual(cfg.feature_name, 'size')
			self.assertEqual(cfg.target_name, 'price')
			self.assertEqual(cfg.thetas.to_list(), [0.0, 1.5, 2.0])

	def test_from_file_converts_string_thetas(self):
		with TemporaryDirectory() as td:
			p = Path(td) / 'model.json'
			data = {
				'feature_name': 'f',
				'target_name': 't',
				'thetas': ['1', '2.5']
			}
			p.write_text(json.dumps(data), encoding='utf-8')
			cfg = ModelConfiguration.from_file(p)
			self.assertEqual(cfg.thetas.to_list(), [1.0, 2.5])
			# no scaler fields to check

	def test_to_json_structure(self):
		cfg = ModelConfiguration(feature_name='a', target_name='b', thetas=Vector([0, 1]))
		j = cfg.to_json()
		self.assertIsInstance(j, dict)
		self.assertEqual(j['feature_name'], 'a')
		self.assertEqual(j['target_name'], 'b')
		self.assertEqual(j['thetas'], [0.0, 1.0])

	def test_save_writes_file_and_creates_parent(self):
		with TemporaryDirectory() as td:
			p = Path(td) / 'nested' / 'cfg.json'
			cfg = ModelConfiguration(feature_name='x', target_name='y', thetas=Vector([3]))
			cfg.save(p)
			self.assertTrue(p.exists())
			loaded = json.loads(p.read_text(encoding='utf-8'))
			self.assertEqual(loaded, cfg.to_json())

	def test_from_file_missing_keys_raises(self):
		with TemporaryDirectory() as td:
			p = Path(td) / 'bad.json'
			# missing 'feature' key
			p.write_text(json.dumps({'target_name': 't', 'thetas': []}), encoding='utf-8')
			with self.assertRaises(KeyError):
				ModelConfiguration.from_file(p)

if __name__ == '__main__':
	unittest.main()
