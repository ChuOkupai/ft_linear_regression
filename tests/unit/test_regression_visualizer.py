import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import unittest
from collections.abc import Iterable
from pathlib import Path
from typing import cast
from unittest.mock import patch
from srcs import Dataset, RegressionVisualizer, Vector

class TestRegressionVisualizer(unittest.TestCase):
	def tearDown(self) -> None:
		plt.close('all')

	def test_plot_labels_and_line(self) -> None:
		dataset = Dataset([0.0, 1.0, 2.0, 3.0], [1.0, 3.0, 5.0, 7.0], 'km', 'price')
		parameters = Vector([1.0, 2.0])
		visualizer = RegressionVisualizer()
		figure = visualizer.plot(dataset, parameters)
		axis = figure.axes[0]
		self.assertEqual(axis.get_xlabel(), 'km')
		self.assertEqual(axis.get_ylabel(), 'price')
		line = axis.get_lines()[0]
		x_data = list(cast(Iterable[float], line.get_xdata()))
		self.assertGreaterEqual(len(x_data), 2)
		y_data = list(cast(Iterable[float], line.get_ydata()))
		self.assertAlmostEqual(y_data[0], 1.0 + 2.0 * x_data[0], places=6)

	def test_plot_can_save_to_path(self) -> None:
		dataset = Dataset([1.0, 2.0], [2.0, 4.0], 'feature', 'target')
		parameters = Vector([0.0, 2.0])
		visualizer = RegressionVisualizer()
		output = Path('tests') / 'artifacts' / 'plot.png'
		if output.exists():
			output.unlink()
		figure = visualizer.plot(dataset, parameters, output)
		self.assertTrue(output.exists())
		axis = figure.axes[0]
		offsets = list(cast(Iterable[object], axis.collections[0].get_offsets()))
		self.assertEqual(len(offsets), dataset.size)
		output.unlink()

	def test_build_line_values_when_min_equals_max(self) -> None:
		dataset_zero = Dataset([0.0, 0.0], [1.0, 1.0], 'f', 't')
		visualizer = RegressionVisualizer()
		line_vals = visualizer._build_line_values(dataset_zero.features.to_list())
		self.assertEqual(len(line_vals), 2)
		self.assertAlmostEqual(line_vals[0], -1.0)
		self.assertAlmostEqual(line_vals[1], 1.0)

		dataset_nonzero = Dataset([2.0, 2.0], [3.0, 3.0], 'f', 't')
		line_vals = visualizer._build_line_values(dataset_nonzero.features.to_list())
		self.assertEqual(len(line_vals), 2)
		self.assertAlmostEqual(line_vals[0], 2.0 - 0.2)
		self.assertAlmostEqual(line_vals[1], 2.0 + 0.2)

	def test_plot_show_calls_plt_show(self) -> None:
		dataset = Dataset([0.0, 1.0], [0.0, 1.0], 'x', 'y')
		parameters = Vector([0.0, 1.0])
		visualizer = RegressionVisualizer()
		with patch('matplotlib.pyplot.show') as mock_show:
			figure = visualizer.plot(dataset, parameters, show=True)
			mock_show.assert_called_once()

	def test_plot_raises_with_invalid_parameters_length(self) -> None:
		dataset = Dataset([0.0, 1.0], [0.0, 1.0], 'x', 'y')
		visualizer = RegressionVisualizer()
		with self.assertRaises(ValueError):
			visualizer.plot(dataset, Vector([1.0, 2.0, 3.0]))

if __name__ == '__main__':
	unittest.main()
