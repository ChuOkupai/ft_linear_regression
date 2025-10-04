import matplotlib.pyplot as plt
import matplotx
from collections.abc import Iterable
from pathlib import Path
from . import Dataset, Vector

class RegressionVisualizer:
	def plot(self, dataset: Dataset, parameters: Vector, save_path: str | Path | None = None, show: bool = False):
		if len(parameters) != 2:
			raise ValueError('Parameters vector must have exactly two elements (intercept and slope)')
		features = dataset.features.to_list()
		targets = dataset.targets.to_list()
		with plt.style.context(matplotx.styles.duftify(matplotx.styles.dracula)):
			figure, axis = plt.subplots(figsize=(9, 5.5), dpi=120)
			axis.scatter(features, targets, s=60, c='#2563eb', edgecolors='#ffffff', linewidths=0.6, alpha=0.9)
			line_x = self._build_line_values(features)
			line_y = [parameters[0] + parameters[1] * value for value in line_x]
			axis.plot(line_x, line_y)
			axis.set_xlabel(dataset.feature_name, fontweight='semibold')
			axis.set_ylabel(dataset.target_name, fontweight='semibold')
			axis.set_title(f'Distribution of {dataset.target_name} vs {dataset.feature_name} — regression fit', fontweight='bold', fontsize=14)
			axis.grid(True, which='major', linestyle='--', linewidth=0.6, alpha=0.6)
			for spine in axis.spines.values():
				spine.set_visible(False)
			figure.tight_layout()
			if save_path is not None:
				save_path = Path(save_path).resolve()
				save_path.parent.mkdir(parents=True, exist_ok=True)
				figure.savefig(save_path, bbox_inches='tight')
			if show:
				plt.show()
			return figure

	def _build_line_values(self, features: Iterable[float]) -> list[float]:
		values = list(features)
		minimum = min(values)
		maximum = max(values)
		if minimum == maximum:
			offset = 1.0 if minimum == 0 else abs(minimum) * 0.1
			return [minimum - offset, maximum + offset]
		step = (maximum - minimum) / 199
		return [minimum + step * index for index in range(200)]
