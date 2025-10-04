from collections.abc import Iterable
import csv
from pathlib import Path
from . import Vector

class Dataset:
	def __init__(self, features: Iterable[float], targets: Iterable[float], feature_name: str, target_name: str) -> None:
		self._features = Vector(features)
		self._targets = Vector(targets)
		self._feature_name = feature_name
		self._target_name = target_name
		self._scaler = None
		if len(self._features) == 0:
			raise ValueError('Dataset must contain at least one sample')
		if len(self._features) != len(self._targets):
			raise ValueError('Features and targets must have the same number of samples')

	@classmethod
	def from_csv(cls, path: Path | str, feature: str | None = None, target: str | None = None) -> 'Dataset':
		with Path(path).open(newline='') as f:
			reader = csv.DictReader(f)
			if reader.fieldnames is None or len(reader.fieldnames) < 2:
				raise ValueError('Dataset must contain at least two columns')
			fieldnames = reader.fieldnames
			if (feature is None) ^ (target is None):
				raise ValueError('Both feature and target column names must be provided together')
			if feature and target:
				if feature not in fieldnames or target not in fieldnames:
					raise ValueError('Provided feature/target column names must exist in CSV')
				if feature == target:
					raise ValueError('Feature and target column names must be different')
				feature_name, target_name = feature, target
			else:
				feature_name, target_name = fieldnames[:2]
			features: list[float] = []
			targets: list[float] = []
			for row in reader:
				features.append(float(row[feature_name]))
				targets.append(float(row[target_name]))
		return cls(features, targets, feature_name, target_name)

	@property
	def size(self) -> int:
		return len(self._features)

	@property
	def feature_name(self) -> str:
		return self._feature_name

	@property
	def target_name(self) -> str:
		return self._target_name

	@property
	def features(self) -> Vector:
		return self._features

	@property
	def targets(self) -> Vector:
		return self._targets
