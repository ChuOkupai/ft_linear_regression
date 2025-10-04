import json
from dataclasses import dataclass
from pathlib import Path
from . import Vector

@dataclass
class ModelConfiguration:
	thetas: Vector
	feature_name: str
	target_name: str

	@classmethod
	def from_file(cls, path: Path | str) -> 'ModelConfiguration':
		with Path(path).open('r', encoding='utf-8') as f:
			data = json.load(f)
		return cls(
			feature_name=str(data['feature_name']),
			target_name=str(data['target_name']),
			thetas=Vector(data['thetas'])
		)

	def to_json(self) -> dict:
		return {
			'feature_name': self.feature_name,
			'target_name': self.target_name,
			'thetas': self.thetas.to_list()
		}

	def save(self, path: Path | str) -> None:
		p = Path(path)
		p.parent.mkdir(parents=True, exist_ok=True)
		with p.open('w', encoding='utf-8') as f:
			json.dump(self.to_json(), f, indent=2)
