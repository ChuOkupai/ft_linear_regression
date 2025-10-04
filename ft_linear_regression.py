import argparse
from pathlib import Path
from srcs import Dataset, GradientDescentTrainer, LinearRegressionModel, ModelConfiguration, RegressionVisualizer, Vector

DEFAULT_LEARNING_RATE = 0.1
DEFAULT_EPOCHS = 1000

class CommandLineApplication:
	def build_parser(self) -> argparse.ArgumentParser:
		parser = argparse.ArgumentParser(prog='ft_linear_regression', add_help=True, description='Train and use a simple linear regression model.')
		sub = parser.add_subparsers(dest='command', required=True)

		train = sub.add_parser('train', help='Train a linear regression model from a CSV dataset.')
		train.add_argument('dataset', type=str, help='Path to CSV dataset (needs at least two columns).')
		train.add_argument('--feature', type=str, default=None, help='Feature column name (override first column).')
		train.add_argument('--target', type=str, default=None, help='Target column name (override second column).')
		train.add_argument('-e', '--epochs', type=int, default=1000, help='Number of training epochs (default: 1000).')
		train.add_argument('-l', '--learning-rate', type=float, default=None, help='Learning rate (default: 0.01 if omitted).')
		train.add_argument('-o', '--output', type=str, default=None, help='Output path for model JSON (mirrors dataset under models/ if omitted).')
		train.add_argument('-p', '--plot', action='store_true', help='Display interactive plot after training.')
		train.add_argument('-s', '--save-plot', nargs='?', const='', default=None, help='Save regression plot (optional path; defaults to model path with .png).')

		predict = sub.add_parser('predict', help='Generate predictions from a trained model.')
		predict.add_argument('model', type=str, help='Path to trained model JSON file.')
		return parser

	def run(self, argv: list[str] | None = None) -> int:
		parser = self.build_parser()
		args = parser.parse_args(argv)
		if args.command == 'train':
			dataset_path = Path(args.dataset)
			feature = args.feature
			target = args.target
			epochs = int(args.epochs) if args.epochs is not None else DEFAULT_EPOCHS
			learning_rate = float(args.learning_rate) if args.learning_rate is not None else DEFAULT_LEARNING_RATE
			output = Path(args.output) if args.output else None
			try:
				dataset = Dataset.from_csv(dataset_path, feature=feature, target=target)
			except Exception as e:
				print(f'Error: {e}')
				return 1
			model = LinearRegressionModel(None)
			trainer = GradientDescentTrainer(model, learning_rate)
			parameters = trainer.train(dataset, epochs)
			if output is None:
				model_dir = Path('models')
				model_dir.mkdir(parents=True, exist_ok=True)
				output = model_dir / (dataset_path.stem + '.json')
			cfg = ModelConfiguration(parameters, dataset.feature_name, dataset.target_name)
			cfg.save(output)
			print(f'Model saved: {output}')

			visualizer = RegressionVisualizer()
			save_plot_arg = args.save_plot
			if save_plot_arg is not None:
				if save_plot_arg == '':
					plot_path = output.with_suffix('.png')
				else:
					plot_path = Path(save_plot_arg)
				visualizer.plot(dataset, parameters, save_path=plot_path, show=args.plot)
			elif args.plot:
				visualizer.plot(dataset, parameters, show=True)
			return 0

		if args.command == 'predict':
			try:
				try:
					cfg = ModelConfiguration.from_file(Path(args.model))
				except FileNotFoundError as e:
					print(f'Warning: {e.filename}: {e.strerror}.')
					cfg = ModelConfiguration(Vector([0.0, 0.0]), 'feature', 'target')
				model = LinearRegressionModel(cfg.thetas)
				print('Enter a feature value to predict (blank line to quit)')
				while True:
					try:
						raw = input('> ')
					except EOFError:
						print()
						break
					except KeyboardInterrupt:
						print()
						break
					if raw.strip() == '':
						continue
					try:
						val = float(raw)
						pred = model.predict(val)
						print(f'{cfg.feature_name}={val} {cfg.target_name}={pred}')
					except ValueError:
						print('Error: Invalid number')
			except Exception as exc:
				print(f'Error: {exc}')
				return 1
			return 0
		return 0

def main() -> int:
	return CommandLineApplication().run()

if __name__ == '__main__':
	main()
