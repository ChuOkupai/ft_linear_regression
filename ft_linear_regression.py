import argparse
from pathlib import Path
from srcs import Dataset, GradientDescentTrainer, LinearRegressionModel, ModelConfiguration, RegressionVisualizer, Vector, ModelEvaluator

DEFAULT_LEARNING_RATE = 0.1
DEFAULT_EPOCHS = 1000

class CommandLineApplication:
	def build_parser(self) -> argparse.ArgumentParser:
		parser = argparse.ArgumentParser(prog='ft_linear_regression', add_help=True, description='Train and use a simple linear regression model.')
		sub = parser.add_subparsers(dest='command', required=True)

		train = sub.add_parser('train', help='Train a linear regression model from a CSV dataset.')
		train.add_argument('-d', '--dataset', type=str, required=True, help='Path to CSV dataset (needs at least two columns).')
		train.add_argument('--feature', type=str, default=None, help='Feature column name (override first column).')
		train.add_argument('--target', type=str, default=None, help='Target column name (override second column).')
		train.add_argument('-e', '--epochs', type=int, default=1000, help='Number of training epochs (default: 1000).')
		train.add_argument('-l', '--learning-rate', type=float, default=None, help='Learning rate (default: 0.01 if omitted).')
		train.add_argument('-o', '--output', type=str, default=None, help='Output path for model JSON (mirrors dataset under models/ if omitted).')
		train.add_argument('-p', '--plot', action='store_true', help='Display interactive plot after training.')
		train.add_argument('-s', '--save-plot', nargs='?', const='', default=None, help='Save regression plot (optional path; defaults to model path with .png).')
		train.add_argument('--statistics', action='store_true', help='Display training statistics (MSR, RMSE, R2, etc.) after training.')

		predict = sub.add_parser('predict', help='Generate predictions from a trained model.')
		predict.add_argument('-m', '--model', type=str, required=True, help='Path to trained model JSON file.')

		return parser

	def _predict(self, args: argparse.Namespace):
		try:
			cfg = ModelConfiguration.from_file(Path(args.model))
		except FileNotFoundError as e:
			print(f'Warning: {e.filename}: {e.strerror}. Using default configuration.')
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
				input_val = float(raw)
				print(f'{cfg.feature_name}={input_val} {cfg.target_name}={model.predict(input_val)}')
			except ValueError:
				print('Error: Invalid number')

	def _train(self, args: argparse.Namespace):
		dataset_path = Path(args.dataset)
		epochs = int(args.epochs) if args.epochs is not None else DEFAULT_EPOCHS
		learning_rate = float(args.learning_rate) if args.learning_rate is not None else DEFAULT_LEARNING_RATE
		dataset = Dataset.from_csv(dataset_path, args.feature, args.target)
		output = Path(args.output) if args.output else None

		trainer = GradientDescentTrainer(LinearRegressionModel(), learning_rate)
		parameters = trainer.train(dataset, epochs)
		if output is None:
			model_dir = Path('models')
			model_dir.mkdir(parents=True, exist_ok=True)
			output = model_dir / (dataset_path.stem + '.json')
		ModelConfiguration(parameters, dataset.feature_name, dataset.target_name).save(output)
		print(f'Model saved: {output}')

		if args.statistics:
			metrics = ModelEvaluator().evaluate(dataset, parameters)
			def fmt(v: float) -> str:
				return f"{v:.6f}"
			print('Training statistics:')
			print(f"  Samples           : {dataset.size}")
			print(f"  Parameters (θ0,θ1): ({fmt(parameters[0])}, {fmt(parameters[1])})")
			print('  Metrics:')
			print(f"    MSE             : {fmt(metrics['MSE'])}")
			print(f"    RMSE            : {fmt(metrics['RMSE'])}")
			print(f"    MAE             : {fmt(metrics['MAE'])}")
			print(f"    R2              : {fmt(metrics['R2'])}")

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

	def run(self, argv: list[str] | None = None) -> int:
		parser = self.build_parser()
		args = parser.parse_args(argv)
		try:
			self._predict(args) if args.command == 'predict' else self._train(args)
		except Exception as e:
			print(f'Error: {e}')
			return 1
		return 0

def main() -> int:
	return CommandLineApplication().run()

if __name__ == '__main__':
    main()
