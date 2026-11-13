import argparse
import json
from pathlib import Path
import yaml

from src.experiment import ExperimentRunner


def main() -> None:
    """Parses command-line arguments, loads the config, and executes the OOP experiment runner."""
    parser = argparse.ArgumentParser(description="Run a transfer learning experiment.")
    parser.add_argument("--config", type=Path, required=True, help="Path to the YAML configuration file.")
    parser.add_argument("--model", required=True, help="Name or identifier of the model architecture.")
    parser.add_argument(
        "--strategy",
        required=True,
        choices=["feature_extraction", "partial_ft", "full_ft"],
        help="Fine-tuning strategy to apply to the model backbone.",
    )
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))

    # Instantiate the class-based runner and execute
    runner = ExperimentRunner(cfg=config, model_name=args.model, strategy=args.strategy)
    _, metrics, _ = runner.run()

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()