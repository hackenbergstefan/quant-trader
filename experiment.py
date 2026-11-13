import logging
from pathlib import Path

from pytrading import BrokerConfig
from pytrading.experiment_runner import (
    default_broker_config,
    default_experiment_config,
    run_standard_comparison,
)
from pytrading.strategies.momentum import MomentumConfig, variant_experiment

ROOT = Path(__file__).resolve().parent
START_DATE = "2010-01-01"
END_DATE = "2017-01-01"
__all__ = ["variant_experiment"]


def _experiment_broker_config() -> BrokerConfig:
    return default_broker_config()


def _experiment_config() -> MomentumConfig:
    return default_experiment_config()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(processName)s %(message)s",
    )
    output_dir = run_standard_comparison(output_root=ROOT / "results" / "experiments")
    print(f"Experiment report: {output_dir / 'report.html'}")


if __name__ == "__main__":
    main()
