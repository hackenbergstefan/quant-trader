from pathlib import Path

import pandas as pd
import typer

from pytrading.experiment_runner import (
    DEFAULT_END_DATE,
    DEFAULT_INITIAL_CAPITAL,
    DEFAULT_START_DATE,
    run_standard_comparison,
)
from pytrading.rolling_analysis import run_standard_rolling_comparison

app = typer.Typer(no_args_is_help=True)
experiment_app = typer.Typer(no_args_is_help=True)
START_DATE_OPTION = typer.Option(DEFAULT_START_DATE, help="Inclusive start date.")
END_DATE_OPTION = typer.Option(DEFAULT_END_DATE, help="Inclusive end date.")
INITIAL_CAPITAL_OPTION = typer.Option(
    DEFAULT_INITIAL_CAPITAL, min=0.01, help="Initial portfolio capital in EUR."
)
OUTPUT_ROOT_OPTION = typer.Option(
    Path("results/experiments"),
    file_okay=False,
    dir_okay=True,
    help="Parent directory for timestamped experiment outputs.",
)


@experiment_app.command("compare")
def compare(
    start_date: str = START_DATE_OPTION,
    end_date: str = END_DATE_OPTION,
    initial_capital: float = INITIAL_CAPITAL_OPTION,
    output_root: Path = OUTPUT_ROOT_OPTION,
) -> None:
    try:
        output_dir = run_standard_comparison(
            start_date=start_date,
            end_date=end_date,
            initial_capital=initial_capital,
            output_root=output_root,
        )
    except Exception as exc:
        typer.secho(
            f"Experiment comparison failed: {type(exc).__name__}: {exc}",
            fg=typer.colors.RED,
        )
        raise typer.Exit(code=1) from exc
    typer.echo(f"Experiment outputs: {output_dir}")
    typer.echo(f"Report: {output_dir / 'report.html'}")
    results = pd.read_csv(output_dir / "results.csv")
    failures = results.loc[results["Status"] != "success"]
    if not failures.empty:
        typer.secho(
            f"{len(failures)} experiment variant(s) failed; see results.csv/report.html.",
            fg=typer.colors.RED,
        )
        raise typer.Exit(code=1)


@experiment_app.command("rolling-3y")
def rolling_3y(
    start_date: str = typer.Option(
        DEFAULT_START_DATE, help="Inclusive start date for rolling windows."
    ),
    initial_capital: float = INITIAL_CAPITAL_OPTION,
    output_root: Path = OUTPUT_ROOT_OPTION,
) -> None:
    try:
        output_dir = run_standard_rolling_comparison(
            start_date=start_date,
            initial_capital=initial_capital,
            output_root=output_root,
        )
    except Exception as exc:
        typer.secho(
            f"Rolling experiment failed: {type(exc).__name__}: {exc}",
            fg=typer.colors.RED,
        )
        raise typer.Exit(code=1) from exc
    typer.echo(f"Rolling experiment outputs: {output_dir}")
    typer.echo(f"Report: {output_dir / 'report.html'}")
    results = pd.read_csv(output_dir / "rolling_3y_results.csv")
    failures = results.loc[results["Status"] != "success"]
    if not failures.empty:
        typer.secho(
            f"{len(failures)} window variant(s) failed; see rolling_3y_results.csv.",
            fg=typer.colors.RED,
        )
        raise typer.Exit(code=1)


app.add_typer(experiment_app, name="experiment")
