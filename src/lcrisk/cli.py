import pandas as pd
import typer

from lcrisk import config

app = typer.Typer(help="Lending Club PD model: ingest, fit, validate, expected loss.")


@app.command()
def ingest() -> None:
    """Read the raw CSV, apply population filters, write data/processed/population.parquet."""
    from lcrisk.ingest import build

    pop, waterfall = build()
    typer.echo(waterfall.to_string(index=False))
    typer.echo(f"\nPopulation: {len(pop):,} loans, bad rate {pop['bad'].mean():.2%}")


@app.command()
def profile() -> None:
    """Feature coverage and default rates by vintage."""
    from lcrisk.profile import coverage_by_vintage, default_rates

    pop = pd.read_parquet(config.PROCESSED / "population.parquet")
    by_vintage, by_grade = default_rates(pop)
    with pd.option_context("display.width", 200, "display.max_rows", 100):
        typer.echo(by_vintage.to_string())
        typer.echo(by_grade.to_string())
        typer.echo(coverage_by_vintage(pop).to_string())


@app.command()
def fit() -> None:
    """Fit scorecard and LightGBM on the development window; score dev, test and OOT samples."""
    from lcrisk.pipeline import run

    perf = run()
    with pd.option_context("display.width", 200, "display.float_format", "{:.4f}".format):
        typer.echo(perf.to_string(index=False))


@app.command()
def validate() -> None:
    """Calibration tests, stability and recalibration tables -> reports/tables/."""
    from lcrisk.validate import run

    out = run()
    with pd.option_context("display.width", 220, "display.max_rows", 200,
                           "display.float_format", "{:.4f}".format):
        for name in ["hosmer_lemeshow", "by_quarter", "feature_stability", "recalibration"]:
            typer.echo(f"\n== {name}")
            typer.echo(out[name].to_string(index=False))


@app.command()
def figures() -> None:
    """Write the documentation figures to reports/figures/."""
    from lcrisk.figures import run

    for path in run():
        typer.echo(path)


@app.command()
def el() -> None:
    """Risk parameters, EL backtest and stress scenarios -> reports/tables/."""
    from lcrisk.el import run

    out = run()
    with pd.option_context("display.width", 220, "display.float_format", "{:,.4f}".format):
        for name, t in out.items():
            typer.echo(f"\n== {name}")
            typer.echo(t.to_string(index=False))


@app.command()
def explain() -> None:
    """VIF, feature importance, reason-code agreement, cut-off value -> reports/tables/."""
    from lcrisk.explain import run

    out = run()
    with pd.option_context("display.width", 220, "display.float_format", "{:,.4f}".format):
        for name in ["vif", "importance", "reason_codes", "cutoff_value"]:
            typer.echo("== " + name)
            typer.echo(out[name].to_string(index=False))


@app.command()
def doc() -> None:
    """Build the model documentation PDF (docs/model_doc -> reports/model_doc)."""
    from lcrisk.report.writeup import build

    page, pdf = build(config.ROOT / "docs" / "model_doc" / "lc-pd-model.md", config.FIGURES,
                      config.ROOT / "reports" / "model_doc")
    typer.echo(f"wrote {page}\nwrote {pdf}")


@app.command()
def reproduce() -> None:
    """Everything from the raw CSV: ingest, fit, validate, EL and stress, figures."""
    ingest()
    fit()
    validate()
    el()
    explain()
    figures()


if __name__ == "__main__":
    app()
