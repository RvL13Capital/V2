"""
AIv4 Main Entry Point - Modernized CLI with Typer

Provides a clean, modern command-line interface for all AIv4 operations.
Replaces AIv3's argparse-based CLI with typer for better UX.

Usage:
    python main.py scan --tickers ALL --start-date 2000-01-01
    python main.py train --patterns output/patterns.csv --trials 30
    python main.py predict --ticker AAPL
    python main.py backtest --start-date 2020-01-01
"""

from __future__ import annotations

import sys
from pathlib import Path
from datetime import datetime
from typing import Optional, List

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

# Add AIv4 to path
sys.path.insert(0, str(Path(__file__).parent))

from shared.config import get_settings
from pattern_detection.services import ScanService
from training.services import TrainingService
from prediction.services import PredictionService
from analysis.services import BacktestService

app = typer.Typer(
    name="AIv4",
    help="AIv4 - Modernized consolidation pattern detection system",
    add_completion=False,
)
console = Console()


@app.command()
def scan(
    tickers: str = typer.Option(
        "ALL",
        help="Tickers to scan (comma-separated or 'ALL')"
    ),
    start_date: str = typer.Option(
        "2000-01-01",
        help="Start date (YYYY-MM-DD)"
    ),
    min_years: float = typer.Option(
        2.0,
        help="Minimum years of data required"
    ),
    output: Optional[Path] = typer.Option(
        None,
        help="Output file path (default: output/patterns_YYYYMMDD.csv)"
    ),
    format: str = typer.Option(
        "csv",
        help="Output format (csv or json)"
    ),
):
    """
    Scan tickers for consolidation patterns.

    Detects consolidation patterns using the ModernPatternTracker state machine
    and saves detected patterns to CSV/JSON.
    """
    try:
        # Create service
        service = ScanService()

        # Parse tickers
        ticker_list: List[str] | str
        if tickers == "ALL":
            ticker_list = "ALL"
        else:
            ticker_list = [t.strip() for t in tickers.split(',')]

        # Run scan
        results = service.scan_patterns(
            tickers=ticker_list,
            start_date=start_date,
            min_years=min_years,
            use_async=True
        )

        # Display summary
        service.display_summary()

        # Save results
        if output:
            service.save_results(output, format=format)
        else:
            # Default output path
            timestamp = datetime.now().strftime("%Y%m%d")
            default_path = Path(f"output/patterns_{timestamp}.{format}")
            service.save_results(default_path, format=format)

    except Exception as e:
        console.print(f"[bold red]Error: {e}[/bold red]")
        raise typer.Exit(code=1)


@app.command()
def train(
    patterns: Path = typer.Argument(
        ...,
        help="Path to labeled patterns CSV",
        exists=True,
    ),
    trials: int = typer.Option(
        30,
        help="Number of Optuna optimization trials"
    ),
    output: Path = typer.Option(
        Path("output/training"),
        help="Output directory for trained models"
    ),
    enable_deep_learning: bool = typer.Option(
        False,
        help="Enable deep learning models (CNN-BiLSTM)"
    ),
):
    """
    Train ML models on labeled patterns.

    Trains XGBoost classifier + regressor with hyperparameter optimization,
    and optionally deep learning models.
    """
    try:
        from training.models import ModelFactory

        # Create service
        model_factory = ModelFactory()
        service = TrainingService(model_factory, output_dir=output)

        # Load patterns
        console.print(f"[bold blue]Loading patterns from {patterns}...[/bold blue]")
        import pandas as pd
        df = pd.read_csv(patterns)
        console.print(f"Loaded {len(df)} patterns\n")

        # Train models
        result = service.train_all_models(
            patterns_df=df,
            optimization_trials=trials,
            enable_deep_learning=enable_deep_learning
        )

        # Display results
        service.display_training_summary(result)

        # Save report
        service.save_training_report(result, output / "training_report.txt")

        console.print(f"\n[bold green]✓ Training complete![/bold green]")
        console.print(f"Models saved to: {output}/")

    except FileNotFoundError:
        console.print(f"[bold red]Error: Patterns file not found: {patterns}[/bold red]")
        raise typer.Exit(code=1)
    except Exception as e:
        console.print(f"[bold red]Error: {e}[/bold red]")
        import traceback
        console.print(f"[dim]{traceback.format_exc()}[/dim]")
        raise typer.Exit(code=1)


@app.command()
def predict(
    ticker: str = typer.Argument(..., help="Ticker to predict"),
    model_dir: Path = typer.Option(
        Path("output/training"),
        help="Directory containing trained models"
    ),
    start_date: Optional[str] = typer.Option(
        None,
        help="Start date for data (default: 2 years ago)"
    ),
    use_hybrid: bool = typer.Option(
        True,
        help="Use hybrid predictor if available"
    ),
    show_probabilities: bool = typer.Option(
        True,
        help="Show class probabilities in output"
    ),
    explain: bool = typer.Option(
        False,
        "--explain",
        help="Enable SHAP explainability (requires SHAP package)"
    ),
    background_data: Optional[Path] = typer.Option(
        None,
        help="Background data CSV for SHAP (optional, uses training data if available)"
    ),
    top_n_features: int = typer.Option(
        5,
        help="Number of top features to show in explanations"
    ),
    export_explanations: Optional[Path] = typer.Option(
        None,
        help="Export explanations to CSV file"
    ),
):
    """
    Make predictions for active patterns in a ticker.

    Loads trained models and predicts outcome class + expected value
    for active consolidation patterns.

    Use --explain to see which features drove the predictions (requires SHAP):
      python main.py predict AAPL --explain
    """
    try:
        # Create service
        service = PredictionService(model_dir=model_dir)

        # Load models
        console.print(f"[bold blue]Loading models from {model_dir}...[/bold blue]")
        if not service.load_models(use_hybrid=use_hybrid):
            console.print(f"\n[bold red]Error: Could not load models from {model_dir}[/bold red]")
            console.print("\n[yellow]Train models first:[/yellow]")
            console.print(f"  python main.py train <patterns.csv> --output {model_dir}")
            raise typer.Exit(code=1)

        console.print("[green]✓ Models loaded successfully[/green]\n")

        # Enable explainability if requested
        if explain:
            console.print("[bold blue]Enabling SHAP explainability...[/bold blue]")

            # Load background data
            if background_data and background_data.exists():
                import pandas as pd
                X_background = pd.read_csv(background_data)
                console.print(f"[green]✓ Loaded background data from {background_data}[/green]")
            else:
                # Try to load training data as background
                training_csv = model_dir / "training_features.csv"
                if training_csv.exists():
                    import pandas as pd
                    X_background = pd.read_csv(training_csv)
                    console.print(f"[green]✓ Using training data as background ({len(X_background)} samples)[/green]")
                else:
                    console.print(
                        "[yellow]⚠ No background data available. "
                        "Explainability requires background data.[/yellow]"
                    )
                    console.print("[dim]Tip: Save training features during model training, or provide --background-data[/dim]")
                    explain = False

            if explain:
                service.enable_explainability(X_background, max_background_samples=100)

        # Make predictions
        results = service.predict_ticker(
            ticker=ticker,
            start_date=start_date
        )

        if not results:
            console.print(f"[yellow]No active patterns found for {ticker}[/yellow]")
            console.print("\n[dim]The ticker may not have any active consolidation patterns.[/dim]")
            return

        # Display predictions
        service.display_predictions(results, show_probabilities=show_probabilities)

        # Generate explanations if enabled
        if explain and service.explainability_enabled:
            console.print("\n[bold blue]═══ SHAP Feature Importance Analysis ═══[/bold blue]\n")

            # We need the features DataFrame to generate explanations
            # For now, we'll note that this requires storing features from predict_ticker
            console.print(
                "[yellow]⚠ Feature-level explanations require access to prediction features.[/yellow]"
            )
            console.print(
                "[dim]This will be available in the next update. "
                "For now, use service.explain_predictions() programmatically.[/dim]"
            )
            # TODO: Modify predict_ticker to optionally return features DataFrame
            # Then call: service.explain_predictions(results, features_df, top_n_features, export_path=export_explanations)

        # Show signal summary
        strong_signals = [r for r in results if r.signal_strength == 'STRONG_SIGNAL']
        good_signals = [r for r in results if r.signal_strength == 'GOOD_SIGNAL']

        console.print(f"\n[bold]Signal Summary:[/bold]")
        console.print(f"  Strong Signals: {len(strong_signals)}")
        console.print(f"  Good Signals: {len(good_signals)}")
        console.print(f"  Total Patterns: {len(results)}")

    except FileNotFoundError as e:
        console.print(f"[bold red]Error: {e}[/bold red]")
        raise typer.Exit(code=1)
    except Exception as e:
        console.print(f"[bold red]Error: {e}[/bold red]")
        import traceback
        console.print(f"[dim]{traceback.format_exc()}[/dim]")
        raise typer.Exit(code=1)


@app.command()
def backtest(
    start_date: str = typer.Option(
        "2020-01-01",
        help="Backtest start date (YYYY-MM-DD)"
    ),
    end_date: Optional[str] = typer.Option(
        None,
        help="Backtest end date (default: today)"
    ),
    model_dir: Path = typer.Option(
        Path("output/backtest_models"),
        help="Directory for backtest models"
    ),
    retrain_frequency: str = typer.Option(
        "90D",
        help="Model retraining frequency (e.g., '90D', '180D')"
    ),
    initial_train_years: float = typer.Option(
        2.0,
        help="Years of data for initial training"
    ),
    test_window_days: int = typer.Option(
        90,
        help="Days in each test period"
    ),
    tickers: Optional[str] = typer.Option(
        None,
        help="Tickers to backtest (comma-separated, default: all available)"
    ),
    initial_capital: float = typer.Option(
        100000.0,
        help="Initial capital for portfolio backtest"
    ),
    max_positions: int = typer.Option(
        10,
        help="Maximum concurrent positions"
    ),
    output_dir: Optional[Path] = typer.Option(
        None,
        help="Output directory for backtest results"
    ),
):
    """
    Run walk-forward backtest with periodic model retraining.

    Tests model performance on historical data with strict temporal integrity.
    Models are retrained at regular intervals to simulate realistic deployment.
    """
    try:
        # Parse end_date
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")

        # Parse tickers
        ticker_list: Optional[List[str]] = None
        if tickers:
            ticker_list = [t.strip() for t in tickers.split(',')]

        # Create service
        service = BacktestService(model_dir=model_dir)

        # Run backtest
        result = service.run_walk_forward_backtest(
            start_date=start_date,
            end_date=end_date,
            retrain_frequency=retrain_frequency,
            initial_train_years=initial_train_years,
            test_window_days=test_window_days,
            tickers=ticker_list,
            max_concurrent_positions=max_positions,
            initial_capital=initial_capital
        )

        # Display report
        service.display_report(result)

        # Save report if output directory specified
        if output_dir:
            service.save_report(result, output_dir)
        else:
            # Default output directory
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            default_dir = Path(f"output/backtest_results_{timestamp}")
            service.save_report(result, default_dir)

        console.print(f"\n[bold green]✓ Backtest complete![/bold green]")

    except Exception as e:
        console.print(f"[bold red]Error: {e}[/bold red]")
        import traceback
        console.print(f"[dim]{traceback.format_exc()}[/dim]")
        raise typer.Exit(code=1)


@app.command()
def config(
    show: bool = typer.Option(
        False,
        "--show",
        help="Show current configuration"
    ),
):
    """
    Manage configuration settings.

    Display or modify system configuration.
    """
    settings = get_settings()

    if show:
        console.print("[bold blue]AIv4 Configuration[/bold blue]\n")

        # Consolidation criteria
        table = Table(title="Consolidation Criteria")
        table.add_column("Parameter", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("BBW Threshold", f"{settings.consolidation.bbw_percentile_threshold:.2f}")
        table.add_row("ADX Threshold", f"{settings.consolidation.adx_threshold:.1f}")
        table.add_row("Volume Ratio", f"{settings.consolidation.volume_ratio_threshold:.2f}")
        table.add_row("Range Ratio", f"{settings.consolidation.range_ratio_threshold:.2f}")
        table.add_row("Qualifying Days", str(settings.consolidation.qualifying_days))
        table.add_row("Max Pattern Days", str(settings.consolidation.max_pattern_days))

        console.print(table)

        # ML configuration
        console.print(f"\n[bold]ML Configuration:[/bold]")
        console.print(f"  Estimators: {settings.ml.n_estimators}")
        console.print(f"  Max Depth: {settings.ml.max_depth}")
        console.print(f"  Learning Rate: {settings.ml.learning_rate}")

        # Deep Learning
        console.print(f"\n[bold]Deep Learning:[/bold] {'Enabled' if settings.deep_learning.enabled else 'Disabled'}")

        # Data
        console.print(f"\n[bold]Data:[/bold]")
        console.print(f"  Min Price: ${settings.data.min_price:.2f}")
        console.print(f"  Min Years: {settings.data.min_years_data:.1f}")
        console.print(f"  GCS: {'Enabled' if settings.data.use_gcs else 'Disabled'}")


@app.command()
def version():
    """Show AIv4 version information."""
    console.print("[bold blue]AIv4 - Modernized Pattern Detection System[/bold blue]")
    console.print("Version: 4.0.0")
    console.print("Architecture: Domain-Driven Design")
    console.print("\nFeatures:")
    console.print("  + Pydantic configuration")
    console.print("  + Type-safe models")
    console.print("  + Modern async data loading")
    console.print("  + Domain-driven structure")


if __name__ == "__main__":
    app()
