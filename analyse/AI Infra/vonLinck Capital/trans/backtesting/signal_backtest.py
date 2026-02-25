"""
Signal Strategy Backtester
==========================

Simulates trading the breakout signal patterns historically.

Strategy Rules:
- Entry: When signal fires (Vol>2x + ROC>8% or other criteria)
- Exit: TARGET hit, DEAD triggered, or max holding period
- Position sizing: Equal weight or risk-based
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


@dataclass
class Trade:
    """Represents a single trade."""
    ticker: str
    entry_date: datetime
    entry_price: float
    target_price: float
    stop_price: float
    upper_boundary: float
    lower_boundary: float

    # Filled on exit
    exit_date: Optional[datetime] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None  # TARGET, DEAD, COMPLETED, TIMEOUT, FAILED

    # Calculated
    pnl_pct: Optional[float] = None
    holding_days: Optional[int] = None

    def close(self, exit_date: datetime, exit_price: float, reason: str):
        """Close the trade."""
        self.exit_date = exit_date
        self.exit_price = exit_price
        self.exit_reason = reason
        self.pnl_pct = (exit_price - self.entry_price) / self.entry_price
        self.holding_days = (exit_date - self.entry_date).days

    def to_dict(self) -> Dict:
        return {
            'ticker': self.ticker,
            'entry_date': self.entry_date,
            'entry_price': self.entry_price,
            'target_price': self.target_price,
            'stop_price': self.stop_price,
            'exit_date': self.exit_date,
            'exit_price': self.exit_price,
            'exit_reason': self.exit_reason,
            'pnl_pct': self.pnl_pct,
            'holding_days': self.holding_days,
        }


@dataclass
class BacktestConfig:
    """Configuration for backtest."""
    # Signal criteria
    min_volume_ratio: float = 2.0
    min_roc_10: float = 0.08
    min_rsi: float = 50.0
    min_price_position: float = 0.5

    # Exit rules
    max_holding_days: int = 100
    target_mult: float = 0.20  # 20% above upper boundary or 3x corridor
    corridor_mult: float = 3.0

    # Risk management
    max_concurrent_positions: int = 10
    position_size_pct: float = 0.10  # 10% per position

    # Trading costs
    commission_pct: float = 0.001  # 0.1% per trade


@dataclass
class BacktestResult:
    """Results from backtest."""
    trades: List[Trade]
    equity_curve: pd.Series

    # Performance metrics
    total_return: float = 0.0
    annualized_return: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0
    total_trades: int = 0

    # By exit reason
    exits_by_reason: Dict[str, int] = field(default_factory=dict)


class SignalBacktester:
    """
    Backtests the breakout signal strategy.

    Uses the training data to simulate what would have happened
    if we traded every signal that matched our criteria.
    """

    def __init__(self, config: Optional[BacktestConfig] = None):
        self.config = config or BacktestConfig()
        self.trades: List[Trade] = []

    def run(self, training_data: pd.DataFrame, price_data: Dict[str, pd.DataFrame]) -> BacktestResult:
        """
        Run backtest on training data.

        Args:
            training_data: DataFrame with labeled patterns
            price_data: Dict of ticker -> OHLCV DataFrame

        Returns:
            BacktestResult with all metrics
        """
        # Filter signals based on criteria
        signals = self._filter_signals(training_data)
        logger.info(f"Found {len(signals)} signals matching criteria")

        if len(signals) == 0:
            return BacktestResult(trades=[], equity_curve=pd.Series())

        # Sort by date
        signals = signals.sort_values('signal_date')

        # Simulate trades
        self.trades = []
        active_positions: Dict[str, Trade] = {}

        for _, signal in signals.iterrows():
            ticker = signal['ticker']
            signal_date = pd.to_datetime(signal['signal_date'])

            # Skip if already in position for this ticker
            if ticker in active_positions:
                continue

            # Skip if max positions reached
            if len(active_positions) >= self.config.max_concurrent_positions:
                continue

            # Get price data for ticker
            if ticker not in price_data:
                continue

            ticker_prices = price_data[ticker]
            if 'date' not in ticker_prices.columns:
                ticker_prices = ticker_prices.reset_index()
            ticker_prices['date'] = pd.to_datetime(ticker_prices['date'])

            # Find entry point (next day after signal)
            future_prices = ticker_prices[ticker_prices['date'] > signal_date]
            if len(future_prices) == 0:
                continue

            entry_row = future_prices.iloc[0]
            entry_price = entry_row['open']  # Enter at open
            entry_date = entry_row['date']

            # Calculate target and stop
            upper_boundary = signal['upper_boundary']
            lower_boundary = signal['lower_boundary']
            corridor_width_pct = (upper_boundary - lower_boundary) / lower_boundary

            target_pct = max(self.config.target_mult, self.config.corridor_mult * corridor_width_pct)
            target_price = upper_boundary * (1 + target_pct)
            stop_price = signal.get('last_active_lowest_close', lower_boundary)
            if pd.isna(stop_price):
                stop_price = lower_boundary

            # Create trade
            trade = Trade(
                ticker=ticker,
                entry_date=entry_date,
                entry_price=entry_price,
                target_price=target_price,
                stop_price=stop_price,
                upper_boundary=upper_boundary,
                lower_boundary=lower_boundary,
            )

            active_positions[ticker] = trade

        # Process all active positions through time
        all_dates = set()
        for ticker, df in price_data.items():
            if 'date' in df.columns:
                all_dates.update(pd.to_datetime(df['date']).tolist())
            else:
                all_dates.update(pd.to_datetime(df.index).tolist())

        all_dates = sorted(all_dates)

        # Track equity over time
        initial_capital = 100000
        cash = initial_capital
        equity_history = []

        # Re-run with proper position tracking
        self.trades = []
        active_positions = {}
        signals_iter = iter(signals.iterrows())
        next_signal = next(signals_iter, (None, None))[1]

        for current_date in all_dates:
            # Check for new signals
            while next_signal is not None and pd.to_datetime(next_signal['signal_date']) <= current_date:
                ticker = next_signal['ticker']

                if ticker not in active_positions and len(active_positions) < self.config.max_concurrent_positions:
                    if ticker in price_data:
                        ticker_prices = price_data[ticker]
                        if 'date' not in ticker_prices.columns:
                            ticker_prices = ticker_prices.reset_index()
                        ticker_prices['date'] = pd.to_datetime(ticker_prices['date'])

                        future_prices = ticker_prices[ticker_prices['date'] > pd.to_datetime(next_signal['signal_date'])]
                        if len(future_prices) > 0:
                            entry_row = future_prices.iloc[0]
                            entry_price = entry_row['open']
                            entry_date = entry_row['date']

                            upper_boundary = next_signal['upper_boundary']
                            lower_boundary = next_signal['lower_boundary']
                            corridor_width_pct = (upper_boundary - lower_boundary) / lower_boundary

                            target_pct = max(self.config.target_mult, self.config.corridor_mult * corridor_width_pct)
                            target_price = upper_boundary * (1 + target_pct)
                            stop_price = lower_boundary * 0.98  # 2% below lower boundary

                            trade = Trade(
                                ticker=ticker,
                                entry_date=entry_date,
                                entry_price=entry_price,
                                target_price=target_price,
                                stop_price=stop_price,
                                upper_boundary=upper_boundary,
                                lower_boundary=lower_boundary,
                            )
                            active_positions[ticker] = trade

                try:
                    next_signal = next(signals_iter)[1]
                except StopIteration:
                    next_signal = None

            # Update positions
            closed_tickers = []
            for ticker, trade in active_positions.items():
                if ticker not in price_data:
                    continue

                ticker_prices = price_data[ticker]
                if 'date' not in ticker_prices.columns:
                    ticker_prices = ticker_prices.reset_index()
                ticker_prices['date'] = pd.to_datetime(ticker_prices['date'])

                day_data = ticker_prices[ticker_prices['date'] == current_date]
                if len(day_data) == 0:
                    continue

                day = day_data.iloc[0]
                close = day['close']
                low = day['low']
                high = day['high']

                holding_days = (current_date - trade.entry_date).days

                # Check exit conditions
                exit_reason = None
                exit_price = None

                # Check stop (DEAD)
                if low <= trade.stop_price:
                    exit_reason = 'DEAD'
                    exit_price = trade.stop_price
                # Check target
                elif high >= trade.target_price:
                    exit_reason = 'TARGET'
                    exit_price = trade.target_price
                # Check completed (2 closes above upper) - simplified
                elif close > trade.upper_boundary * 1.02:
                    # Could track consecutive closes, but simplified here
                    pass
                # Check timeout
                elif holding_days >= self.config.max_holding_days:
                    exit_reason = 'TIMEOUT'
                    exit_price = close

                if exit_reason:
                    trade.close(current_date, exit_price, exit_reason)
                    self.trades.append(trade)
                    closed_tickers.append(ticker)

            # Remove closed positions
            for ticker in closed_tickers:
                del active_positions[ticker]

            # Calculate equity
            position_value = sum(
                trade.entry_price * (1 + (price_data[trade.ticker].loc[
                    price_data[trade.ticker]['date'] == current_date, 'close'
                ].values[0] - trade.entry_price) / trade.entry_price)
                if trade.ticker in price_data and
                   len(price_data[trade.ticker][price_data[trade.ticker]['date'] == current_date]) > 0
                else trade.entry_price
                for trade in active_positions.values()
            ) if active_positions else 0

            # Simplified equity calculation
            closed_pnl = sum(t.pnl_pct or 0 for t in self.trades) * initial_capital * self.config.position_size_pct
            equity = initial_capital + closed_pnl
            equity_history.append({'date': current_date, 'equity': equity})

        # Close any remaining positions at last price
        for ticker, trade in active_positions.items():
            if ticker in price_data:
                ticker_prices = price_data[ticker]
                if 'date' not in ticker_prices.columns:
                    ticker_prices = ticker_prices.reset_index()
                last_price = ticker_prices.iloc[-1]['close']
                trade.close(ticker_prices.iloc[-1]['date'], last_price, 'END')
                self.trades.append(trade)

        # Calculate metrics
        return self._calculate_metrics(equity_history)

    def _filter_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter signals based on config criteria."""
        mask = pd.Series([True] * len(df))

        if 'volume_ratio' in df.columns:
            mask &= df['volume_ratio'] >= self.config.min_volume_ratio

        if 'roc_10' in df.columns:
            mask &= df['roc_10'] >= self.config.min_roc_10

        if 'rsi_14' in df.columns:
            mask &= df['rsi_14'] >= self.config.min_rsi

        if 'price_position' in df.columns:
            mask &= df['price_position'] >= self.config.min_price_position

        return df[mask]

    def _calculate_metrics(self, equity_history: List[Dict]) -> BacktestResult:
        """Calculate performance metrics."""
        if not self.trades:
            return BacktestResult(trades=[], equity_curve=pd.Series())

        equity_df = pd.DataFrame(equity_history)
        if len(equity_df) == 0:
            equity_curve = pd.Series()
        else:
            equity_curve = equity_df.set_index('date')['equity']

        # Trade statistics
        pnls = [t.pnl_pct for t in self.trades if t.pnl_pct is not None]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]

        total_trades = len(pnls)
        win_rate = len(wins) / total_trades if total_trades > 0 else 0
        avg_win = np.mean(wins) if wins else 0
        avg_loss = np.mean(losses) if losses else 0

        total_wins = sum(wins)
        total_losses = abs(sum(losses))
        profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')

        # Returns
        total_return = sum(pnls)

        # Count exits by reason
        exits_by_reason = {}
        for trade in self.trades:
            reason = trade.exit_reason or 'UNKNOWN'
            exits_by_reason[reason] = exits_by_reason.get(reason, 0) + 1

        # Drawdown
        if len(equity_curve) > 0:
            rolling_max = equity_curve.expanding().max()
            drawdown = (equity_curve - rolling_max) / rolling_max
            max_drawdown = drawdown.min()
        else:
            max_drawdown = 0

        # Annualized return and Sharpe
        if len(pnls) > 0 and total_trades > 0:
            # Approximate trading days
            if len(equity_df) > 1:
                days = (equity_df['date'].max() - equity_df['date'].min()).days
                years = days / 365
                annualized_return = (1 + total_return) ** (1/years) - 1 if years > 0 else total_return
            else:
                annualized_return = total_return

            # Sharpe (simplified)
            if np.std(pnls) > 0:
                sharpe_ratio = np.mean(pnls) / np.std(pnls) * np.sqrt(252 / (np.mean([t.holding_days or 1 for t in self.trades])))
            else:
                sharpe_ratio = 0
        else:
            annualized_return = 0
            sharpe_ratio = 0

        return BacktestResult(
            trades=self.trades,
            equity_curve=equity_curve,
            total_return=total_return,
            annualized_return=annualized_return,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            profit_factor=profit_factor,
            total_trades=total_trades,
            exits_by_reason=exits_by_reason,
        )

    def print_summary(self, result: BacktestResult):
        """Print backtest summary."""
        print("=" * 60)
        print("BACKTEST RESULTS")
        print("=" * 60)
        print()
        print(f"Total Trades: {result.total_trades}")
        print(f"Win Rate: {result.win_rate:.1%}")
        print(f"Avg Win: {result.avg_win:.2%}")
        print(f"Avg Loss: {result.avg_loss:.2%}")
        print(f"Profit Factor: {result.profit_factor:.2f}")
        print()
        print(f"Total Return: {result.total_return:.2%}")
        print(f"Annualized Return: {result.annualized_return:.2%}")
        print(f"Sharpe Ratio: {result.sharpe_ratio:.2f}")
        print(f"Max Drawdown: {result.max_drawdown:.2%}")
        print()
        print("Exits by Reason:")
        for reason, count in sorted(result.exits_by_reason.items()):
            print(f"  {reason}: {count} ({count/result.total_trades*100:.1f}%)")


def run_backtest(
    training_data_path: str = 'output/training_data_v2.parquet',
    data_dir: str = 'data/raw',
    config: Optional[BacktestConfig] = None
) -> BacktestResult:
    """
    Run backtest from files.

    Args:
        training_data_path: Path to training data parquet
        data_dir: Directory with price data CSVs
        config: Backtest configuration

    Returns:
        BacktestResult
    """
    from pathlib import Path

    # Load training data
    training_data = pd.read_parquet(training_data_path)
    print(f"Loaded {len(training_data)} patterns")

    # Load price data for all tickers
    tickers = training_data['ticker'].unique()
    price_data = {}

    data_path = Path(data_dir)
    for ticker in tickers:
        csv_path = data_path / f"{ticker}.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            df.columns = df.columns.str.lower()
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
            price_data[ticker] = df

    print(f"Loaded price data for {len(price_data)} tickers")

    # Run backtest
    backtester = SignalBacktester(config)
    result = backtester.run(training_data, price_data)
    backtester.print_summary(result)

    return result


if __name__ == '__main__':
    import logging
    logging.basicConfig(level=logging.INFO)

    # Run with default config (Vol>2x + ROC>8%)
    config = BacktestConfig(
        min_volume_ratio=2.0,
        min_roc_10=0.08,
        min_rsi=50.0,
        min_price_position=0.5,
    )

    result = run_backtest(config=config)
