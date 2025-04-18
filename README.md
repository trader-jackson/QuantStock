# Stock Prediction Methods

A comprehensive framework for stock return prediction and portfolio backtesting.

## Overview

This repository implements various predictive models for stock returns, including Transformer-based architectures, LSTM networks, and tree-based models such as XGBoost and LightGBM. The framework provides a unified interface for training, validating, and backtesting these models on financial time series data.

## Key Features

- **Multiple Model Architectures**: Transformer, LSTM, XGBoost, and LightGBM implementations
- **End-to-End Pipeline**: From data preprocessing to model training and evaluation
- **Sophisticated Backtesting**: Implements a trading strategy with configurable parameters:
  - Top-K stock selection based on model predictions
  - Customizable holding period
  - Performance comparison against market benchmark
- **Comprehensive Metrics**: Sharpe ratio, Sortino ratio, Information ratio, Maximum drawdown, etc.
- **Detailed Logging**: Model-specific logging with performance tracking
- **Results Export**: Save performance metrics, returns, and portfolio holdings to CSV and pickle files

## Project Structure

```
├── PM.py                    # Base class for prediction models
├── PM_transformer.py        # Transformer-based prediction model
├── PM_lstm.py               # LSTM-based prediction model
├── PM_xgboost.py            # XGBoost-based prediction model
├── PM_lightgbm.py           # LightGBM-based prediction model
├── model_config/            # Configuration files for different models
├── models/                  # Model implementations
│   ├── Transformer_dir/     # Transformer model architecture
│   ├── LSTM.py              # LSTM model architecture
│   └── GBDT.py              # Gradient Boosting Decision Trees
├── stock_data_handle.py     # Data loading and preprocessing
├── utils/                   # Utility functions
│   ├── metrics.py           # Performance metrics calculations
│   └── metrics_object.py    # Metric object implementations
└── log/                     # Logging directory
    └── backtest_results/    # Backtest results and performance metrics
```

## Usage

### Model Training

```python
# Example: Training a Transformer model
from PM_transformer import PM_Transformer
import types

# Create arguments
args = types.SimpleNamespace()
args.model = "transformer"
args.batch_size = 32
args.learning_rate = 0.001
args.train_epochs = 50
args.use_gpu = True
args.num_workers = 4
args.rank_alpha = 0.1
# ...additional parameters

# Load data
from stock_data_handle import Stock_Data
data_all = Stock_Data(...)

# Initialize and train model
model = PM_Transformer(args, data_all)
trained_model = model.train()
```

### Backtesting

```python
# Backtest with top-5 stocks and 10-day holding period
results = model.backtest(topk=5, holding_period=10)
```

## Performance Metrics

The framework calculates and reports the following performance metrics:

- **Strategy Performance**:
  - Sharpe Ratio: Risk-adjusted return
  - Sortino Ratio: Downside risk-adjusted return
  - Maximum Drawdown: Largest peak-to-trough decline
  - Annualized Return: Return annualized to yearly basis
  - Total Return: Cumulative return over the backtest period

- **Benchmark Comparison**:
  - Market returns (equal-weighted portfolio of all stocks)
  - Information Ratio: Excess return per unit of tracking risk

## Acknowledgements

This project builds upon and extends the following frameworks:

- [FinTSB](https://github.com/ZhangXFeng/FinTSB): A Time Series Benchmark for Financial Forecasting
- [StockFormer](https://github.com/gyCSR/StockFormer): A Transformer-based framework for stock prediction

We gratefully acknowledge the contributions of these projects to the financial machine learning community.

## Requirements

- Python 3.7+
- PyTorch 1.7+
- NumPy
- Pandas
- scikit-learn
- XGBoost
- LightGBM

## License

This project is licensed under the MIT License - see the LICENSE file for details. 