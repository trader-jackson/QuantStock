# Stock Prediction Methods

A comprehensive framework for stock return prediction and portfolio backtesting.

## Overview

This repository implements various predictive models for stock returns, including Transformer-based architectures, LSTM networks, graph-based models (GAT, GCN), sequence models (GRU, TCN, Mamba), and tree-based models such as XGBoost and LightGBM. The framework provides a unified interface for training, validating, and backtesting these models on financial time series data with detailed performance metrics and visualization capabilities.

## Key Features

- **Multiple Model Architectures**: 
  - Transformer-based models
  - Recurrent Neural Networks (LSTM, GRU)
  - Graph Neural Networks (GAT, GCN)
  - Temporal Convolutional Networks (TCN)
  - State Space Models (Mamba)
  - Tree-based models (XGBoost, LightGBM)
  - Diffusion models
- **End-to-End Pipeline**: From data preprocessing to model training and evaluation
- **Multiple Datasets**: Support for DOW30, NASDAQ100, and SSE50 stock indices
- **Sophisticated Backtesting**: Implements a trading strategy with configurable parameters:
  - Top-K stock selection based on model predictions (1 to N stocks)
  - Customizable holding period (1 to N days)
  - Performance comparison against market benchmark
  - Detailed performance visualization
- **Comprehensive Metrics**: 
  - Sharpe ratio: Risk-adjusted return measure
  - Sortino ratio: Downside risk-adjusted return
  - Information ratio: Excess return per unit of tracking risk
  - Maximum drawdown: Largest peak-to-trough decline
  - Annualized return: Return normalized to yearly basis
  - Total return: Cumulative return over the backtest period
- **Detailed Logging**: Model-specific logging with performance tracking
- **Results Export**: Save performance metrics, returns, and portfolio holdings to CSV and pickle files

## Project Structure

```
├── pm/                     # Prediction model implementations
│   ├── PM.py               # Base class for prediction models
│   ├── PM_transformer.py   # Transformer-based prediction model
│   ├── PM_lstm.py          # LSTM-based prediction model
│   ├── PM_gru.py           # GRU-based prediction model
│   ├── PM_gcn.py           # GCN-based prediction model
│   ├── PM_diffusion.py     # Diffusion-based prediction model
│   ├── PM_xgboost.py       # XGBoost-based prediction model
│   └── PM_lightgbm.py      # LightGBM-based prediction model
├── models/                 # Model architectures
│   ├── Transformer_dir/    # Transformer model architecture
│   ├── LSTM.py             # LSTM model architecture
│   ├── GRU.py              # GRU model architecture
│   ├── GAT.py              # Graph Attention Network architecture
│   ├── GCN.py              # Graph Convolutional Network architecture
│   ├── TCN.py              # Temporal Convolutional Network architecture
│   ├── Mamba.py            # Mamba state space model architecture
│   ├── GBDT.py             # Gradient Boosting Decision Trees
│   └── diffusion_stock.py  # Diffusion model for stock prediction
├── model_config/           # Configuration files for different models
├── FinTSB/                 # Financial Time Series Benchmark configurations
│   └── configs/            # YAML configuration files for models
├── data_dir/               # Dataset directory
│   ├── DOW30/              # Dow Jones 30 dataset
│   ├── NASQ100/            # NASDAQ 100 dataset
│   └── SSE50/              # Shanghai Stock Exchange 50 dataset
├── stock_data_handle.py    # Data loading and preprocessing
├── utils/                  # Utility functions
│   ├── metrics.py          # Performance metrics calculations
│   ├── metrics_object.py   # Metric object implementations
│   ├── masking.py          # Masking utilities
│   ├── tools.py            # General utility tools
│   └── preprocess.py       # Data preprocessing utilities
├── train.py                # Training script for individual models
├── train_all_models.py     # Script to train all models
└── log/                    # Logging directory
    └── backtest_results/   # Backtest results and performance metrics
```

## Usage

### Model Configuration

Each model type has a corresponding JSON configuration file in the `model_config` directory. These files define all parameters needed for model training and evaluation.

```json
// Example: transformer_config.json
{
    "model": "Transformer",
    "project_name": "DOW30",
    "root_path": "data_dir",
    "data_dict": {
        "DOW30": {
            "dataset_name": "DOW",
            "full_stock_path": "DOW30"
        },
        "SSE30": {
            "dataset_name": "SSE",
            "full_stock_path": "SSE50"
        },
        "NASDAQ100": {
            "dataset_name": "NAS",
            "full_stock_path": "NASQ100"
        }
    },
    "seq_len": 60,
    "prediction_len": 5,
    "rank_alpha": 1,
    "batch_size": 32,
    "num_workers": 1,
    "learning_rate": 3e-4,
    "train_epochs": 50,
    "use_multi_gpu": false,
    "use_gpu": true,
    "enc_in": 13,
    "dec_in": 13,
    "c_out": 1,
    "d_model": 256,
    "n_heads": 4,
    "e_layers": 2,
    "d_layers": 1,
    "d_ff": 512,
    "dropout": 0.3,
    "activation": "gelu"
}
```

### Model Training

```python
# Example: Training a Transformer model
from pm.PM_transformer import PM_Transformer
import types
import json
from utils.tools import dict_to_namespace

# Load configuration from JSON file
with open("model_config/transformer_config.json", "r") as f:
    config_dict = json.load(f)

# Convert to namespace for easier access
args = dict_to_namespace(config_dict)

# Set device
import torch
device = torch.device("cuda:0" if args.use_gpu else "cpu")
args.device = device

# Load data
from stock_data_handle import Stock_Data
stock_data = Stock_Data(
    dataset_name=args.data_dict[args.project_name]["dataset_name"],
    full_stock_path=args.data_dict[args.project_name]["full_stock_path"],
    window_size=args.seq_len,
    root_path=args.root_path,
    prediction_len=args.prediction_len,
    scale=True
)

# Initialize and train model
model = PM_Transformer(args, stock_data)
trained_model = model.train()
```

### Training All Models

The repository provides a script to train all available models:

```bash
python train_all_models.py --project_name DOW30
```

### Backtesting

The framework provides a comprehensive backtesting functionality that allows you to evaluate model performance with different parameters:

```python
# Backtest with top-5 stocks and 10-day holding period
results = model.backtest(topk=5, holding_period=10)

# Try different configurations
results_top1 = model.backtest(topk=1, holding_period=5)  # More concentrated portfolio
results_top10 = model.backtest(topk=10, holding_period=5)  # More diversified portfolio
results_longer = model.backtest(topk=5, holding_period=20)  # Longer holding period
```

Backtesting results include detailed performance metrics that are automatically logged and saved to CSV files in the `log/backtest_results/` directory.

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

## Backtesting Results

Backtesting results are automatically saved to the `log/backtest_results/` directory with the following structure:

```
log/
└── backtest_results/
    ├── Transformer/
    │   ├── perf_top1_hold5.csv   # Performance metrics for top-1 strategy with 5-day holding
    │   ├── perf_top10_hold5.csv  # Performance metrics for top-10 strategy with 5-day holding
    │   └── ... 
    ├── LSTM/
    │   ├── perf_top5_hold3.csv   # Performance metrics for top-5 strategy with 3-day holding
    │   └── ...
    └── ...
```

Each CSV file contains detailed performance metrics including daily returns, cumulative returns, drawdowns, and benchmark comparisons. The framework also generates performance plots that visualize the strategy's performance against the benchmark.

## Model-Specific Configurations

The framework supports various model architectures, each with its own configuration parameters:

### Transformer
```json
{
    "model": "Transformer",
    "seq_len": 60,           // Input sequence length
    "prediction_len": 5,     // Prediction horizon
    "rank_alpha": 1,         // Weight for rank loss
    "enc_in": 13,           // Input dimension
    "dec_in": 13,           // Decoder input dimension
    "c_out": 1,             // Output dimension
    "d_model": 256,         // Model dimension
    "n_heads": 4,           // Number of attention heads
    "e_layers": 2,          // Number of encoder layers
    "d_layers": 1,          // Number of decoder layers
    "d_ff": 512,            // Dimension of feed-forward network
    "dropout": 0.3,         // Dropout rate
    "activation": "gelu"    // Activation function
}
```

### LSTM/ALSTM
```json
{
    "model": "LSTM",
    "seq_len": 60,           // Input sequence length
    "prediction_len": 1,     // Prediction horizon
    "rank_alpha": 1,         // Weight for rank loss
    "model_config": {
        "input_size": 5,      // Input feature dimension
        "hidden_size": 128,   // Hidden state dimension
        "num_layers": 2,      // Number of LSTM layers
        "dropout": 0.05,      // Dropout rate
        "bidirectional": true, // Whether to use bidirectional LSTM
        "attention": true,    // Whether to use attention mechanism
        "noise_level": 0.0,   // Noise level for regularization
        "d_ff": 256,          // Dimension of feed-forward network
        "c_out": 1            // Output dimension
    }
}
```

### XGBoost
```json
{
    "model": "XGBoost",
    "seq_len": 60,           // Input sequence length
    "prediction_len": 1,     // Prediction horizon
    "rank_alpha": 1,         // Weight for rank loss
    "model_config": {
        "max_depth": 6,        // Maximum tree depth
        "min_child_weight": 1, // Minimum sum of instance weight needed in a child
        "subsample": 0.8,      // Subsample ratio of training instances
        "colsample_bytree": 0.8, // Subsample ratio of columns for each tree
        "gamma": 0,            // Minimum loss reduction required for a split
        "reg_alpha": 0,        // L1 regularization term
        "reg_lambda": 1,       // L2 regularization term
        "learning_rate": 0.1,   // Learning rate
        "n_estimators": 100,   // Number of boosting rounds
        "early_stopping_rounds": 10 // Early stopping rounds
    }
}
```

### LightGBM
```json
{
    "model": "LightGBM",
    "seq_len": 60,           // Input sequence length
    "prediction_len": 1,     // Prediction horizon
    "rank_alpha": 1,         // Weight for rank loss
    "model_config": {
        "num_leaves": 31,      // Number of leaves in one tree
        "max_depth": -1,       // Maximum tree depth (-1 means no limit)
        "learning_rate": 0.1,   // Learning rate
        "n_estimators": 100,   // Number of boosting rounds
        "subsample": 0.8,      // Subsample ratio of training instances
        "colsample_bytree": 0.8, // Subsample ratio of columns for each tree
        "reg_alpha": 0.0,      // L1 regularization term
        "reg_lambda": 1.0,     // L2 regularization term
        "min_child_samples": 20, // Minimum number of data needed in a leaf
        "early_stopping_rounds": 10 // Early stopping rounds
    }
}
```

## Datasets

The framework includes support for multiple stock market indices:

- **DOW30**: Dow Jones Industrial Average (30 stocks)
- **NASDAQ100**: NASDAQ 100 Index (100 stocks)
- **SSE50**: Shanghai Stock Exchange 50 Index (50 stocks)

Each dataset is pre-processed and split into training, validation, and test sets located in the `data_dir` directory.

## FinTSB Integration

The framework integrates with the Financial Time Series Benchmark (FinTSB) through YAML configuration files located in the `FinTSB/configs/` directory. These files provide alternative configuration options for models like GAT, GCN, and diffusion models.

```yaml
# Example: config_gat.yaml
model_config:
  seq_len: 20
  pred_len: 1
  e_layers: 2
  factor: 3
  enc_in: 5
  c_out: 1
  d_model: 64
  d_ff: 64
  dropout: 0.1
  base_model: LSTM

task:
  model:
    class: QniverseModel
    module_path: src/model_backbone.py
    kwargs:
      lr: 0.0001
      n_epochs: 3
      max_steps_per_epoch: 100
      early_stop: 3
      seed: 2025
      logdir: output/gat
      model_type: GAT
```

## Interpreting Backtesting Results

The backtesting results provide a comprehensive comparison between the model-based trading strategy and a market benchmark (equal-weighted portfolio):

```
── Back‑test ─────────────────────────────────────────
top‑10 | hold 5‑d
Sharpe       : 0.2468
Sortino      : 0.3629
Max DD       : -0.2725
Ann. Return  : 0.0280
Total Return : 0.0858

Benchmark (Market Avg):
Sharpe       : 0.4685
Sortino      : 0.6832
Max DD       : -0.2156
Ann. Return  : 0.0599
Total Return : 0.1892

Information Ratio: -0.4147
Trading days: 751
```

Key points for interpretation:
- **Sharpe Ratio**: Higher is better, indicates return per unit of risk
- **Sortino Ratio**: Higher is better, focuses on downside risk
- **Max Drawdown**: Smaller negative value is better, shows worst peak-to-trough decline
- **Information Ratio**: Measures excess return vs. tracking error against benchmark
- **Trading Days**: Number of days in the backtest period

A positive Information Ratio indicates the strategy outperforms the benchmark on a risk-adjusted basis.

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
- Mamba-ssm (for Mamba models)

## License

This project is licensed under the MIT License - see the LICENSE file for details.