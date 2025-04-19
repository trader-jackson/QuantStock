# directory
from __future__ import annotations

from config_tickers import DOW_30_TICKER,SSE_50_TICKER,NAS_100_TICKER

DATA_SAVE_DIR = "datasets"
TRAINED_MODEL_DIR = "trained_models"
TENSORBOARD_LOG_DIR = "tensorboard_log"
RESULTS_DIR = "results"
INF = 1100

# date format: '%Y-%m-%d'
TRAIN_START_DATE = "2010-01-01"  # bug fix: set Monday right, start date set 2014-01-01 ValueError: all the input array dimensions for the concatenation axis must match exactly, but along dimension 0, the array at index 0 has size 1658 and the array at index 1 has size 1657
TRAIN_END_DATE = "2020-12-31"

EVAL_START_DATE = "2021-01-02"
EVAL_END_DATE = "2022-12-31"

TEST_START_DATE = "2023-01-01"
TEST_END_DATE = "2024-12-31"

# stockstats technical indicator column names
# check https://pypi.org/project/stockstats/ for different names
TECHICAL_INDICATORS = [
    "macd",
    "boll_ub",
    "boll_lb",
    "rsi_30",
    "cci_30",
    "dx_30",
    "close_30_sma",
    "close_60_sma",
]
TEMPORAL_FEATURE = [
    'open', 
    'close', 
    'high', 
    'low', 
    'volume', 
]

ADDITIONAL_FEATURE = [
    'label_short_term',
    'label_long_term'
]




# Model Parameters
A2C_PARAMS = {"n_steps": 5, "ent_coef": 0.01, "learning_rate": 0.0007}
PPO_PARAMS = {
    "n_steps": 2048,
    "ent_coef": 0.01,
    "learning_rate": 0.00025,
    "batch_size": 64,
}
DDPG_PARAMS = {"batch_size": 128, "buffer_size": 50000, "learning_rate": 0.001}
TD3_PARAMS = {"batch_size": 100, "buffer_size": 1000000, "learning_rate": 0.001}
SAC_PARAMS = {
    "batch_size": 64,
    "buffer_size": 100000,
    "learning_rate": 0.0001,
    "learning_starts": 100,
    "ent_coef": "auto_0.1",
}
ERL_PARAMS = {
    "learning_rate": 3e-5,
    "batch_size": 2048,
    "gamma": 0.985,
    "seed": 312,
    "net_dimension": 512,
    "target_step": 5000,
    "eval_gap": 30,
    "eval_times": 64,  # bug fix:KeyError: 'eval_times' line 68, in get_model model.eval_times = model_kwargs["eval_times"]
}

NASDAQ_date = ["20110119", "20201230", "20210104", "20231230",  "20220103", "20241227"]
DOW_date = ["20110119", "20201230", "20210104", "20231229",  "20220401", "20241227"]
SSE_date = ["20110119", "20201230", "20210104", "20231229",  "20220401", "20241227"]
""
date_dict = {'NAS': NASDAQ_date, 'DOW': DOW_date,"SSE":SSE_date}

use_ticker_dict={"NAS":NAS_100_TICKER,"DOW":DOW_30_TICKER,"SSE":SSE_50_TICKER}