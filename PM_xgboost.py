from PM import PM
import torch
import torch.nn as nn
import numpy as np
from models.GBDT import XGBoost
import os
import time
import logging
from sklearn.metrics import mean_squared_error

class PM_XGBoost(PM):
    def __init__(self, args, data_all, identifier):
        super(PM_XGBoost, self).__init__(args, data_all, identifier)
        
    def _build_model(self):
        model = XGBoost(self.args)
        return model
    
    def train(self):
        dataset_train, _ = self._get_data(flag="train")
        X_train_list, y_train_list = [], []
        for i in range(len(dataset_train)):
            seq_x, _, seq_y, _ = dataset_train[i]
            for s in range(seq_x.shape[0]):
                X_train_list.append(seq_x[s].reshape(-1))
                y_train_list.append(seq_y[s])
        X_train = np.stack(X_train_list, axis=0)
        y_train = np.array(y_train_list)

        dataset_valid, _ = self._get_data(flag="valid")
        X_valid_list, y_valid_list = [], []
        for i in range(len(dataset_valid)):
            seq_x, _, seq_y, _ = dataset_valid[i]
            for s in range(seq_x.shape[0]):
                X_valid_list.append(seq_x[s].reshape(-1))
                y_valid_list.append(seq_y[s])
        X_valid = np.stack(X_valid_list, axis=0)
        y_valid = np.array(y_valid_list)

        early_stopping_rounds = self.args.early_stopping if hasattr(self.args, "early_stopping") else 10
        print("Training XGBoost model with early stopping...")
        self.model.model.fit(X_train, y_train,
                           eval_set=[(X_valid, y_valid)],
                           early_stopping_rounds=early_stopping_rounds,
                           verbose=True)
        
        valid_loss = mean_squared_error(y_valid, self.model.model.predict(X_valid))
        print("XGBoost model validation MSE:", valid_loss)
        return self.model

    def test(self):
        dataset_test, _ = self._get_data(flag="test")
        X_test_list, y_test_list = [], []
        for i in range(len(dataset_test)):
            seq_x, _, seq_y, _ = dataset_test[i]
            for s in range(seq_x.shape[0]):
                X_test_list.append(seq_x[s].reshape(-1))
                y_test_list.append(seq_y[s])
        X_test = np.stack(X_test_list, axis=0)
        y_test = np.array(y_test_list)
        
        y_pred = self.model.model.predict(X_test)
        test_loss = mean_squared_error(y_test, y_pred)
        print("XGBoost model test MSE:", test_loss)
        return

    def backtest(self, topk: int = 5, holding_period: int = 5):
        """
        Run an out‑of‑sample back‑test.

        • Every *holding_period* days the portfolio is re‑balanced to the
          model's top‑k stocks.
        • Between re‑balances the holdings are kept constant.
        • Daily portfolio return = mean of held‑stock returns that day.
        • Compared against a market average benchmark (equal weight all stocks).
        """
        test_dataset, _ = self._get_data(flag="test")
        self.model.eval()

        daily_returns   = []
        market_returns  = []  # Market average benchmark (equal-weight all stocks)
        current_holdings: np.ndarray | None = None
        days_held       = 0               # 1‑based counter AFTER re‑balance

        for i in range(len(test_dataset)):
            seq_x, _, seq_y, _ = test_dataset[i]          # seq_y MUST be daily returns

            # Calculate market average return (equal weight all stocks)
            market_return = np.mean(seq_y)
            market_returns.append(market_return)

            # ─── Re‑balance? ────────────────────────────────────────
            if days_held == 0 or days_held >= holding_period:
                # ─ inference ───────────────────────────────────────
                if self.args.model.lower() in {"transformer", "lstm"}:
                    with torch.no_grad():
                        x_t = torch.from_numpy(seq_x).float().to(self.device)
                        x_t = x_t.reshape(-1, x_t.shape[-2], x_t.shape[-1])

                        if self.args.model.lower() == "transformer":
                            _, _, pred_t = self.model(x_t, x_t)
                        else:  # LSTM
                            pred_t = self.model(x_t)

                    pred = pred_t.squeeze().flatten().cpu().numpy()  # 1‑D
                else:  # tree‑based
                    pred = self.model.model.predict(seq_x.reshape(seq_x.shape[0], -1))

                # Make sure we don't select more stocks than available
                available_stocks = min(len(seq_y), len(pred))
                topk_effective = min(topk, available_stocks)
                
                
                # pick top‑k indices but ensure they're bounded by array size
                current_holdings = np.argsort(pred[:available_stocks])[-topk_effective:]
                days_held = 1      # start counting days *after* rebalance

                 

            # ─── Portfolio return for today ────────────────────────
            if current_holdings is not None and len(current_holdings) > 0:
                portfolio_ret = np.mean(seq_y[current_holdings])
            
            
            daily_returns.append(portfolio_ret)
            days_held += 1         # increment until next rotation

        # ─── Performance metrics ───────────────────────────────────
        daily_returns = np.asarray(daily_returns)
        market_returns = np.asarray(market_returns)
        trading_days  = len(daily_returns)
        ann_factor    = np.sqrt(252)

        # Strategy metrics
        sharpe  = np.mean(daily_returns) / (np.std(daily_returns) + 1e-9) * ann_factor
        downside = daily_returns[daily_returns < 0]
        sortino = np.mean(daily_returns) / (np.std(downside) + 1e-9) * ann_factor

        cum_ret  = np.cumprod(1 + daily_returns)
        peak     = np.maximum.accumulate(cum_ret)
        max_dd   = np.min((cum_ret - peak) / peak)
        ann_ret  = cum_ret[-1] ** (252 / trading_days) - 1

        # Benchmark metrics
        market_sharpe = np.mean(market_returns) / (np.std(market_returns) + 1e-9) * ann_factor
        market_downside = market_returns[market_returns < 0]
        market_sortino = np.mean(market_returns) / (np.std(market_downside) + 1e-9) * ann_factor

        market_cum_ret = np.cumprod(1 + market_returns)
        market_peak = np.maximum.accumulate(market_cum_ret)
        market_max_dd = np.min((market_cum_ret - market_peak) / market_peak)
        market_ann_ret = market_cum_ret[-1] ** (252 / trading_days) - 1

        # Information ratio
        excess_returns = daily_returns - market_returns
        information_ratio = np.mean(excess_returns) / (np.std(excess_returns) + 1e-9) * ann_factor

        print("\n── Back‑test ─────────────────────────────────────────")
        print(f"top‑{topk} | hold {holding_period}‑d")
        print("\nStrategy Performance:")
        print(f"Sharpe  : {sharpe:8.4f}")
        print(f"Sortino : {sortino:8.4f}")
        print(f"Max DD  : {max_dd:8.4f}")
        print(f"Ann.Return: {ann_ret:8.4f}")
        print(f"Total Return: {cum_ret[-1] - 1:8.4f}")
        
        print("\nBenchmark (Market Average):")
        print(f"Sharpe  : {market_sharpe:8.4f}")
        print(f"Sortino : {market_sortino:8.4f}")
        print(f"Max DD  : {market_max_dd:8.4f}")
        print(f"Ann.Return: {market_ann_ret:8.4f}")
        print(f"Total Return: {market_cum_ret[-1] - 1:8.4f}")
        
        print(f"\nInformation Ratio: {information_ratio:8.4f}")
        print(f"Trading days: {trading_days}")

        return dict(
            # Strategy metrics
            sharpe=sharpe,
            sortino=sortino,
            max_drawdown=max_dd,
            annual_return=ann_ret,
            total_return=cum_ret[-1] - 1,
            daily_returns=daily_returns,
            cumulative_returns=cum_ret,
            
            # Benchmark metrics
            market_sharpe=market_sharpe,
            market_sortino=market_sortino,
            market_max_drawdown=market_max_dd,
            market_annual_return=market_ann_ret,
            market_total_return=market_cum_ret[-1] - 1,
            market_daily_returns=market_returns,
            market_cumulative_returns=market_cum_ret,
            
            # Relative metrics
            information_ratio=information_ratio,
        )