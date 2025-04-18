from PM import PM
import torch
import torch.nn as nn
import numpy as np
from models.LSTM import LSTM
import os
import time
import logging

class PM_LSTM(PM):
    def __init__(self, args, data_all, identifier):
        super(PM_LSTM, self).__init__(args, data_all, identifier)
        
    def _build_model(self):
        model = LSTM(self.args)
        return model.float()
    
    def vali(self, vali_data, vali_loader, criterion, metric_builders, stage='test'):
        self.model.eval()
        total_loss = []
        metric_objs = [builder(stage) for builder in metric_builders]

        for i, (batch_x1, batch_x2, batch_y, _) in enumerate(vali_loader):
            bs, stock_num = batch_x1.shape[0], batch_x1.shape[1]
            batch_x1 = batch_x1.reshape(-1, batch_x1.shape[-2], batch_x1.shape[-1]).float().to(self.device)
            batch_y = batch_y.float().to(self.device)
            
            output = self.model(batch_x1)
            output = output.reshape(bs, stock_num)
            loss = criterion(output, batch_y) + self.args.rank_alpha * ranking_loss(output, batch_y)
            total_loss.append(loss.item())

            with torch.no_grad():
                for metric in metric_objs:
                    metric.update(output, batch_y)

        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss, metric_objs

    def train(self):
        _, train_loader = self._get_data(flag="train")
        _, vali_loader = self._get_data(flag="valid")
        _, test_loader = self._get_data(flag="test")
        metrics_builders = [metrics_object.MIRRTop1, metrics_object.RankIC]
        checkpoint_path = os.path.join('./checkpoints/', "")
        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)
        
        time_now = time.time()
        train_steps = len(train_loader)
        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        metric_objs = [builder("train") for builder in metrics_builders]
        valid_loss_global = np.inf
        best_model_index = -1

        self.model.train()
        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            for i, (batch_x1, batch_x2, batch_y, _) in enumerate(train_loader):
                iter_count += 1
                bs, stock_num = batch_x1.size(0), batch_x1.size(1)
                batch_x1 = batch_x1.contiguous().view(-1, batch_x1.size(-2), batch_x1.size(-1)).float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                
                output = self.model(batch_x1)
                output = output.view(bs, stock_num)
                
                loss = criterion(output, batch_y) + self.args.rank_alpha * ranking_loss(output, batch_y)
                train_loss.append(loss.item())
                
                model_optim.zero_grad()
                loss.backward()
                model_optim.step()
                
                if (i + 1) % 100 == 0:
                    print(f"\titers: {i+1}, epoch: {epoch+1} | loss: {loss.item():.7f}")
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print(f"\tspeed: {speed:.4f}s/iter; left time: {left_time:.4f}s")
                    iter_count = 0
                    time_now = time.time()
                    logging.info(f"Epoch {epoch+1}, Iteration {i+1} | Loss: {loss.item():.7f}")
                
                with torch.no_grad():
                    for metric in metric_objs:
                        metric.update(output, batch_y)
            
            train_loss = np.average(train_loss)
            valid_loss, valid_metrics = self.vali(None, vali_loader, criterion, metrics_builders, stage="valid")
            test_loss, test_metrics = self.vali(None, test_loader, criterion, metrics_builders, stage="test")
            
            logging.info(f"Epoch {epoch+1}: Train Loss: {train_loss:.7f}, Valid Loss: {valid_loss:.7f}, Test Loss: {test_loss:.7f}")
            all_logs = {metric.name: metric.value for metric in metric_objs + valid_metrics + test_metrics}
            for name, value in all_logs.items():
                logging.info(f"{name}: {value.mean()}")
            
            print(f"Epoch: {epoch+1}, Steps: {train_steps} | Train Loss: {train_loss:.7f} Valid Loss: {valid_loss:.7f} Test Loss: {test_loss:.7f}")
            
            cp_path = os.path.join(checkpoint_path, f"checkpoint_{epoch+1}.pth")
            torch.save(self.model.state_dict(), cp_path)
            
            if valid_loss < valid_loss_global:
                valid_loss_global = valid_loss
                best_model_index = epoch + 1
        
        best_model_path = os.path.join(checkpoint_path, f"checkpoint_{best_model_index}.pth")
        self.model.load_state_dict(torch.load(best_model_path))
        print("Best model index:", best_model_index)
        return self.model

    def test(self):
        _, test_loader = self._get_data(flag="test")
        self.model.eval()
        metrics_builders = [metrics_object.MIRRTop1, metrics_object.RankIC]
        metric_objs = [builder("test") for builder in metrics_builders]
        
        for i, (batch_x1, batch_x2, batch_y, _) in enumerate(test_loader):
            bs, stock_num = batch_x1.size(0), batch_x1.size(1)
            batch_x1 = batch_x1.contiguous().view(-1, batch_x1.size(-2), batch_x1.size(-1)).float().to(self.device)
            batch_y = batch_y.float().to(self.device)
            
            output = self.model(batch_x1)
            output = output.view(bs, stock_num)
            
            with torch.no_grad():
                for metric in metric_objs:
                    metric.update(output, batch_y)
        
        folder_path = os.path.join("./results/", "")
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        all_logs = {metric.name: metric.value for metric in metric_objs}
        for name, value in all_logs.items():
            print(name, value.mean())
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