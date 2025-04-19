import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import logging
import matplotlib.pyplot as plt

from models.Transformer_dir.transformer import Transformer_base as Transformer  # These should be wrapped as PyTorch modules with attribute 'model'
from utils.metrics import ranking_loss
import utils.metrics_object as metrics_object
from .PM import PM

current_dir = os.path.dirname(os.path.abspath(__file__))

sys.path.insert(0, current_dir)

class PM_Transformer(PM):
    def __init__(self, args, data_all):
        super(PM_Transformer, self).__init__(args, data_all)
        
    def _build_model(self):
        model = Transformer(self.args.enc_in, self.args.dec_in, self.args.c_out,
                          self.args.d_model, self.args.n_heads, self.args.e_layers,
                          self.args.d_layers, self.args.d_ff, self.args.dropout,
                          self.args.activation)
        return model.float()
    
    def vali(self, vali_data, vali_loader, criterion, metric_builders, stage='test'):
        self.model.eval()
        total_loss = []
        metric_objs = [builder(stage) for builder in metric_builders]

        for i, (batch_x1, batch_x2, batch_y, _) in enumerate(vali_loader):
            bs, stock_num = batch_x1.shape[0], batch_x1.shape[1]
            batch_x1 = batch_x1.reshape(-1, batch_x1.shape[-2], batch_x1.shape[-1]).float().to(self.device)
            batch_x2 = batch_x2.reshape(-1, batch_x2.shape[-2], batch_x2.shape[-1]).float().to(self.device)
            batch_y = batch_y.float().to(self.device)
            
            _, _, output = self.model(batch_x1, batch_x2)
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
        checkpoint_path = os.path.join('./checkpoints/', "transformer")
        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)
        
        time_now = time.time()
        train_steps = len(train_loader)
        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        metric_objs = [builder("train") for builder in metrics_builders]
        valid_loss_global = np.inf
        best_model_index = -1

        self.logger.info(f"Starting training for {self.args.model} model")
        self.logger.info(f"Training epochs: {self.args.train_epochs}, Steps per epoch: {train_steps}")

        # Create lists to store metrics for each epoch
        epochs = []
        train_rankic_values = []
        valid_rankic_values = []
        test_rankic_values = []
        train_mirrtop1_values = []
        valid_mirrtop1_values = []
        test_mirrtop1_values = []

        self.model.train()
        for epoch in range(self.args.train_epochs):
            epochs.append(epoch + 1)  # Store epoch number (1-indexed)
            iter_count = 0
            train_loss = []
            for i, (batch_x1, batch_x2, batch_y, _) in enumerate(train_loader):
                iter_count += 1
                bs, stock_num = batch_x1.size(0), batch_x1.size(1)
                batch_x1 = batch_x1.contiguous().view(-1, batch_x1.size(-2), batch_x1.size(-1)).float().to(self.device)
                batch_x2 = batch_x2.contiguous().view(-1, batch_x2.size(-2), batch_x2.size(-1)).float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                
                _, _, output = self.model(batch_x1, batch_x2)
                output = output[:, 0]
                output = output.view(bs, stock_num)
                
                loss = criterion(output, batch_y) + self.args.rank_alpha * ranking_loss(output, batch_y)
                train_loss.append(loss.item())
                
                model_optim.zero_grad()
                loss.backward()
                model_optim.step()
                
                if (i + 1) % 100 == 0:
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    self.logger.info(f"Epoch {epoch+1}, Iteration {i+1} | Loss: {loss.item():.7f}")
                    self.logger.info(f"Speed: {speed:.4f}s/iter; Estimated time left: {left_time:.4f}s")
                    iter_count = 0
                    time_now = time.time()
                
                with torch.no_grad():
                    for metric in metric_objs:
                        metric.update(output, batch_y)
            
            train_loss = np.average(train_loss)
            valid_loss, valid_metrics = self.vali(None, vali_loader, criterion, metrics_builders, stage="valid")
            test_loss, test_metrics = self.vali(None, test_loader, criterion, metrics_builders, stage="test")
            
            self.logger.info(f"Epoch {epoch+1}: Train Loss: {train_loss:.7f}, Valid Loss: {valid_loss:.7f}, Test Loss: {test_loss:.7f}")
            
            # Store metrics for this epoch
            train_metrics_dict = {}
            valid_metrics_dict = {}
            test_metrics_dict = {}
            
            # Extract metrics correctly - checking for varying possible names
            for metric in metric_objs:
                if 'mirr' in metric.name.lower() and 'top1' in metric.name.lower():
                    train_metrics_dict['MIRRTop1'] = metric.mean()
                elif 'rank' in metric.name.lower() and 'ic' in metric.name.lower():
                    train_metrics_dict['RankIC'] = np.mean(metric.data) if len(metric.data) > 0 else 0
            
            for metric in valid_metrics:
                if 'mirr' in metric.name.lower() and 'top1' in metric.name.lower():
                    valid_metrics_dict['MIRRTop1'] = metric.mean()
                elif 'rank' in metric.name.lower() and 'ic' in metric.name.lower():
                    valid_metrics_dict['RankIC'] = np.mean(metric.data) if len(metric.data) > 0 else 0
                    
            for metric in test_metrics:
                if 'mirr' in metric.name.lower() and 'top1' in metric.name.lower():
                    test_metrics_dict['MIRRTop1'] = metric.mean()
                elif 'rank' in metric.name.lower() and 'ic' in metric.name.lower():
                    test_metrics_dict['RankIC'] = np.mean(metric.data) if len(metric.data) > 0 else 0
            
            # Store values for plotting with fallbacks to avoid zeros if possible
            train_rankic_values.append(train_metrics_dict.get('RankIC', 0))
            valid_rankic_values.append(valid_metrics_dict.get('RankIC', 0))
            test_rankic_values.append(test_metrics_dict.get('RankIC', 0))
            train_mirrtop1_values.append(train_metrics_dict.get('MIRRTop1', 0))
            valid_mirrtop1_values.append(valid_metrics_dict.get('MIRRTop1', 0))
            test_mirrtop1_values.append(test_metrics_dict.get('MIRRTop1', 0))
            
            all_logs = {metric.name: metric.value for metric in metric_objs + valid_metrics + test_metrics}
            for name, value in all_logs.items():
                self.logger.info(f"{name}: {value.mean()}")
            
            cp_path = os.path.join(checkpoint_path, f"checkpoint_{epoch+1}.pth")
            torch.save(self.model.state_dict(), cp_path)
            self.logger.info(f"Model checkpoint saved: {cp_path}")
            
            if valid_loss < valid_loss_global:
                valid_loss_global = valid_loss
                best_model_index = epoch + 1
                self.logger.info(f"New best model at epoch {epoch+1} with validation loss: {valid_loss:.7f}")
        
        # Create plots directory
        plots_dir = os.path.join('log', 'plots', self.args.model)
        os.makedirs(plots_dir, exist_ok=True)
        
        # Plot RankIC metrics
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, train_rankic_values, label='Train RankIC', marker='o')
        plt.plot(epochs, valid_rankic_values, label='Valid RankIC', marker='s')
        plt.plot(epochs, test_rankic_values, label='Test RankIC', marker='^')
        plt.xlabel('Epoch')
        plt.ylabel('RankIC')
        plt.title('RankIC Metrics by Epoch')
        plt.legend()
        plt.grid(True)
        rankic_plot_path = os.path.join(plots_dir, 'rankic_metrics.png')
        plt.savefig(rankic_plot_path)
        plt.close()
        
        # Plot MIRRTop1 metrics
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, train_mirrtop1_values, label='Train MIRRTop1', marker='o')
        plt.plot(epochs, valid_mirrtop1_values, label='Valid MIRRTop1', marker='s')
        plt.plot(epochs, test_mirrtop1_values, label='Test MIRRTop1', marker='^')
        plt.xlabel('Epoch')
        plt.ylabel('MIRRTop1')
        plt.title('MIRRTop1 Metrics by Epoch')
        plt.legend()
        plt.grid(True)
        mirrtop1_plot_path = os.path.join(plots_dir, 'mirrtop1_metrics.png')
        plt.savefig(mirrtop1_plot_path)
        plt.close()
        
        self.logger.info(f"Metric plots saved to {plots_dir}")

        best_model_path = os.path.join(checkpoint_path, f"checkpoint_{best_model_index}.pth")
        self.model.load_state_dict(torch.load(best_model_path))
        self.logger.info(f"Training completed. Best model from epoch {best_model_index} loaded.")
        return self.model


    def backtest(self, topk: int = 5, holding_period: int = 5):
        """
        Run an out‑of‑sample back‑test.

        • Every `holding_period` days the portfolio is re‑balanced to the
        model's top‑k stocks.
        • Between re‑balances the holdings are kept constant.
        • Daily portfolio return = mean of held‑stock returns that day.
        • Compared against a market average benchmark (equal weight all stocks).
        """
        test_dataset, _ = self._get_data(flag="test")
        self.model.eval()

        daily_returns  = []
        market_returns = []
        current_holdings: np.ndarray | None = None
        days_held = 0

        # histories (if you still want to save them)
        holdings_history = []
        weights_history  = []
        dates_history    = []

        for day_idx in range(len(test_dataset)):
            seq_x1, seq_x2, seq_y, _ = test_dataset[day_idx]
            market_ret = float(np.mean(seq_y))
            market_returns.append(market_ret)

            # Re‑balance condition
            if days_held == 0 or days_held >= holding_period:
                with torch.no_grad():
                    # mirror your train reshaping exactly
                    x1 = (
                        torch.from_numpy(seq_x1)
                            .float()
                            .to(self.device)
                            .view(-1, seq_x1.shape[-2], seq_x1.shape[-1])
                    )
                    x2 = (
                        torch.from_numpy(seq_x2)
                            .float()
                            .to(self.device)
                            .view(-1, seq_x2.shape[-2], seq_x2.shape[-1])
                    )
                    _, _, preds = self.model(x1, x2)
                    # preds has shape [stock_num, seq_len, 1], so:
                    pred = preds[:, 0].cpu().numpy().reshape(-1)  # -> (stock_num,)

                # select top‑k
                avail = len(pred)
                k = min(topk, avail)
                current_holdings = np.argsort(pred)[-k:]
                days_held = 1
            else:
                days_held += 1

            # compute portfolio return
            if current_holdings is not None and current_holdings.size > 0:
                port_ret = float(np.mean(seq_y[current_holdings]))
            else:
                port_ret = 0.0

            daily_returns.append(port_ret)
            dates_history.append(day_idx)
            holdings_history.append(current_holdings.copy() if current_holdings is not None else np.array([]))

            # equal‑weight
            if current_holdings is not None and current_holdings.size > 0:
                w = np.zeros_like(seq_y)
                w[current_holdings] = 1.0 / len(current_holdings)
                weights_history.append(w)
            else:
                weights_history.append(np.array([]))

        # ─── Performance metrics ───────────────────────────────────
        daily_returns  = np.array(daily_returns)
        market_returns = np.array(market_returns)
        days = len(daily_returns)
        ann = np.sqrt(252)

        # Strategy
        sharpe  = daily_returns.mean() / (daily_returns.std() + 1e-9) * ann
        downside = daily_returns[daily_returns < 0]
        sortino = daily_returns.mean() / (downside.std() + 1e-9) * ann
        cum = np.cumprod(1 + daily_returns)
        peak = np.maximum.accumulate(cum)
        max_dd = np.min((cum - peak) / peak)
        ann_ret = cum[-1]**(252/days) - 1

        # Benchmark
        m_sharpe  = market_returns.mean() / (market_returns.std() + 1e-9) * ann
        m_down = market_returns[market_returns < 0]
        m_sortino = market_returns.mean() / (m_down.std() + 1e-9) * ann
        m_cum = np.cumprod(1 + market_returns)
        m_peak = np.maximum.accumulate(m_cum)
        m_dd = np.min((m_cum - m_peak) / m_peak)
        m_ann_ret = m_cum[-1]**(252/days) - 1

        # Information ratio
        ex_ret = daily_returns - market_returns
        ir = ex_ret.mean() / (ex_ret.std() + 1e-9) * ann

        # Logging
        self.logger.info("\n── Back‑test ─────────────────────────────────────────")
        self.logger.info(f"top‑{topk} | hold {holding_period}‑d")
        self.logger.info(f"Sharpe       : {sharpe:.4f}")
        self.logger.info(f"Sortino      : {sortino:.4f}")
        self.logger.info(f"Max DD       : {max_dd:.4f}")
        self.logger.info(f"Ann. Return  : {ann_ret:.4f}")
        self.logger.info(f"Total Return : {cum[-1]-1:.4f}")
        self.logger.info("\nBenchmark (Market Avg):")
        self.logger.info(f"Sharpe       : {m_sharpe:.4f}")
        self.logger.info(f"Sortino      : {m_sortino:.4f}")
        self.logger.info(f"Max DD       : {m_dd:.4f}")
        self.logger.info(f"Ann. Return  : {m_ann_ret:.4f}")
        self.logger.info(f"Total Return : {m_cum[-1]-1:.4f}")
        self.logger.info(f"\nInformation Ratio: {ir:.4f}")
        self.logger.info(f"Trading days: {days}")

        # Save to CSV
        import pandas as pd
        df = pd.DataFrame({
            'day': dates_history,
            'strategy_return': daily_returns,
            'market_return': market_returns,
            'strategy_cumulative': cum,
            'market_cumulative': m_cum,
            'excess_return': ex_ret
        })
        outdir = os.path.join('log', 'backtest_results', self.args.model)
        os.makedirs(outdir, exist_ok=True)
        path = os.path.join(outdir, f'perf_top{topk}_hold{holding_period}.csv')
        df.to_csv(path, index=False)
        self.logger.info(f"Saved performance to {path}")