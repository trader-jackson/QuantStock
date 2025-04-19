import os
import time
import sys
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from types import SimpleNamespace
import matplotlib.pyplot as plt

from models.LSTM import LSTM                     # LSTM model definition
from utils.metrics import ranking_loss
import utils.metrics_object as metrics_object
from .PM import PM

current_dir = os.path.dirname(os.path.abspath(__file__))

sys.path.insert(0, current_dir)

class PM_LSTM(PM):
    """
    Portfolio management using an LSTM-based prediction model.
    Uses `model_config` dict from args for LSTM settings.
    """
    def __init__(self, args, data_all):
        super(PM_LSTM, self).__init__(args, data_all)

    def _build_model(self):
        mc = self.args.model_config
        cfg = SimpleNamespace(
            enc_in=mc.get('input_size'),
            c_out=mc.get('c_out'),
            d_model=mc.get('hidden_size'),
            e_layers=mc.get('num_layers'),
            dropout=mc.get('dropout'),
            use_attn=mc.get('attention'),
            noise_level=mc.get('noise_level'),
            d_ff=mc.get('d_ff'),
            input_drop=mc.get('input_drop')
        )
        model = LSTM(cfg)
        return model.float()

    def vali(self, _, loader, criterion, builders, stage='test'):
        self.model.eval()
        losses = []
        metrics = [b(stage) for b in builders]
        for batch_x1, _, batch_y, _ in loader:
            bs, stocks, seq_len, feat = batch_x1.shape
            x = batch_x1.reshape(-1, seq_len, feat).float().to(self.device)
            y = batch_y.float().to(self.device)
            preds = self.model(x).view(bs, stocks)
            loss = criterion(preds, y) + self.args.rank_alpha * ranking_loss(preds, y)
            losses.append(loss.item())
            with torch.no_grad():
                for m in metrics:
                    m.update(preds, y)
        avg = np.mean(losses)
        self.model.train()
        return avg, metrics

    def train(self):
        _, train_loader = self._get_data(flag='train')
        _, val_loader = self._get_data(flag='valid')
        _, test_loader = self._get_data(flag='test')
        metrics_builders = [metrics_object.MIRRTop1, metrics_object.RankIC]
        ckpt_dir = os.path.join('checkpoints', f'{self.args.model}')
        os.makedirs(ckpt_dir, exist_ok=True)

        self.logger.info(f"LSTM training start: epochs={self.args.train_epochs}, batch_size={self.args.batch_size}")
        time0 = time.time()
        steps = len(train_loader)
        optimizer = self._select_optimizer()
        criterion = self._select_criterion()
        metric_objs = [builder("train") for builder in metrics_builders]
        best_val = float('inf'); best_epoch = 0
        
        # Create lists to store metrics for each epoch
        epochs = []
        train_rankic_values = []
        valid_rankic_values = []
        test_rankic_values = []
        train_mirrtop1_values = []
        valid_mirrtop1_values = []
        test_mirrtop1_values = []

        self.model.train()
        for epoch in range(1, self.args.train_epochs+1):
            epochs.append(epoch)
            epoch_losses = []
            cnt = 0
            for i, (bx, _, by, _) in enumerate(train_loader, 1):
                cnt += 1
                bs, stocks, seq_len, feat = bx.shape
                x = bx.reshape(-1, seq_len, feat).float().to(self.device)
                y = by.float().to(self.device)
                preds = self.model(x).view(bs, stocks)
                loss = criterion(preds, y) + self.args.rank_alpha * ranking_loss(preds, y)
                optimizer.zero_grad(); loss.backward(); optimizer.step()
                epoch_losses.append(loss.item())
                if i % 100 == 0:
                    elapsed = time.time() - time0
                    speed = elapsed / cnt; remain = speed * (self.args.train_epochs*steps - epoch*steps - i)
                    self.logger.info(f"Epoch {epoch}/{self.args.train_epochs}, Step {i}/{steps}, Loss {loss.item():.6f}, ETA {remain:.1f}s")
                    time0, cnt = time.time(), 0
                with torch.no_grad():
                    for m in metric_objs: m.update(preds, y)
            train_loss = np.mean(epoch_losses)
            val_loss, val_metrics = self.vali(None, val_loader, criterion, metrics_builders, 'valid')
            test_loss, test_metrics = self.vali(None, test_loader, criterion, metrics_builders, 'test')
            self.logger.info(f"Epoch {epoch}: Train {train_loss:.6f}, Valid {val_loss:.6f}, Test {test_loss:.6f}")

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
                    
            
            for metric in val_metrics:
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
            
            all_logs = {metric.name: metric.value for metric in metric_objs + val_metrics + test_metrics}
            for name, value in all_logs.items():
                self.logger.info(f"{name}: {value.mean()}")

            cp = os.path.join(ckpt_dir, f'epoch{epoch}.pth')
            torch.save(self.model.state_dict(), cp)
            if val_loss < best_val:
                best_val, best_epoch = val_loss, epoch
                self.logger.info(f"Best updated at epoch {epoch}, Val {val_loss:.6f}")
        
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
        
        best_cp = os.path.join(ckpt_dir, f'epoch{best_epoch}.pth')
        self.model.load_state_dict(torch.load(best_cp))
        self.logger.info(f"Training complete. Loading epoch {best_epoch}")
        return self.model

    def backtest(self, topk: int = 5, holding_period: int = 5):
        test_data, _ = self._get_data(flag='test')
        self.model.eval()
        daily_returns, market_returns = [], []
        holdings_history, weights_history, dates = [], [], []
        current_holdings = None; days_held = 0
        for idx in range(len(test_data)):
            seq_x1, _, seq_y, _ = test_data[idx]
            market_ret = float(np.mean(seq_y)); market_returns.append(market_ret)
            if days_held == 0 or days_held >= holding_period:
                x = torch.from_numpy(seq_x1).float().to(self.device)
                stocks, seq_len, feat = seq_x1.shape
                x = x.view(-1, seq_len, feat)
                with torch.no_grad():
                    preds = self.model(x).cpu().numpy().reshape(-1)
                k = min(topk, len(preds))
                current_holdings = np.argsort(preds)[-k:]
                days_held = 1
            else:
                days_held += 1
            if current_holdings is not None and current_holdings.size>0:
                port_ret = float(np.mean(seq_y[current_holdings]))
            else:
                port_ret = 0.0
            daily_returns.append(port_ret)
            dates.append(idx)
            holdings_history.append(current_holdings.copy() if current_holdings is not None else np.array([]))
            if current_holdings is not None and current_holdings.size>0:
                w = np.zeros_like(seq_y); w[current_holdings] = 1.0/len(current_holdings)
                weights_history.append(w)
            else:
                weights_history.append(np.array([]))
        dr = np.array(daily_returns); mr = np.array(market_returns)
        ann = np.sqrt(252)
        sharpe = dr.mean()/(dr.std()+1e-9)*ann
        sortino = dr.mean()/(dr[dr<0].std()+1e-9)*ann
        cum = np.cumprod(1+dr); peak = np.maximum.accumulate(cum); max_dd = np.min((cum-peak)/peak)
        ann_ret = cum[-1]**(252/len(dr)) - 1
        m_sharpe = mr.mean()/(mr.std()+1e-9)*ann
        m_sortino = mr.mean()/(mr[mr<0].std()+1e-9)*ann
        m_cum = np.cumprod(1+mr); m_peak = np.maximum.accumulate(m_cum); m_dd = np.min((m_cum-m_peak)/m_peak)
        m_ann_ret = m_cum[-1]**(252/len(mr)) - 1
        ir = (dr-mr).mean()/(dr-mr).std()*ann
        self.logger.info("\n── Back-test Results ────────────────────────────")
        self.logger.info(f"top-{topk}, hold {holding_period}: Sharpe {sharpe:.4f}, Sortino {sortino:.4f}, MaxDD {max_dd:.4f}, AnnRet {ann_ret:.4f}")
        self.logger.info("Benchmark: Sharpe %.4f, Sortino %.4f, MaxDD %.4f, AnnRet %.4f" % (m_sharpe, m_sortino, m_dd, m_ann_ret))
        self.logger.info(f"Info Ratio: {ir:.4f}")
        out = {
            'day': dates,
            'strategy_return': dr,
            'market_return': mr,
            'strategy_cumulative': cum,
            'market_cumulative': m_cum,
            'excess_return': dr-mr
        }
        import pandas as pd
        df = pd.DataFrame(out)
        outdir = os.path.join('log', 'backtest_results', self.args.model)
        os.makedirs(outdir, exist_ok=True)
        path = os.path.join(outdir, f'perf_top{topk}_hold{holding_period}.csv')
        df.to_csv(path, index=False)
        self.logger.info(f"Saved backtest CSV to {path}")
