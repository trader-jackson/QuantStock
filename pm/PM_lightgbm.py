import os
import sys
import time
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from types import SimpleNamespace
import matplotlib.pyplot as plt

from models.GBDT import LightGBM
from utils.metrics import ranking_loss
import utils.metrics_object as metrics_object
from .PM import PM

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

class PM_LightGBM(PM):
    """
    Portfolio management using a LightGBM-based prediction model.
    Uses `model_config` dict from args for LightGBM settings.
    """
    def __init__(self, args, data_all):
        super(PM_LightGBM, self).__init__(args, data_all)
        self.device="cpu"

    def _build_model(self):
        config = SimpleNamespace(
            boosting_type=self.args.model_config.get('boosting_type', 'gbdt'),
            num_leaves=self.args.model_config.get('num_leaves', 31),
            max_depth=self.args.model_config.get('max_depth', 6),
            learning_rate=self.args.model_config.get('learning_rate', 0.1),
            n_estimators=self.args.model_config.get('n_estimators', 100),
            feature_fraction=self.args.model_config.get('feature_fraction', 0.8),
            bagging_fraction=self.args.model_config.get('bagging_fraction', 0.8),
            bagging_freq=self.args.model_config.get('bagging_freq', 5),
            random_state=self.args.model_config.get('random_state', 42)
        )
        model = LightGBM(config)
        return model

    def vali(self, vali_data, vali_loader, criterion, builders, stage='test'):
        self.model.eval()
        losses = []
        metrics = [b(stage) for b in builders]
        
        for batch_x1, _, batch_y, _ in vali_loader:
            bs, stocks, seq_len, feat = batch_x1.shape
            # Reshape for LightGBM input - it accepts flattened features
            x = batch_x1.reshape(bs * stocks, -1).float().to(self.device)
            y = batch_y.float().to(self.device)
            
            # Get predictions and reshape
            with torch.no_grad():
                preds = self.model(x).view(bs, stocks)
                loss = criterion(preds, y) + self.args.rank_alpha * ranking_loss(preds, y)
                losses.append(loss.item())
                
                for m in metrics:
                    m.update(preds, y)
                    
        avg = np.mean(losses)
        return avg, metrics

    def train(self):
        train_data, train_loader = self._get_data(flag='train')
        valid_data, valid_loader = self._get_data(flag='valid')
        test_data, test_loader = self._get_data(flag='test')
        metrics_builders = [metrics_object.MIRRTop1, metrics_object.RankIC]
        ckpt_dir = os.path.join('checkpoints', f'{self.args.model}')
        os.makedirs(ckpt_dir, exist_ok=True)

        self.logger.info(f"LightGBM training start")
        
        # Prepare data for one-time training (LightGBM doesn't use epochs)
        # We need to collect all training data
        all_x = []
        all_y = []
        
        for batch_x1, _, batch_y, _ in train_loader:
            bs, stocks, seq_len, feat = batch_x1.shape
            x = batch_x1.reshape(bs * stocks, seq_len * feat).cpu().numpy()
            y = batch_y.reshape(bs * stocks).cpu().numpy()
            all_x.append(x)
            all_y.append(y)
            
        all_x = np.vstack(all_x)
        all_y = np.concatenate(all_y)
        
        # Create lists to store metrics for plots
        epochs = []
        train_rankic_values = []
        valid_rankic_values = []
        test_rankic_values = []
        train_mirrtop1_values = []
        valid_mirrtop1_values = []
        test_mirrtop1_values = []
        
        # Train LightGBM model
        time_start = time.time()
        self.logger.info(f"Training LightGBM with {all_x.shape[0]} samples, {all_x.shape[1]} features")
        
        # Set up validation data for early stopping
        valid_x = []
        valid_y = []
        for batch_x1, _, batch_y, _ in valid_loader:
            bs, stocks, seq_len, feat = batch_x1.shape
            x = batch_x1.reshape(bs * stocks, seq_len * feat).cpu().numpy()
            y = batch_y.reshape(bs * stocks).cpu().numpy()
            valid_x.append(x)
            valid_y.append(y)
        
        valid_x = np.vstack(valid_x)
        valid_y = np.concatenate(valid_y)
        
        # Get the internal LightGBM model
        lgb_model = self.model.model
        
        # Configure eval set for early stopping
        eval_set = [(all_x, all_y), (valid_x, valid_y)]
        
        # Train with early stopping
        lgb_model.fit(
            all_x, all_y,
            eval_set=eval_set,
            eval_metric='rmse',
            early_stopping_rounds=self.args.model_config.get('early_stopping_rounds', 10),
            verbose=False
        )
        
        # Track training time
        elapsed = time.time() - time_start
        self.logger.info(f"LightGBM training completed in {elapsed:.2f} seconds")
        
        # Now evaluate on train, validation, test
        criterion = self._select_criterion()
        
        # Since we don't have epochs for LightGBM, we'll just record metrics once
        epochs.append(1)  # Just a single point for plotting
        
        # Calculate metrics on train set
        train_metrics_dict = {}
        train_preds = []
        train_actuals = []
        
        for batch_x1, _, batch_y, _ in train_loader:
            bs, stocks, seq_len, feat = batch_x1.shape
            x = batch_x1.reshape(bs * stocks, seq_len * feat).float()
            y = batch_y.float()
            
            with torch.no_grad():
                batch_preds = torch.tensor(
                    lgb_model.predict(x.cpu().numpy()), 
                    dtype=torch.float32
                ).view(bs, stocks)
                
                train_preds.append(batch_preds)
                train_actuals.append(y)
        
        train_preds = torch.cat(train_preds, dim=0)
        train_actuals = torch.cat(train_actuals, dim=0)
        
        # Calculate RankIC and MIRRTop1 for training set
        rankic_train = metrics_object.RankIC("train")
        mirrtop1_train = metrics_object.MIRRTop1("train")
        rankic_train.update(train_preds, train_actuals)
        mirrtop1_train.update(train_preds, train_actuals)
        
        train_metrics_dict['RankIC'] = np.mean(rankic_train.data) if len(rankic_train.data) > 0 else 0
        train_metrics_dict['MIRRTop1'] = mirrtop1_train.mean()
        
        # Calculate metrics on validation set
        valid_loss, valid_metrics = self.vali(None, valid_loader, criterion, metrics_builders, 'valid')
        valid_metrics_dict = {}
        
        for metric in valid_metrics:
            if 'mirr' in metric.name.lower() and 'top1' in metric.name.lower():
                valid_metrics_dict['MIRRTop1'] = metric.mean()
            elif 'rank' in metric.name.lower() and 'ic' in metric.name.lower():
                valid_metrics_dict['RankIC'] = np.mean(metric.data) if len(metric.data) > 0 else 0
        
        # Calculate metrics on test set
        test_loss, test_metrics = self.vali(None, test_loader, criterion, metrics_builders, 'test')
        test_metrics_dict = {}
        
        for metric in test_metrics:
            if 'mirr' in metric.name.lower() and 'top1' in metric.name.lower():
                test_metrics_dict['MIRRTop1'] = metric.mean()
            elif 'rank' in metric.name.lower() and 'ic' in metric.name.lower():
                test_metrics_dict['RankIC'] = np.mean(metric.data) if len(metric.data) > 0 else 0
        
        # Store values for plotting
        train_rankic_values.append(train_metrics_dict.get('RankIC', 0))
        valid_rankic_values.append(valid_metrics_dict.get('RankIC', 0))
        test_rankic_values.append(test_metrics_dict.get('RankIC', 0))
        train_mirrtop1_values.append(train_metrics_dict.get('MIRRTop1', 0))
        valid_mirrtop1_values.append(valid_metrics_dict.get('MIRRTop1', 0))
        test_mirrtop1_values.append(test_metrics_dict.get('MIRRTop1', 0))
        
        # Log metrics
        self.logger.info(f"Train RankIC: {train_metrics_dict.get('RankIC', 0):.6f}, MIRRTop1: {train_metrics_dict.get('MIRRTop1', 0):.6f}")
        self.logger.info(f"Valid RankIC: {valid_metrics_dict.get('RankIC', 0):.6f}, MIRRTop1: {valid_metrics_dict.get('MIRRTop1', 0):.6f}")
        self.logger.info(f"Test RankIC: {test_metrics_dict.get('RankIC', 0):.6f}, MIRRTop1: {test_metrics_dict.get('MIRRTop1', 0):.6f}")
        
        # Save model checkpoint
        model_path = os.path.join(ckpt_dir, 'model.pth')
        torch.save(self.model.state_dict(), model_path)
        self.logger.info(f"Model saved to {model_path}")
        
        # Create plots directory
        plots_dir = os.path.join('log', 'plots', self.args.model)
        os.makedirs(plots_dir, exist_ok=True)
        
        # Plot RankIC metrics
        plt.figure(figsize=(10, 6))
        plt.bar(['Train', 'Valid', 'Test'], 
               [train_rankic_values[0], valid_rankic_values[0], test_rankic_values[0]])
        plt.xlabel('Dataset')
        plt.ylabel('RankIC')
        plt.title('RankIC Metrics')
        plt.grid(True)
        rankic_plot_path = os.path.join(plots_dir, 'rankic_metrics.png')
        plt.savefig(rankic_plot_path)
        plt.close()
        
        # Plot MIRRTop1 metrics
        plt.figure(figsize=(10, 6))
        plt.bar(['Train', 'Valid', 'Test'], 
               [train_mirrtop1_values[0], valid_mirrtop1_values[0], test_mirrtop1_values[0]])
        plt.xlabel('Dataset')
        plt.ylabel('MIRRTop1')
        plt.title('MIRRTop1 Metrics')
        plt.grid(True)
        mirrtop1_plot_path = os.path.join(plots_dir, 'mirrtop1_metrics.png')
        plt.savefig(mirrtop1_plot_path)
        plt.close()
        
        self.logger.info(f"Metric plots saved to {plots_dir}")
        
        return self.model

    def backtest(self, topk: int = 5, holding_period: int = 5):
        """
        Run a backtest on test data.
        
        Args:
            topk: Number of top stocks to hold
            holding_period: Number of days to hold positions before rebalancing
        """
        test_data, _ = self._get_data(flag='test')
        self.model.eval()
        daily_returns, market_returns = [], []
        holdings_history, weights_history, dates = [], [], []
        current_holdings = None
        days_held = 0
        
        for idx in range(len(test_data)):
            seq_x1, _, seq_y, _ = test_data[idx]
            market_ret = float(np.mean(seq_y))
            market_returns.append(market_ret)
            
            if days_held == 0 or days_held >= holding_period:
                # Need to reshape for LightGBM
                stocks, seq_len, feat = seq_x1.shape
                x = torch.from_numpy(seq_x1).reshape(stocks, -1)
                
                # Get predictions
                with torch.no_grad():
                    preds = self.model.model.predict(x.cpu().numpy())
                    
                # Select top-k stocks
                k = min(topk, len(preds))
                current_holdings = np.argsort(preds)[-k:]
                days