from PM import PM
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import os
import time
import matplotlib.pyplot as plt
from models.GCN import GCN

class PM_GCN(PM):
    def __init__(self, args, data_all):
        self.identifier = f"gcn_{id(self)}"  # Add identifier before calling super
        super(PM_GCN, self).__init__(args, data_all)
        self.logger.info(f"Initialized GCN model with identifier: {self.identifier}")
        
        # Set the device
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            self.logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            self.device = torch.device('cpu')
            self.logger.info("Using CPU")

    def _build_model(self):
        """Build the GCN model with the specified parameters"""
        self.logger.info("Building GCN model")
        
        model = GCN(self.args)
        model = model.to(self.device)
        self.logger.info(f"GCN model structure:\n{model}")
        
        return model

    def _select_optimizer(self):
        """Select optimizer for GCN model"""
        self.logger.info(f"Creating Adam optimizer with lr={self.args.learning_rate}, weight_decay={self.args.weight_decay}")
        return optim.Adam(
            self.model.parameters(),
            lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay
        )

    def _select_criterion(self):
        """Select loss criterion for GCN model"""
        self.logger.info("Using MSE loss for GCN model")
        return nn.MSELoss()

    def train(self):
        """Train the GCN model on training data"""
        train_data, train_loader = self._get_data(flag="train")
        valid_data, valid_loader = self._get_data(flag="valid")
        
        self.logger.info(f"Starting GCN model training for {self.args.epochs} epochs")
        self.logger.info(f"Train loader: {len(train_loader)} batches, Valid loader: {len(valid_loader)} batches")
        
        # Create the optimizer and criterion
        optimizer = self._select_optimizer()
        criterion = self._select_criterion()
        
        # Create checkpoint directory if it doesn't exist
        checkpoint_dir = os.path.join('./checkpoints/')
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        
        # Early stopping variables
        best_valid_loss = float('inf')
        patience_counter = 0
        
        # Performance tracking
        train_losses = []
        valid_losses = []
        
        # Training loop
        start_time = time.time()
        
        for epoch in range(self.args.epochs):
            epoch_start_time = time.time()
            
            # Training phase
            self.model.train()
            train_epoch_loss = 0.0
            train_batch_count = 0
            
            for i, (batch_x, _, batch_y, _) in enumerate(train_loader):
                # Move data to device
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                
                # Zero gradients
                optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(batch_x)
                loss = criterion(outputs, batch_y)
                
                # Backward pass and optimize
                loss.backward()
                optimizer.step()
                
                # Accumulate loss
                train_epoch_loss += loss.item()
                train_batch_count += 1
                
                # Log progress periodically
                if (i + 1) % 10 == 0 or (i + 1) == len(train_loader):
                    self.logger.info(f"Epoch [{epoch+1}/{self.args.epochs}], Batch [{i+1}/{len(train_loader)}], Train Loss: {loss.item():.6f}")
            
            # Calculate average training loss for the epoch
            avg_train_loss = train_epoch_loss / train_batch_count if train_batch_count > 0 else 0
            train_losses.append(avg_train_loss)
            
            # Validation phase
            self.model.eval()
            valid_epoch_loss = 0.0
            valid_batch_count = 0
            
            with torch.no_grad():
                for i, (batch_x, _, batch_y, _) in enumerate(valid_loader):
                    # Move data to device
                    batch_x = batch_x.float().to(self.device)
                    batch_y = batch_y.float().to(self.device)
                    
                    # Forward pass
                    outputs = self.model(batch_x)
                    loss = criterion(outputs, batch_y)
                    
                    # Accumulate loss
                    valid_epoch_loss += loss.item()
                    valid_batch_count += 1
            
            # Calculate average validation loss for the epoch
            avg_valid_loss = valid_epoch_loss / valid_batch_count if valid_batch_count > 0 else 0
            valid_losses.append(avg_valid_loss)
            
            # Log epoch results
            epoch_time = time.time() - epoch_start_time
            self.logger.info(f"Epoch [{epoch+1}/{self.args.epochs}] completed in {epoch_time:.2f}s - Train Loss: {avg_train_loss:.6f}, Valid Loss: {avg_valid_loss:.6f}")
            
            # Check for improvement
            if avg_valid_loss < best_valid_loss:
                best_valid_loss = avg_valid_loss
                patience_counter = 0
                
                # Save the best model
                best_model_path = os.path.join(checkpoint_dir, f"gcn_best_{self.identifier}.pth")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': avg_train_loss,
                    'valid_loss': avg_valid_loss,
                }, best_model_path)
                
                self.logger.info(f"Model improved! Saved best model at epoch {epoch+1} with validation loss: {avg_valid_loss:.6f}")
            else:
                patience_counter += 1
                self.logger.info(f"No improvement for {patience_counter} epochs. Best validation loss: {best_valid_loss:.6f}")
                
                if patience_counter >= self.args.early_stopping_patience:
                    self.logger.info(f"Early stopping triggered after {epoch+1} epochs")
                    break
        
        # Calculate total training time
        total_training_time = time.time() - start_time
        self.logger.info(f"Total training time: {total_training_time:.2f} seconds")
        
        # Load the best model
        best_model_path = os.path.join(checkpoint_dir, f"gcn_best_{self.identifier}.pth")
        if os.path.exists(best_model_path):
            checkpoint = torch.load(best_model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.logger.info(f"Loaded best model from epoch {checkpoint['epoch']+1} with valid loss: {checkpoint['valid_loss']:.6f}")
        
        # Save final model
        final_model_path = os.path.join(checkpoint_dir, f"gcn_final_{self.identifier}.pth")
        torch.save({
            'model_state_dict': self.model.state_dict(),
        }, final_model_path)
        self.logger.info(f"Saved final model to {final_model_path}")
        
        # Create directory for plots if it doesn't exist
        plots_dir = os.path.join('./plots/')
        if not os.path.exists(plots_dir):
            os.makedirs(plots_dir)
        
        # Plot training and validation loss
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='Training Loss')
        plt.plot(valid_losses, label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('GCN Training and Validation Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        loss_plot_path = os.path.join(plots_dir, f"gcn_loss_{self.identifier}.png")
        plt.savefig(loss_plot_path)
        plt.close()
        
        self.logger.info(f"Loss plot saved to {loss_plot_path}")
        
        return self.model

    def test(self):
        """Test the GCN model on test data"""
        test_data, test_loader = self._get_data(flag="test")
        
        self.logger.info("Testing GCN model")
        
        criterion = self._select_criterion()
        
        # Model in evaluation mode
        self.model.eval()
        
        # Collect all predictions and targets
        all_preds = []
        all_targets = []
        test_loss = 0.0
        
        with torch.no_grad():
            for i, (batch_x, _, batch_y, _) in enumerate(test_loader):
                # Move data to device
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                
                # Forward pass
                outputs = self.model(batch_x)
                loss = criterion(outputs, batch_y)
                
                # Accumulate loss
                test_loss += loss.item()
                
                # Store predictions and targets
                all_preds.append(outputs.cpu().numpy())
                all_targets.append(batch_y.cpu().numpy())
                
                # Log progress periodically
                if (i + 1) % 10 == 0 or (i + 1) == len(test_loader):
                    self.logger.info(f"Tested batch [{i+1}/{len(test_loader)}], Loss: {loss.item():.6f}")
        
        # Convert lists to numpy arrays
        all_preds = np.concatenate(all_preds, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        
        # Calculate metrics
        mse = np.mean((all_preds - all_targets) ** 2)
        rmse = np.sqrt(mse)
        
        # Calculate correlation coefficients
        from scipy.stats import pearsonr, spearmanr
        
        # Flatten arrays if necessary for correlation calculation
        preds_flat = all_preds.flatten()
        targets_flat = all_targets.flatten()
        
        pearson, _ = pearsonr(preds_flat, targets_flat)
        spearman, _ = spearmanr(preds_flat, targets_flat)
        
        # Log metrics
        self.logger.info(f"Test Loss: {test_loss / len(test_loader):.6f}")
        self.logger.info(f"MSE: {mse:.6f}")
        self.logger.info(f"RMSE: {rmse:.6f}")
        self.logger.info(f"Pearson correlation: {pearson:.6f}")
        self.logger.info(f"Spearman correlation: {spearman:.6f}")
        
        # Create directory for plots if it doesn't exist
        plots_dir = os.path.join('./plots/')
        if not os.path.exists(plots_dir):
            os.makedirs(plots_dir)
        
        # Plot predictions vs targets
        plt.figure(figsize=(10, 6))
        plt.scatter(targets_flat, preds_flat, alpha=0.3)
        plt.plot([targets_flat.min(), targets_flat.max()], [targets_flat.min(), targets_flat.max()], 'r--')
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.title('GCN: Predicted vs Actual')
        
        scatter_plot_path = os.path.join(plots_dir, f"gcn_scatter_{self.identifier}.png")
        plt.savefig(scatter_plot_path)
        plt.close()
        
        self.logger.info(f"Scatter plot saved to {scatter_plot_path}")
        
        return mse, rmse, pearson, spearman

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
        
        daily_returns = []
        market_returns = []  # Market average benchmark (equal-weight all stocks)
        current_holdings = None
        days_held = 0  # 1‑based counter AFTER re‑balance
        
        # Create lists to track holdings and weights for each day
        holdings_history = []  # List of stock indices held each day
        weights_history = []   # List of weights for each stock each day
        dates_history = []     # List of dates (day indices)
        
        self.logger.info(f"Starting backtest with top-{topk} stocks and {holding_period}-day holding period")
        
        # Set model to evaluation mode
        self.model.eval()
        
        for i in range(len(test_dataset.dataset)):
            # Get features and targets for day i
            seq_x, _, seq_y, _ = test_dataset.dataset[i]
            
            # Calculate market average return (equal weight all stocks)
            market_return = np.mean(seq_y.numpy())
            market_returns.append(market_return)
            
            # ─── Re‑balance? ─────────────────────────────────────────
            if days_held == 0 or days_held >= holding_period:
                # Prepare input for GCN model (add batch dimension)
                x = seq_x.float().unsqueeze(0).to(self.device)
                
                # Make predictions
                with torch.no_grad():
                    predictions = self.model(x).squeeze().cpu().numpy()
                
                # Make sure we don't select more stocks than available
                available_stocks = min(len(seq_y), len(predictions))
                topk_effective = min(topk, available_stocks)
                
                # Pick top‑k indices
                current_holdings = np.argsort(predictions[:available_stocks])[-topk_effective:]
                days_held = 1  # Start counting days *after* rebalance
                
                if i < 3:  # Log only first few iterations for debugging
                    self.logger.info(f"[DAY {i}] top‑{topk_effective} indices → {current_holdings}")
                    if i == 0:  # More detailed logging for first iteration
                        self.logger.info(f"seq_y shape: {seq_y.shape}, predictions shape: {predictions.shape}")
                        self.logger.info(f"First few predictions: {predictions[:5] if len(predictions) >= 5 else predictions}")
            
            # ─── Portfolio return for today ─────────────────────────
            if current_holdings is not None and len(current_holdings) > 0:
                portfolio_ret = np.mean(seq_y.numpy()[current_holdings])
            else:
                portfolio_ret = 0.0  # Default if no holdings
                self.logger.warning(f"No valid holdings for day {i}")
            
            # Append to history trackers
            daily_returns.append(portfolio_ret)
            dates_history.append(i)
            holdings_history.append(current_holdings.copy() if current_holdings is not None else np.array([]))
            
            # Equal weight allocation for stocks in the portfolio
            if current_holdings is not None and len(current_holdings) > 0:
                weights = np.zeros(len(seq_y))
                weight_per_stock = 1.0 / len(current_holdings)
                for idx in current_holdings:
                    weights[idx] = weight_per_stock
                weights_history.append(weights)
            else:
                weights_history.append(np.array([]))
            
            days_held += 1  # Increment until next rotation
        
        # ─── Performance metrics ────────────────────────────────────
        daily_returns = np.asarray(daily_returns)
        market_returns = np.asarray(market_returns)
        trading_days = len(daily_returns)
        ann_factor = np.sqrt(252)  # Annualization factor for daily data
        
        # Strategy metrics
        sharpe = np.mean(daily_returns) / (np.std(daily_returns) + 1e-9) * ann_factor
        downside = daily_returns[daily_returns < 0]
        sortino = np.mean(daily_returns) / (np.std(downside) + 1e-9) * ann_factor
        
        cum_ret = np.cumprod(1 + daily_returns)
        peak = np.maximum.accumulate(cum_ret)
        max_dd = np.min((cum_ret - peak) / peak)
        ann_ret = cum_ret[-1] ** (252 / trading_days) - 1
        
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
        
        # Log results
        self.logger.info("\n── Back‑test ─────────────────────────────────────────")
        self.logger.info(f"top‑{topk} | hold {holding_period}‑d")
        self.logger.info("\nStrategy Performance:")
        self.logger.info(f"Sharpe Ratio      : {sharpe:.4f}")
        self.logger.info(f"Sortino Ratio     : {sortino:.4f}")
        self.logger.info(f"Maximum Drawdown  : {max_dd:.4f}")
        self.logger.info(f"Annualized Return : {ann_ret:.4f}")
        self.logger.info(f"Total Return      : {cum_ret[-1] - 1:.4f}")
        
        self.logger.info("\nBenchmark (Market Average):")
        self.logger.info(f"Sharpe Ratio      : {market_sharpe:.4f}")
        self.logger.info(f"Sortino Ratio     : {market_sortino:.4f}")
        self.logger.info(f"Maximum Drawdown  : {market_max_dd:.4f}")
        self.logger.info(f"Annualized Return : {market_ann_ret:.4f}")
        self.logger.info(f"Total Return      : {market_cum_ret[-1] - 1:.4f}")
        
        self.logger.info(f"\nInformation Ratio : {information_ratio:.4f}")
        self.logger.info(f"Trading Days      : {trading_days}")
        
        # Store all data in a DataFrame
        performance_df = pd.DataFrame({
            'day': dates_history,
            'strategy_return': daily_returns,
            'market_return': market_returns,
            'strategy_cumulative': cum_ret,
            'market_cumulative': market_cum_ret,
            'excess_return': excess_returns
        })
        
        # Create directory for plots if it doesn't exist
        plots_dir = os.path.join('./plots/')
        if not os.path.exists(plots_dir):
            os.makedirs(plots_dir)
        
        # Plot cumulative returns
        plt.figure(figsize=(12, 6))
        plt.plot(cum_ret, label='Strategy')
        plt.plot(market_cum_ret, label='Market')
        plt.title(f'GCN Strategy (top-{topk}, hold-{holding_period}d) vs Market')
        plt.xlabel('Trading Days')
        plt.ylabel('Cumulative Return')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        returns_plot_path = os.path.join(plots_dir, f"gcn_returns_{self.identifier}_top{topk}_hold{holding_period}.png")
        plt.savefig(returns_plot_path)
        plt.close()
        
        self.logger.info(f"Cumulative returns plot saved to {returns_plot_path}")
        
        # Save to CSV files
        log_dir = os.path.join('log', 'backtest_results')
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        # Save performance data
        performance_file = os.path.join(log_dir, f'gcn_performance_{self.identifier}_top{topk}_hold{holding_period}.csv')
        performance_df.to_csv(performance_file, index=False)
        self.logger.info(f"Performance data saved to {performance_file}")
        
        # Create a pickled file with all data including holdings and weights
        import pickle
        backtest_data = {
            'performance': performance_df,
            'holdings_history': holdings_history,
            'weights_history': weights_history,
            'params': {
                'topk': topk,
                'holding_period': holding_period
            },
            'metrics': {
                'sharpe': sharpe,
                'sortino': sortino,
                'max_drawdown': max_dd,
                'annual_return': ann_ret,
                'total_return': cum_ret[-1] - 1,
                'market_sharpe': market_sharpe,
                'market_sortino': market_sortino,
                'market_max_drawdown': market_max_dd,
                'market_annual_return': market_ann_ret,
                'market_total_return': market_cum_ret[-1] - 1,
                'information_ratio': information_ratio
            }
        }
        
        pickle_file = os.path.join(log_dir, f'gcn_backtest_data_{self.identifier}_top{topk}_hold{holding_period}.pkl')
        with open(pickle_file, 'wb') as f:
            pickle.dump(backtest_data, f)
        
        self.logger.info(f"Complete backtest data saved to {pickle_file}")
        
        return backtest_data 