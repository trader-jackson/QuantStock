import torch
import torch.nn as nn
import numpy as np
import lightgbm as lgb
import xgboost as xgb

class LightGBM(nn.Module):
    """
    LightGBM model wrapper that follows PyTorch module interface
    """
    def __init__(self, config):
        super(LightGBM, self).__init__()
        self.config = config
        self.model = lgb.LGBMRegressor(
            boosting_type=config.boosting_type,
            num_leaves=config.num_leaves,
            max_depth=config.max_depth,
            learning_rate=config.learning_rate,
            n_estimators=config.n_estimators,
            feature_fraction=config.feature_fraction,
            bagging_fraction=config.bagging_fraction,
            bagging_freq=config.bagging_freq,
            verbose=-1,
            random_state=config.random_state
        )
        
    def forward(self, x):
        """
        Forward pass - convert tensor to numpy for LightGBM prediction
        then convert back to tensor
        
        Args:
            x: input tensor of shape [batch_size, seq_len, features]
            
        Returns:
            Tensor with predictions [batch_size]
        """
        # Reshape and convert to numpy for LightGBM
        batch_size = x.size(0)
        x_numpy = x.view(batch_size, -1).cpu().detach().numpy()
        
        # Predict
        preds = self.model.predict(x_numpy)
        
        # Convert back to tensor
        return torch.tensor(preds, device=x.device, dtype=torch.float32)


class XGBoost(nn.Module):
    """
    XGBoost model wrapper that follows PyTorch module interface
    """
    def __init__(self, config):
        super(XGBoost, self).__init__()
        self.config = config
        self.model = xgb.XGBRegressor(
            n_estimators=config.get('n_estimators', 100),
            max_depth=config.get('max_depth', 6),
            learning_rate=config.get('learning_rate', 0.1),
            min_child_weight=config.get('min_child_weight', 1),
            subsample=config.get('subsample', 0.8),
            colsample_bytree=config.get('colsample_bytree', 0.8),
            gamma=config.get('gamma', 0),
            reg_alpha=config.get('reg_alpha', 0),
            reg_lambda=config.get('reg_lambda', 1),
            scale_pos_weight=config.get('scale_pos_weight', 1),
            objective='reg:squarederror',
            random_state=config.get('random_state', 42),
            tree_method=config.get('tree_method', 'hist'),
            verbosity=0
        )
    
    def forward(self, x):
        """
        Forward pass - convert tensor to numpy for XGBoost prediction
        then convert back to tensor
        
        Args:
            x: input tensor of shape [batch_size, seq_len, features]
            
        Returns:
            Tensor with predictions [batch_size]
        """
        # Reshape and convert to numpy for XGBoost
        batch_size = x.size(0)
        x_numpy = x.view(batch_size, -1).cpu().detach().numpy()
        
        # Predict
        preds = self.model.predict(x_numpy)
        
        # Convert back to tensor
        return torch.tensor(preds, device=x.device, dtype=torch.float32) 