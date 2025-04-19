import os
import numpy as np
import torch
import sys
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import logging
from abc import ABC, abstractmethod
from utils.metrics import ranking_loss
from stock_data_handle import Stock_Data, DatasetStock_PRED
current_dir = os.path.dirname(os.path.abspath(__file__))

sys.path.insert(0, current_dir)

class PM(ABC):  # Inherit from ABC to use abstractmethod
    def __init__(self, args, data_all):
        # Check if log_dir exists in args, if not, set default
        
        log_dir = os.path.join('log', 'pred_' + args.project_name + "_" + args.model)
         # Store the log directory for future reference
        self.log_dir = log_dir
            
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        self.args = args
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)
        
       

        # Create a logger specific to this instance
        model_name = args.model if hasattr(args, 'model') else 'unknown'
        self.logger = logging.getLogger(f'PM_{model_name}')
        self.logger.setLevel(logging.INFO)
        
        # Remove any existing handlers to avoid duplicate logs
        if self.logger.hasHandlers():
            self.logger.handlers.clear()
            
        # Create file handler for this instance
        log_file = os.path.join(log_dir, f'PM_{model_name}.log')
        file_handler = logging.FileHandler(log_file,mode='w')
        file_handler.setLevel(logging.INFO)
        
        # Create formatter and add it to the handler
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        
        # Add handler to logger
        self.logger.addHandler(file_handler)
        
        # Also add a console handler for immediate feedback
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        self.logger.info(f"Initializing model with log directory: {log_dir}")
        self.logger.info(f"Log file created at {log_file}")

        self.data_all = data_all  # data_all is an instance of Stock_Data

    @abstractmethod
    def _build_model(self):
        pass

    def _get_data(self, flag):
        args = self.args

        if flag == 'train':
            shuffle_flag = True; drop_last = False; batch_size = args.batch_size
        else:
            shuffle_flag = False; drop_last = True; batch_size = args.batch_size
      
        dataset = DatasetStock_PRED(self.data_all, flag=flag)
        
        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)

        return dataset, data_loader

    def _select_optimizer(self):
        return optim.Adam(self.model.parameters(), lr=self.args.learning_rate)

    def _select_criterion(self):
        return nn.MSELoss()
    
    @abstractmethod
    def vali(self, vali_data, vali_loader, criterion, metric_builders, stage='test'):
        pass
    
    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def backtest(self, topk: int = 5, holding_period: int = 5):
        pass

    def _acquire_device(self):
        if self.args.use_gpu:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print('Use GPU: cuda:{}'.format(device))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device
