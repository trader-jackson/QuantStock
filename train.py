# test_pm.py
import os
import torch
from PM import PM
from stock_data_handle import Stock_Data
import config

def main():
    
    class DummyArgs:
        # Choose one among "Transformer", "LSTM", "XGBoost", "LightGBM"
        project_name = "DOW30_Project"
        rank_alpha = 1
        batch_size = 32
        num_workers = 1
        learning_rate = 0.001
        train_epochs = 50
        use_multi_gpu = False
        use_gpu = True
        device_ids = [0]
        # Model-specific configs:
        enc_in = 5
        dec_in = 5
        c_out = 1       # Assuming DOW30 has 30 stocks
        d_model = 128
        n_heads = 4
        e_layers = 2
        d_layers = 1
        d_ff = 256
        dropout = 0.05
        activation = "gelu"
        use_attn = True
        noise_level = 0.05
        pred_type = "label_short_term"  # This parameter is ignored by DatasetStock_PRED
        early_stopping = 10

    args = DummyArgs()
    device = torch.device("cuda:0" if args.use_gpu else "cpu")
    args.device = device

    root_path = "data_dir"       # Folder containing the "DOW30" folder with CSV files.
    dataset_name = "DOW"         # Key defined in config.use_ticker_dict.
    full_stock_path = "DOW30"      # Folder name for DOW30 data.
    seq_len = 60
    prediction_len = 1

    stock_data = Stock_Data(dataset_name=dataset_name, full_stock_path=full_stock_path,
                            window_size=seq_len, root_path=root_path, prediction_len=prediction_len, scale=True)
    
    for model_type in ["Transformer",]: #"LSTM", "XGBoost", "LightGBM"
        print("\n" + "="*40)
        print("Testing model type:", model_type)
        args.model = model_type
        exp = PM(args, stock_data, id="test")
        exp.device = device
        print("Training the model...")
        model = exp.train("test_checkpoint")
        print("Testing the model...")
        exp.test("test_results")
        print("Running backtest...")
        backtest_results = exp.backtest(topk=5, holding_period=5)
        print("Backtest Results:", backtest_results)
        print("="*40)

if __name__ == "__main__":
    main()
