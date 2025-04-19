import os
import numpy as np
import pandas as pd
import datetime
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
import config  # This module should define: TECHICAL_INDICATORS, TEMPORAL_FEATURE, use_ticker_dict, date_dict, INF

# -----------------------------------------------------------------
# Stock_Data class definition (unchanged as provided)
# -----------------------------------------------------------------
class Stock_Data():
    def __init__(self, dataset_name, full_stock_path, window_size, root_path="data_dir",
                 attr=config.TECHICAL_INDICATORS, temporal_feature=config.TEMPORAL_FEATURE, prediction_len=1, scale=True):
        # size: [seq_len, label_len, pred_len] (only seq_len is used)
        self.attr = attr
        self.root_path = root_path
        self.full_stock = full_stock_path  
        self.ticker_list = config.use_ticker_dict[dataset_name]
        self.border_dates = config.date_dict[dataset_name]
        self.temporal_feature = temporal_feature
        self.prediction_len = prediction_len
        self.seq_len = window_size
        self.type_map = {'train': 0, 'valid': 1, 'test': 2}
        self.scale = scale
        self.__read_data__()
        
    def __read_data__(self):
        stock_num = len(self.ticker_list)
        scaler = StandardScaler()
      
        
        # Read data for all types (train, valid, test) into a single DataFrame.
        df = pd.DataFrame([], columns=['date', 'close', 'high', 'low', 'open', 'volume', 'tic', 'price', 'day',
                                         'macd', 'boll_ub', 'boll_lb', 'rsi_30', 'cci_30', 'dx_30', 'close_30_sma', 'close_60_sma'])
        for type_ in ["train", "valid", "test"]:
            temp_loc = os.path.join(self.root_path, self.full_stock, type_ + ".csv")
            temp_df = pd.read_csv(temp_loc, usecols=['date','close','high','low','open','volume','tic','price','day',
                                              'macd','boll_ub','boll_lb','rsi_30','cci_30','dx_30','close_30_sma','close_60_sma'])
            temp_df['date'] = pd.to_datetime(temp_df['date'])
            df = pd.concat((df, temp_df))
        
        if df.empty:
            raise ValueError("No data was loaded. Check if the CSV files exist and contain the required columns.")
            
        
        # Calculate short-term labels grouped by each stock using close price percentage change.
        try:
            df['label_short_term'] = df.groupby('tic')['price'].transform(
                lambda x: x.pct_change(periods=self.prediction_len).shift(-self.prediction_len)
            )
            df = df.dropna(subset=['label_short_term']).reset_index(drop=True)
            df = df.sort_values(by=['date', 'tic']).reset_index(drop=True)
        except Exception as e:
            print(f"Error calculating labels: {str(e)}")
            raise

        # Use factorized dates for the index (each unique date gets a unique index)
        df.index = df.date.factorize()[0]
        
        cov_list = []
        return_list = []

        print("Generate covariate matrix...")
        lookback = 252  # One year lookback period
        unique_date_count = len(df.index.unique())
        for i in range(lookback, unique_date_count):
            data_lookback = df.loc[i-lookback:i, :]
            price_lookback = data_lookback.pivot_table(index='date', columns='tic', values='price')
            return_lookback = price_lookback.pct_change().dropna()
            return_list.append(return_lookback)
    
            covs = return_lookback.cov().values 
            cov_list.append(covs)

        df_cov = pd.DataFrame({
            'date': df.date.unique()[lookback:],
            'cov_list': cov_list,
            'return_list': return_list
        })
        df = df.merge(df_cov, on='date')
        df = df.sort_values(['date','tic']).reset_index(drop=True)
        # Use factorized dates for the index (each unique date gets a unique index)
        df.index = df.date.factorize()[0]
        
        # Generate date strings for dataset splitting.
        df['date_str'] = df['date'].apply(lambda x: datetime.datetime.strftime(x, '%Y%m%d'))
        
        # Get border dates for splitting using self.border_dates.
        dates = df['date_str'].unique().tolist()
        boarder1_ = dates.index(self.border_dates[0])
        boarder1  = dates.index(self.border_dates[1])
        boarder2_ = dates.index(self.border_dates[2])
        boarder2  = dates.index(self.border_dates[3])
        boarder3_ = dates.index(self.border_dates[4])
       
        self.boarder_start = [max(boarder1_,self.seq_len), boarder2_, boarder3_]
        self.boarder_end = [boarder1, boarder2, len(dates)-1]

 
        
        df_data = df[self.attr]
        df_data = df_data.replace([np.inf], config.INF)
        df_data = df_data.replace([-np.inf], config.INF * (-1))
        if self.scale:
            data = scaler.fit_transform(df_data.values)
        else:
            data = df_data.values
        
        df[self.attr]=data
        
        # Additionally, split the DataFrame for backtesting analysis.
        self.df=df
        self.train_df = df[df['date_str'].isin(dates[self.boarder_start[0]:self.boarder_end[0]+1])].reset_index(drop=True)
        self.valid_df = df[df['date_str'].isin(dates[self.boarder_start[1]:self.boarder_end[1]+1])].reset_index(drop=True)
        self.test_df  = df[df['date_str'].isin(dates[self.boarder_start[2]:self.boarder_end[2]+1])].reset_index(drop=True)

        cov_list = np.array(df['cov_list'].values.tolist())
        feature_list = np.array(df[self.temporal_feature].values.tolist())
        close_list = np.array(df['price'].values.tolist())

        # Reshape the covariate data.
        data_cov = cov_list.reshape(-1, stock_num, cov_list.shape[1], cov_list.shape[2])
        data_technical = data.reshape(-1, stock_num, len(self.attr))
        data_feature = feature_list.reshape(-1, stock_num, len(self.temporal_feature))
        data_close = close_list.reshape(-1, stock_num)

        label_short_term = np.array(df['label_short_term'].values.tolist()).reshape(-1, stock_num)
        
        # Combine features: use the first slice from covariate data, then technical and temporal features.
        self.data_all = np.concatenate((data_cov[:, 0, :, :], data_technical, data_feature), axis=-1)
        # Create labels with an extra axis, shape becomes (1, days, stocks)
        self.label_all = np.expand_dims(label_short_term, axis=0)
        self.dates = np.array(dates)
        self.data_close = data_close

        print("data shape: ", self.data_all.shape)
        print("label shape: ", self.label_all.shape)
        print("Price shape: ", self.data_close.shape)

# -----------------------------------------------------------------
# DatasetStock_PRED class definition (unchanged except for extra output of price)
# -----------------------------------------------------------------
class DatasetStock_PRED(Dataset):
    def __init__(self, stock: Stock_Data, flag='train', feature=config.TEMPORAL_FEATURE,techical=config.TECHICAL_INDICATORS):
        super().__init__()
        assert flag in ['train', 'valid', 'test']
        pos = stock.type_map[flag]
        self.start_pos = stock.boarder_start[pos]
        self.end_pos = stock.boarder_end[pos] + 1
        # The length of the temporal features (last portion of the feature dimension)
        self.feature_len = len(feature)+len(techical)
        self.feature_day_len = stock.seq_len
        self.data = stock.data_all
        self.label = stock.label_all
        self.price = stock.data_close  # Current prices per day
        self.dates = stock.dates[self.start_pos: self.end_pos]

    def __getitem__(self, index):
        position = self.start_pos + index
        window_start = position - (self.feature_day_len - 1)
        window_end = position + 1

        # Get window data: shape [seq_len, stocks, total_feature_dim]
        seq_x = self.data[window_start: window_end]
        # Transpose to shape [stocks, seq_len, total_feature_dim]
        seq_x = seq_x.transpose(1, 0, 2)
        # Only keep the last self.feature_len columns from the feature dimension (temporal features)
        seq_x = seq_x[:, :, -self.feature_len:]
        # Ensure the array is contiguous and writable.
        seq_x = np.ascontiguousarray(seq_x)
        seq_x = np.copy(seq_x)

        # Decoder input: last time step (for consistency), shape [stocks, 1, feature_len]
        seq_x_dec = seq_x[:, -1:, :]
        seq_x_dec = np.ascontiguousarray(seq_x_dec)
        seq_x_dec = np.copy(seq_x_dec)

        # Label: current day's label, shape [stocks]
        seq_y = self.label[0][position]
        seq_y = np.ascontiguousarray(seq_y)
        seq_y = np.copy(seq_y)

        # Price for the current day: shape [stocks]
        seq_price = self.price[position]
        seq_price = np.ascontiguousarray(seq_price)
        seq_price = np.copy(seq_price)

        return seq_x, seq_x_dec, seq_y, seq_price

    def __len__(self):
        return self.end_pos - self.start_pos



# -----------------------------------------------------------------
# Testing functions for DatasetStock_PRED
# -----------------------------------------------------------------
def test_stock_data():
    # Test parameters
    root_path = 'data_dir'        # Ensure this path exists and contains your CSV files in the proper subfolder
    dataset_name = 'DOW'          # Key used in use_ticker_dict (must be defined in config)
    dataset_dir = 'DOW30'         # Subfolder containing train.csv, valid.csv, test.csv
    seq_len = 10                  # Example sequence length

    # Create a Stock_Data instance
    stock_data = Stock_Data(
        dataset_name=dataset_name,
        full_stock_path=dataset_dir,
        window_size=seq_len,
        root_path=root_path,
        prediction_len=5
    )
    return stock_data

def test_dataset_pred():
    # Create stock_data from test function.
    stock_data = test_stock_data()
    # Create a DatasetStock_PRED instance for training data (adjust type as needed)
    train_dataset = DatasetStock_PRED(stock_data, flag='test', feature=config.TEMPORAL_FEATURE)
    print("Train dataset length:", len(train_dataset))
    
    # Get the first sample and print output shapes
    sample = train_dataset[0]
    seq_x, seq_x_dec, seq_y, seq_price = sample
    print("seq_x shape:", seq_x.shape)          # Expected: [stocks, seq_len, feature_dim]
    print("seq_x_dec shape:", seq_x_dec.shape)    # Expected: [stocks, 1, feature_dim]
    print("seq_y shape:", seq_y.shape)            # Expected: [stocks]
    print("seq_price shape:", seq_price.shape)    # Expected: [stocks]

if __name__ == "__main__":
    test_dataset_pred()
    print("DatasetStock_PRED test completed successfully.")
