"""Contains methods and classes to collect data from
Yahoo Finance API
"""

from __future__ import annotations

import pandas as pd
import os
import yfinance as yf
from tqdm import tqdm
import numpy as np
from sklearn.linear_model import LinearRegression
from pathlib import Path
import sys
import os.path as osp
ROOT = str(Path(__file__).resolve().parents[0])
sys.path.append(str(Path(__file__).resolve().parents[1]))
from config_tickers import DOW_30_TICKER, NAS_100_TICKER, SSE_50_TICKER
from config import TECHICAL_INDICATORS,TRAIN_START_DATE,TRAIN_END_DATE,EVAL_START_DATE,EVAL_END_DATE,TEST_START_DATE,TEST_END_DATE
from sklearn.preprocessing import MinMaxScaler,StandardScaler
data_dict={"DOW30":DOW_30_TICKER,"NASQ100":NAS_100_TICKER,"SSE50":SSE_50_TICKER}

from utils import get_attr
from sklearn.preprocessing import MinMaxScaler

class YfinancePreprocessor():
    def __init__(self, dataset_name="DOW30",**kwargs):
        super(YfinancePreprocessor, self).__init__()
        self.kwargs = kwargs

        self.data_path = osp.join(ROOT, get_attr(kwargs, "data_path", "data_dir"))
        
        if not os.path.exists(self.data_path):
            os.makedirs(self.data_path)
        self.dataset_name=dataset_name
        self.dataset_path = osp.join(self.data_path, self.dataset_name)
        
        if not os.path.exists(self.dataset_path):
            os.makedirs(self.dataset_path,exist_ok=False)
            
        self.train_valid_test_portion = get_attr(kwargs,"train_valid_test_portion", [0.8, 0.1, 0.1])

        self.train_path = osp.join(self.dataset_path, get_attr(kwargs, "train_path", "train.csv"))
        self.valid_path = osp.join(self.dataset_path, get_attr(kwargs, "valid_path", "valid.csv"))
        self.test_path = osp.join(self.dataset_path, get_attr(kwargs, "test_path", "test.csv"))

        self.start_date = get_attr(kwargs, "start_date", "2012-01-01")
        self.end_date = get_attr(kwargs, "end_date", "2025-03-01")
        self.tickers=data_dict[self.dataset_name]

        self.feature_mode = get_attr(kwargs, "feature_mode", 'basic')
        self.tech_indicator_list=get_attr(kwargs, "techical_indicator", TECHICAL_INDICATORS)
        
        self.add_vix=get_attr(kwargs, "add_vis", False)
    def download_data(self,proxy=None, auto_adjust=False):
        """Fetches data from Yahoo API
        Parameters
        ----------

        Returns
        -------
        `pd.DataFrame`
            7 columns: A date, open, high, low, close, volume and tick symbol
            for the specified stock ticker
        """
        # Download and save the data in a pandas DataFrame:
        data_df = pd.DataFrame()
        num_failures = 0
        for tic in self.tickers:
            temp_df = yf.download(
                tic,
                start=self.start_date,
                end=self.end_date,
                proxy=proxy,
                auto_adjust=auto_adjust,
            )
            if temp_df.columns.nlevels != 1:
                temp_df.columns = temp_df.columns.droplevel(1)
            temp_df["ticker"] = tic
            if len(temp_df) > 0:
                # data_df = data_df.append(temp_df)
                data_df = pd.concat([data_df, temp_df], axis=0)
            else:
                num_failures += 1
        if num_failures == len(self.tickers):
            raise ValueError("no data is fetched.")
        # reset the index, we want to use numbers as index instead of dates
        data_df = data_df.reset_index()
        try:
            # Add a new column 'price' to store the original unstandardized adjusted price
            data_df["price"] = data_df["Adj Close"]
            
            # convert the column names to standardized names
            data_df.rename(
                columns={
                    "Date": "date",
                    "Adj Close": "adjcp",
                    "Close": "close",
                    "High": "high",
                    "Low": "low",
                    "Volume": "volume",
                    "Open": "open",
                    "tic": "tic",
                },
                inplace=True,
            )
            
            # use adjusted close price instead of close price
            data_df["close"] = data_df["adjcp"]
            # drop the adjusted close price column
            data_df = data_df.drop(labels="adjcp", axis=1)
        except NotImplementedError:
            print("the features are not supported currently")
        # create day of the week column (monday = 0)
        data_df["day"] = data_df["date"].dt.dayofweek
        # convert date to standard string format, easy to filter
        data_df["date"] = data_df.date.apply(lambda x: x.strftime("%Y-%m-%d"))
        # drop missing data
        data_df = data_df.dropna()
        data_df = data_df.reset_index(drop=True)
        print("Shape of DataFrame: ", data_df.shape)
        # print("Display DataFrame: ", data_df.head())

        data_df = data_df.sort_values(by=["date", "ticker"]).reset_index(drop=True)

        self.df=data_df
    
    def clean_data(self):
        initial_ticker_list = self.df[self.df.date == self.df.date.unique()
                                      [0]]["ticker"].values.tolist()
        initial_ticker_list = set(initial_ticker_list)
        for date in tqdm(self.df.date.unique()):
            ticker_list = self.df[self.df.date ==
                                  date]["ticker"].values.tolist()
            ticker_list = set(ticker_list)
            initial_ticker_list = initial_ticker_list & ticker_list
        df_list = []
        for ticker in initial_ticker_list:
            new_df = self.df[self.df.ticker == ticker]
            df_list.append(new_df)
        df = pd.concat(df_list)
        
        df = df.copy()
        df = df.sort_values(["date", "ticker"], ignore_index=True)
        df.index = df.date.factorize()[0]
        merged_closes = df.pivot_table(index="date",
                                       columns="ticker",
                                       values="close")
        merged_closes = merged_closes.dropna(axis=1)
        tics = merged_closes.columns
        df = df[df.ticker.isin(tics)]
        self.df=df
        
    def add_technical_indicator(self):
        """
        calculate technical indicators
        use stockstats package to add technical inidactors
        :param data: (df) pandas dataframe
        :return: (df) pandas dataframe
        """
        from stockstats import StockDataFrame as Sdf
        df = self.df.copy()
        df = df.sort_values(by=["ticker", "date"])
        stock = Sdf.retype(df.copy())
        unique_ticker = stock.ticker.unique()

        for indicator in self.tech_indicator_list:
            indicator_df = pd.DataFrame()
            for i in range(len(unique_ticker)):
                try:
                    temp_indicator = stock[stock.ticker == unique_ticker[i]][indicator]
                    temp_indicator = pd.DataFrame(temp_indicator)
                    temp_indicator["ticker"] = unique_ticker[i]
                    temp_indicator["date"] = df[df.ticker == unique_ticker[i]][
                        "date"
                    ].to_list()
                    # indicator_df = indicator_df.append(
                    #     temp_indicator, ignore_index=True
                    # )
                    indicator_df = pd.concat(
                        [indicator_df, temp_indicator], axis=0, ignore_index=True
                    )
                except Exception as e:
                    print(e)
            df = df.merge(
                indicator_df[["ticker", "date", indicator]], on=["ticker", "date"], how="left"
            )
        df = df.sort_values(by=["date", "ticker"])
        df = df.ffill().bfill()
        self.df=df
        
    def add_vix(self):
        """
        add vix from yahoo finance
        :param data: (df) pandas dataframe
        :return: (df) pandas dataframe
        """
        df = self.df.copy()
        df_vix = YahooDownloader(
            start_date=df.date.min(), end_date=pd.to_datetime(df.date.max())+pd.Timedelta(days=1), ticker_list=["^VIX"]
        ).fetch_data()
        vix = df_vix[["date", "close"]]
        vix.columns = ["date", "vix"]

        df = df.merge(vix, on="date")
        df = df.sort_values(["date", "ticker"]).reset_index(drop=True)
        self.df=df

    def add_turbulence(self):
        """
        add turbulence index from a precalcualted dataframe
        :param data: (df) pandas dataframe
        :return: (df) pandas dataframe
        """
        df = self.df.copy()
        turbulence_index = self.calculate_turbulence(df)
        df = df.merge(turbulence_index, on="date")
        df = df.sort_values(["date", "ticker"]).reset_index(drop=True)
        self.df= df

    def calculate_turbulence(self, data):
        """calculate turbulence index based on dow 30"""
        # can add other market assets
        df = data.copy()
        df_price_pivot = df.pivot(index="date", columns="ticker", values="close")
        # use returns to calculate turbulence
        df_price_pivot = df_price_pivot.pct_change()

        unique_date = df.date.unique()
        # start after a year
        start = 252
        turbulence_index = [0] * start
        # turbulence_index = [0]
        count = 0
        for i in range(start, len(unique_date)):
            current_price = df_price_pivot[df_price_pivot.index == unique_date[i]]
            # use one year rolling window to calcualte covariance
            hist_price = df_price_pivot[
                (df_price_pivot.index < unique_date[i])
                & (df_price_pivot.index >= unique_date[i - 252])
            ]
            # Drop tickers which has number missing values more than the "oldest" ticker
            filtered_hist_price = hist_price.iloc[
                hist_price.isna().sum().min() :
            ].dropna(axis=1)

            cov_temp = filtered_hist_price.cov()
            current_temp = current_price[[x for x in filtered_hist_price]] - np.mean(
                filtered_hist_price, axis=0
            )
            # cov_temp = hist_price.cov()
            # current_temp=(current_price - np.mean(hist_price,axis=0))

            temp = current_temp.values.dot(np.linalg.pinv(cov_temp)).dot(
                current_temp.values.T
            )
            if temp > 0:
                count += 1
                if count > 2:
                    turbulence_temp = temp[0][0]
                else:
                    # avoid large outlier because of the calculation just begins
                    turbulence_temp = 0
            else:
                turbulence_temp = 0
            turbulence_index.append(turbulence_temp)
        try:
            turbulence_index = pd.DataFrame(
                {"date": df_price_pivot.index, "turbulence": turbulence_index}
            )
        except ValueError:
            raise Exception("Turbulence information could not be added.")
        return turbulence_index
    

    def get_date(self,df):
        date = df.date.unique()
        start_date = pd.to_datetime(date[0])
        end_date = pd.to_datetime(date[-1])
        return start_date, end_date


    def data_split(self,df, start, end, target_date_col="date"):

        data = df[(df[target_date_col] >= start) & (df[target_date_col] < end)]
        data = data.sort_values([target_date_col, "ticker"], ignore_index=True)
        data.index = data[target_date_col].factorize()[0]
        return data

    

    def standardize_features(self, train, valid, test, exclude_columns=["ticker", "date", "price"]):
        """
        Standardize features for each stock individually.
        
        For every ticker in the training set:
        - For OHLC columns ("open", "high", "low", "close"): Divide each value by the maximum of that column (computed on train data).
        - For the volume column ("volume"): Fit a StandardScaler on the training data and transform volume.
        
        Args:
            train (pd.DataFrame): Training data.
            valid (pd.DataFrame): Validation data.
            test (pd.DataFrame): Test data.
            exclude_columns (list): Columns to exclude from scaling.
        
        Returns:
            train, valid, test (pd.DataFrame): Updated DataFrames with standardized features.
            scalers (dict): A dictionary with ticker as key and the fitted volume StandardScaler for that stock.
        """
        # Define the columns to process for OHLC and volume.
        ohlc_columns = ["open", "high", "low", "close"]
        volume_column = "volume"
        
        # For safety, ensure the OHLC and volume columns are float.
        for col in ohlc_columns + [volume_column]:
            train[col] = train[col].astype(np.float64)
            valid[col] = valid[col].astype(np.float64)
            test[col] = test[col].astype(np.float64)
        
        scalers = {}
        
        # Process each ticker in the training data.
        for ticker in train["ticker"].unique():
            # Get data for this ticker from train, valid, and test.
            train_stock = train[train["ticker"] == ticker]
            valid_stock = valid[valid["ticker"] == ticker]
            test_stock  = test[test["ticker"] == ticker]
            
            # Process OHLC columns: Divide by the maximum (computed from training data).
            # If a maximum value is 0, it leaves the column unchanged to avoid division by zero.
            ohlc_max = train_stock[ohlc_columns].max().replace(0, np.nan)
            for col in ohlc_columns:
                # Use the computed maximum for the current column.
                max_val = ohlc_max[col]
                if pd.notna(max_val) and max_val != 0:
                    train.loc[train["ticker"] == ticker, col] = train_stock[col] / max_val
                    valid.loc[valid["ticker"] == ticker, col] = valid_stock[col] / max_val
                    test.loc[test["ticker"] == ticker, col]  = test_stock[col] / max_val
            
            # Process volume column: Standardize via StandardScaler.
            vol_scaler = StandardScaler()
            # Fit the scaler on the training volume data (reshaped to 2D)
            vol_scaler.fit(train_stock[[volume_column]])
            scalers[ticker] = vol_scaler
            
            # Transform the volume column for train, valid, and test.
            train.loc[train["ticker"] == ticker, volume_column] = vol_scaler.transform(train_stock[[volume_column]])
            valid.loc[valid["ticker"] == ticker, volume_column] = vol_scaler.transform(valid_stock[[volume_column]])
            test.loc[test["ticker"] == ticker, volume_column]  = vol_scaler.transform(test_stock[[volume_column]])
        
        return train, valid, test, scalers

    def split(self,data):
        """ split the data by the portion into train, valid, test, which is convinent for the users to do the time rolling
        experiment"""
       
        train = self.data_split(data, TRAIN_START_DATE, TRAIN_END_DATE)
        valid = self.data_split(data, EVAL_START_DATE, EVAL_END_DATE)
        test = self.data_split(data, TEST_START_DATE, TEST_END_DATE)
        
        train_std, valid_std, test_std, scalers = self.standardize_features(train, valid, test, exclude_columns=["ticker", "date","price"])
    
        return train_std, valid_std, test_std, scalers
        


    def normalization(self,portion):
        portion = np.array(portion)
        sum = np.sum(portion)
        portion = portion / sum
        return portion

    def run(self):


        self.download_data()


        self.clean_data()
        if self.feature_mode=="basic":
            self.add_technical_indicator()
        elif self.feature_mode == 'alpha158_novolume':  
            self.df = self.make_alpha()
        elif self.feature_mode == 'alpha158': 
            self.df = self.make_alpha158()
        else: 
            pass
        if self.add_vix:
            self.add_vix()
            self.add_turbulence()
        train, valid, test, scalers = self.split(self.df)
        
        #Modify column names as needed.
        train = train.rename(columns={"ticker": "tic"})
        valid = valid.rename(columns={"ticker": "tic"})
        test = test.rename(columns={"ticker": "tic"})

        train.to_csv(self.train_path)
        valid.to_csv(self.valid_path)
        test.to_csv(self.test_path)

def main():
    yf=YfinancePreprocessor()
    yf.run()
    
    
    
    
    





if __name__ == '__main__':
    main()