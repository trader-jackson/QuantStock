from YahooFinance import YfinancePreprocessor
from config_tickers import DOW_30_TICKER, SSE_50_TICKER, NAS_100_TICKER
from config import TRAIN_START_DATE, TRAIN_END_DATE, EVAL_START_DATE, EVAL_END_DATE, TEST_START_DATE, TEST_END_DATE

def download_stock_data():
    # Download DOW30 data
    print("Downloading DOW30 data...")
    dow30_processor = YfinancePreprocessor(
        dataset_name="DOW30",
        start_date=TRAIN_START_DATE,
        end_date=TEST_END_DATE,
        train_valid_test_portion=[0.8, 0.1, 0.1]
    )
    dow30_processor.run()

    # Download NASQ100 data
    print("\nDownloading NASQ100 data...")
    nasq100_processor = YfinancePreprocessor(
        dataset_name="NASQ100",
        start_date=TRAIN_START_DATE,
        end_date=TEST_END_DATE,
        train_valid_test_portion=[0.8, 0.1, 0.1]
    )
    nasq100_processor.run()

    # Download SSE50 data
    print("\nDownloading SSE50 data...")
    sse50_processor = YfinancePreprocessor(
        dataset_name="SSE50",
        start_date=TRAIN_START_DATE,
        end_date=TEST_END_DATE,
        train_valid_test_portion=[0.8, 0.1, 0.1]
    )
    sse50_processor.run()

if __name__ == "__main__":
    download_stock_data() 