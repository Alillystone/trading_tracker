from tiingo import TiingoClient

def configure_client():
    config = {}
    config['session'] = True
    config['api_key'] = "4069fe9d132ea2d8cf8b50cedbc4b0d6d764e5eb"

    return TiingoClient(config)

def download_tickers(CLIENT):
    return CLIENT.list_stock_tickers()

def download_stock_data(CLIENT, ticker, start, end):
    successful = True
    df_ticker = None
    try:
        df_ticker = CLIENT.get_dataframe(ticker,
                                         startDate = start,
                                         endDate   = end)
    except:
        successful = False

    return successful, df_ticker
        