import csv
import time
import datetime
import numpy as np
import pandas as pd
from tiingo import TiingoClient

# create web_client to pull stock data
config = {}
config['session'] = True
config['api_key'] = "4069fe9d132ea2d8cf8b50cedbc4b0d6d764e5eb"
web_client = TiingoClient(config)

# download list of ticker symbols and other details
df_ticker_list = pd.DataFrame(web_client.list_stock_tickers())

# select tickers only found in the NASDAQ exchange
df_nasdaq_list = df_ticker_list.loc[(df_ticker_list.exchange.isin(['NASDAQ']))]

# create date parameters to pass into web client, in this case only 1 day to
# reduce data consumption. This may lead to stocks not updated recently to not
# be downloaded, however the big companies should be.
current_date = datetime.datetime.now()
yesterday    = current_date - datetime.timedelta(days=1)

# list to hold ticker and corresponding price
price_list = []

# external counter to approximate data consumption
company_counter = 1

# loop over every ticker
for index, row in df_nasdaq_list.iterrows():
    symbol = row['ticker']
    successful = True
    df_ticker = None

    # every 500 company data downloads, pause for an hour to reset data limit
    if ((company_counter % 500) == 0):
        msg = ("Company counter at " + str(company_counter) + 
               "/" + str(len(df_nasdaq_list)))
        print (msg)
        time.sleep(3605)

    # pull stock data for current ticker symbol between dates specified above
    try:
        df_ticker = web_client.get_dataframe(symbol,
                                     startDate = yesterday,
                                     endDate   = current_date)
    except:
        # if unsuccesfully downloaded for some reason, move onto next ticker
        continue

    # validity check to ensure data correctly parsed
    if (df_ticker is not None):
        # extract array of adjusted close values
        close_stock_price = (df_ticker.adjClose).values

        # take the latest stock price (i.e last item in array) and create
        # tuple with symbol
        company_data = (symbol, close_stock_price[-1])
        price_list.append(company_data)

    company_counter += 1

# sort list into descending order
sorted_list = sorted(top_list,key=lambda l:l[1], reverse=True)

# write to file
filename = "top_symbols.csv"
with open(filename, 'a') as csvfile:
    writer = csv.writer(csvfile, delimiter=str(','))
    for symbol_data in sorted_list:
        writer.writerow(symbol_data)








