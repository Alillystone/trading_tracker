import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import csv
from tiingo import TiingoClient

# csvs for each state, initialisation (i.e each period), then a live CSV
#

def ensure_directory_exists(directory):
    if (not os.path.exists(directory)):
        os.makedirs(directory)

class TraderAI:

    def __init__(self, CLIENT):
        self.CLIENT = CLIENT
        self.company_info = CLIENT.list_stock_tickers()

    def initialise(self):

        # self.SCRAPER  = ScraperAI()
        # self.SCRAPER.initialise()

        self.ANALYSER = AnalysisAI(self.CLIENT)
        self.ANALYSER.initialise(self.company_info[29000:29300])
        

    def live(self):
        alive = True
        while (alive):
            self.SCRAPER.increment()

class ScraperAI():
    def __init__(self):
        self.hi = "hi" # Have variables tracked accumulated variables

class AnalysisAI():

    def __init__(self, CLIENT):

        self.CLIENT =   CLIENT

        subfolder = "csvs/"
        headers   = [("ticker_symbol", "FFT_response", "differential_response")]
        self.week_filename    = subfolder + "week_data.csv"
        self.month_filename   = subfolder + "month_data.csv"
        self.quarter_filename = subfolder + "quarter_data.csv"
        self.year_filename    = subfolder + "year_data.csv"
        self.lustrum_filename = subfolder + "lustrum_data.csv"

        ensure_directory_exists(subfolder)

        iter_list = []
        iter_list.append(self.week_filename)
        iter_list.append(self.month_filename)
        iter_list.append(self.quarter_filename)
        iter_list.append(self.year_filename)
        iter_list.append(self.lustrum_filename)

        for filename in iter_list:
            with open(filename, 'a') as csvfile:
                writer = csv.writer(csvfile, delimiter=str(','))
                writer.writerows(headers)
        # Have variables tracked accumulated variables

    def initialise(self, company_info):

        current_date = datetime.datetime.today()
        week_date    = current_date - datetime.timedelta(days=7)
        month_date   = current_date - datetime.timedelta(days=31)
        quarter_date = current_date - datetime.timedelta(days=93)
        yearly_date  = current_date - datetime.timedelta(days=365)
        lustrum_date = current_date - datetime.timedelta(days=1826)

        date_list = []
        date_list.append((week_date,    
                          self.week_filename))
        date_list.append((month_date,   
                          self.month_filename))
        date_list.append((quarter_date, 
                          self.quarter_filename))
        date_list.append((yearly_date,
                          self.year_filename))
        date_list.append((lustrum_date,
                          self.lustrum_filename))
        current_date = current_date.strftime('%Y-%m-%d')

        for ordereddict in company_info:
            info = list(ordereddict.items())
            ticker_symbol       = info[0][1]
            stock_start_date    = info[4][1]
            if (not len(stock_start_date)):
                stock_start_date = '2000-01-01'
            stock_start_date = datetime.datetime.strptime(stock_start_date, 
                                                          '%Y-%m-%d')

            for start_date, filename in date_list:
                
                if (start_date < stock_start_date):
                    start_date = stock_start_date

                start_date = start_date.strftime('%Y-%m-%d')

                try:
                    data = self.CLIENT.get_dataframe(ticker_symbol,
                                                     startDate = start_date,
                                                     endDate = current_date)
                except:
                    msg = ("Could not retrieve data for ticker " 
                            + ticker_symbol +".")
                    print (msg)
                    continue

                with open(filename, 'a') as csvfile:
                    writer = csv.writer(csvfile, delimiter=str(','))
                    if (len(data) < 3):
                        output = [(ticker_symbol, 0.0, 0.0)]
                        writer.writerows(output)
                    else:
                        close_stock_price = (data.adjClose).values

                        buffer_width = 3
                        FFT_response = self.calculate_FFT(close_stock_price, 
                                                          buffer_width)
                        diff_response = self.simple_diff(close_stock_price)

                        output = [(ticker_symbol, FFT_response, diff_response)] 
                        writer.writerows(output)
                        
    def calculate_FFT(self, data, buffer_width):
        num_items    = len(data)
        prev_idx     = num_items - 1
        next_idx     = num_items - 2

        response_accumulation = 0.0
        while (next_idx >= buffer_width):

            prev_stock_slice = data[prev_idx - buffer_width:prev_idx]
            next_stock_slice = data[next_idx - buffer_width:next_idx]

            prev_fft = np.fft.fft(prev_stock_slice)
            next_fft = np.fft.fft(next_stock_slice)

            prev_response = np.angle(prev_fft)
            next_response = np.angle(next_fft)

            predictability_response  = np.sum(np.abs(prev_response - 
                                                     next_response))
            response_accumulation += predictability_response
            prev_idx -= 1
            next_idx -= 1

        return response_accumulation

    def simple_diff(self, data):

        start_price = (data[0] + 
                       data[1] + 
                       data[2]) / 3

        end_price   = (data[-1] + 
                       data[-2] + 
                       data[-3]) / 3

        differential = (end_price - start_price) / start_price
        differential *= 100

        return differential

    def increment():
        self.hi = "hi"
        # retrieve most up to date stock data 

def main():
    config = {}
    config['session'] = True
    config['api_key'] = "4069fe9d132ea2d8cf8b50cedbc4b0d6d764e5eb"

    client = TiingoClient(config)
    TRADER = TraderAI(client)
    TRADER.initialise()

if __name__ == "__main__":
    main()

