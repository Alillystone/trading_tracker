import datetime
import os
import csv
import urllib.request
import shutil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup as HTMLReader
from tiingo import TiingoClient

# csvs for each state, initialisation (i.e each period), then a live CSV
#

def ensure_directory_exists(directory):
    if (not os.path.exists(directory)):
        os.makedirs(directory)
    else:
        shutil.rmtree(directory)
        os.makedirs(directory)

def retrieve_current_date():
    return datetime.datetime.today()

class TraderAI:

    def __init__(self, CLIENT):
        self.CLIENT = CLIENT
        self.use_scraper  = False
        self.use_analyser = True

    def initialise(self):

        company_info = self.CLIENT.list_stock_tickers()
        self.company_info = []
        for ordereddict in company_info:
            info = list(ordereddict.items())
            ticker_symbol   = info[0][1]
            exchange_symbol = info[1][1]
            start_date      = info[4][1]

            if (exchange_symbol == 'NYSE' or
                exchange_symbol == 'LSE' or
                exchange_symbol == 'NASDAQ') :

                company_data = (ticker_symbol, exchange_symbol, start_date)
                self.company_info.append(company_data)
        
        if (self.use_scraper):
            self.SCRAPER  = ScraperAI()
            self.SCRAPER.initialise(self.company_info)

        if (self.use_analyser):
            self.ANALYSER = AnalysisAI(self.CLIENT)
            self.ANALYSER.initialise(self.company_info)
        

    def live(self):
        alive = True
        while (alive):
            self.SCRAPER.increment()

class ScraperAI():
    def __init__(self, CLIENT):
        self.CLIENT = CLIENT
        self.headline_weight = 5
        self.paragraph_weight = 1

        apiURL = 'http://api.wordnik.com/v4'
        apikey = 'MY API KEY'
        wordclient = swagger.ApiClient(apikey, apiURL)
        self.DICTIONARY = WordApi.WordApi(wordclient)

        # Create ordered dicts for words: tone, multuplier, base_words
        self.grammer_types = ["verb",
                              "adverb",
                              "adjective",
                              "noun",
                              "preposition"]

    def initialise(self, company_info):

        url = "https://www.bloomberg.com/quote/AMD:US"
        page = urllib.request.urlopen(url)
        HTML = HTMLReader(page, "lxml")
        article_links = HTML.find_all("article")

        for article_link in article_links:
            article_tone = self.determine_article_tone(article_link)
            

    def determine_tone(self, word_list):
        print ("------")
        for word in word_list:
            print (word)
            print (self.DICTIONARY.getDefinitions(word))

    def determine_article_tone(self, article_link):
        article_parameters = article_link.find_all("a")
        for parameter in article_parameters:
            article_url  = parameter.get("href")
            article = urllib.request.urlopen(article_url)
            article_HTML = HTMLReader(article, "lxml")

            article_paragraphs = article_HTML.find_all("p")
            article_headlines  = article_HTML.find_all("h1")
            
            for headlines in article_headlines:
                headline = headlines.text
                headline_word_list = headline.split()

                response = self.determine_tone(headline_word_list)

class AnalysisAI():

    def __init__(self, CLIENT):

        self.CLIENT =   CLIENT

        subfolder = "csvs/"
        headers   = [("ticker_symbol", 
                      "trend", 
                      "deviation",
                      "predictability",
                      "mean_differential",
                      "var_differential")]

        self.week_filename    = subfolder + "week_data.csv"
        self.month_filename   = subfolder + "month_data.csv"
        self.quarter_filename = subfolder + "quarter_data.csv"
        self.year_filename    = subfolder + "year_data.csv"
        self.lustrum_filename = subfolder + "lustrum_data.csv"

        ensure_directory_exists(subfolder)

        iter_list = [self.week_filename,
                     self.month_filename,
                     self.quarter_filename,
                     self.year_filename,
                     self.lustrum_filename]

        for filename in iter_list:
            with open(filename, 'a') as csvfile:
                writer = csv.writer(csvfile, delimiter=str(','))
                writer.writerows(headers)

    def initialise(self, company_info):

        self.company_info = company_info[20:40]

        date_list = self.retrieve_dates_filenames_list()
        current_date = self.current_date.strftime('%Y-%m-%d')
        
        for company_info in self.company_info:

            ticker_symbol       = company_info[0]
            stock_start_date    = company_info[2]

            if (not len(stock_start_date)):
                stock_start_date = '2000-01-01'

            stock_start_date = datetime.datetime.strptime(stock_start_date, 
                                                          '%Y-%m-%d')

            try:
                df_ticker = self.CLIENT.get_dataframe(ticker_symbol,
                                                 startDate = stock_start_date,
                                                 endDate =   self.current_date)
            except:
                msg = ("Could not retrieve data for ticker " 
                        + ticker_symbol +".")
                print (msg)
                continue 

            for start_date, filename in date_list:

                df_analysis_data = df_ticker[df_ticker.index.to_pydatetime() > start_date]

                with open(filename, 'a') as csvfile:
                    writer = csv.writer(csvfile, delimiter=str(','))
                    if (len(df_analysis_data) > 3):
                        close_stock_price = (df_analysis_data.adjClose).values

                        percentage_change   = self.trend(
                                                           close_stock_price)
                        deviation           = self.deviation(
                                                           close_stock_price)
                        predictability      = self.predictability(
                                                           close_stock_price)
                        mean_diff, var_diff = self.differentials(
                                                           close_stock_price)

                        output = [(ticker_symbol, 
                                   percentage_change,
                                   deviation,
                                   predictability, 
                                   mean_diff,
                                   var_diff)]

                        writer.writerows(output)

        self.date_of_initialisation = self.current_date
        self.initialise_live_dataframe(date_list)
    
    def initialise_live_dataframe(self, date_list):

        df_all = pd.DataFrame()
        with open(self.week_filename, 'rb') as datafile:
            df_all = pd.read_csv(self.month_filename, sep = ",")

        df_trend          = df_all.sort_values(by=['trend'], ascending=False)
        df_deviation      = df_all.sort_values(by=['deviation'])
        df_predictability = df_all.sort_values(by=['predictability'])
        df_differential   = df_all.sort_values(by=['mean_differential'], ascending=False)
        
        data_frame_list = []
        data_frame_list.append((df_trend.reset_index(drop=True),5))
        data_frame_list.append((df_deviation.reset_index(drop=True),2))
        data_frame_list.append((df_differential.reset_index(drop=True),1))
        data_frame_list.append((df_predictability.reset_index(drop=True),4))
        num_analysis_methods = len(data_frame_list)

        tickers = np.asarray(df_all.ticker_symbol)
        num_tickers = len(tickers)
        df_all['rank'] = pd.Series(np.zeros(num_tickers))

        for data_frame, weight in data_frame_list:
            for index, row in data_frame.iterrows():
                symbol_idx = df_all.index[df_all.ticker_symbol == row[0]].tolist()[0]
                df_all.loc[symbol_idx, 'rank'] += index

        self.df_live = df_all.sort_values(by=['rank']).reset_index(drop=True)
        print (self.df_live)

    def retrieve_dates_filenames_list(self):

        self.current_date = retrieve_current_date()
        self.week_date    = self.current_date - datetime.timedelta(days=7)
        self.month_date   = self.current_date - datetime.timedelta(days=31)
        self.quarter_date = self.current_date - datetime.timedelta(days=93)
        self.yearly_date  = self.current_date - datetime.timedelta(days=365)
        self.lustrum_date = self.current_date - datetime.timedelta(days=1826)

        date_list = []
        date_list.append((self.week_date,    
                          self.week_filename))
        date_list.append((self.month_date,   
                          self.month_filename))
        date_list.append((self.quarter_date, 
                          self.quarter_filename))
        date_list.append((self.yearly_date,
                          self.year_filename))
        date_list.append((self.lustrum_date,
                          self.lustrum_filename))

        return date_list

    def increment(self, relevant_tickers):
        
        for ticker in relevant_tickers:
            self.hi = 0
        # retrieve most up to date stock data

    def predictability(self, data):
        num_items    = len(data)
        prev_idx     = num_items - 1
        next_idx     = num_items - 2

        buffer_width = 3
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

    def trend(self, data):

        start_price = (data[0] + 
                       data[1] + 
                       data[2]) / 3

        end_price   = (data[-1] + 
                       data[-2] + 
                       data[-3]) / 3

        differential = (end_price - start_price) / start_price
        differential *= 100

        return differential

    def deviation(self, data):

        deviation_response = 0.0

        num_data_points = len(data)
        start_price     = data[0]
        end_price       = data[-1]
        gradient        = (end_price - start_price) / num_data_points
        median          = (start_price + end_price) / 2

        if (gradient == 0.0):
            return deviation_response

        model = np.arange(start_price, 
                          end_price, 
                          gradient)

        model.resize(num_data_points)

        deviation_response = np.sum(np.abs(model - data))
        deviation_response /= median
        deviation_response /= num_data_points

        return deviation_response

    def differentials(self, data):

        diff_data = np.diff(data)
        mean = np.mean(diff_data)
        var = np.var(diff_data)

        return mean, var

def main():
    config = {}
    config['session'] = True
    config['api_key'] = "4069fe9d132ea2d8cf8b50cedbc4b0d6d764e5eb"

    client = TiingoClient(config)
    TRADER = TraderAI(client)
    TRADER.initialise()

if __name__ == "__main__":
    main()

