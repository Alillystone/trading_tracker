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

        self.csv_folder = "csvs/"
        ensure_directory_exists(self.csv_folder)

        self.create_metric_dataframe()
        self.create_dates_dataframe()
        self.write_csv_headers()

    def create_metric_dataframe(self):
        metric_data = {'metric_label' : ['percentage_change',
                                         'stability_deviation',
                                         'fft_predictability',
                                         'mean_stock_difference',
                                         'var_stock_difference'],
                       'metric_rank_weight' : [5,
                                               2,
                                               1,
                                               4,
                                               1],
                       'higher' : [False,
                                   True,
                                   True,
                                   False,
                                   True]}
        self.df_metrics = pd.DataFrame(data=metric_data)

    def create_dates_dataframe(self):
        self.current_date = retrieve_current_date()
        week_date    = self.current_date - datetime.timedelta(days=7)
        month_date   = self.current_date - datetime.timedelta(days=31)
        quarter_date = self.current_date - datetime.timedelta(days=93)
        yearly_date  = self.current_date - datetime.timedelta(days=365)
        lustrum_date = self.current_date - datetime.timedelta(days=1826)

        date_data = {'date_label' : ['current',
                                     'week',
                                     'month',
                                     'quarter',
                                     'year',
                                     'lustrum'], 
                     'date' : [self.current_date,
                               week_date,
                               month_date,
                               quarter_date,
                               yearly_date,
                               lustrum_date],
                     'date_filename' : [self.csv_folder + 'current.csv',
                                        self.csv_folder + 'week.csv',
                                        self.csv_folder + 'month.csv',
                                        self.csv_folder + 'quarter.csv',
                                        self.csv_folder + 'yearly.csv',
                                        self.csv_folder + 'lustrum.csv']}
        self.df_dates = pd.DataFrame(data=date_data)

    def write_csv_headers(self):

        metric_headers = self.df_metrics.metric_label.values.tolist()
        metric_headers = ['ticker_symbol'] + metric_headers
        files = self.df_dates.date_filename.values
        for filename in files:
            with open(filename, 'a') as csvfile:
                writer = csv.writer(csvfile, delimiter=str(',')) 
                writer.writerow(metric_headers)

    def initialise(self, company_info):

        self.company_info = company_info[20:40]

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

            for index, row in self.df_dates.iterrows():

                start_date = row['date']
                filename = row['date_filename']

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
        self.initialise_live_dataframe()
    
    def initialise_live_dataframe(self):

        month_filename = self.df_dates.date_filename[
                         self.df_dates.date_label == 'month'].tolist()[0]
        df_all = pd.read_csv(month_filename, sep = ",", error_bad_lines=False)

        num_data_points = len(df_all)
        rank_label = 'rank'
        df_all[rank_label] = pd.Series(np.zeros(num_data_points))

        for index, row in self.df_metrics.iterrows():
            sort_type     = row['higher']
            metric_label  = row['metric_label']
            metric_weight = row['metric_rank_weight']
            df_metric = (df_all.sort_values(
                         by=[metric_label], 
                         ascending=sort_type)).reset_index(drop=True)
            for index, row in df_metric.iterrows():
                symbol_idx = df_all.index[
                             df_all.ticker_symbol == row['ticker_symbol']].tolist()[0]
                df_all.loc[symbol_idx, rank_label] += index * metric_weight
                
        self.df_live = df_all.sort_values(by=[rank_label]).reset_index(drop=True)
        print (self.df_live)

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

        stock_mean = np.mean(data)
        diff_data = np.diff(data)
        mean = np.mean(diff_data) / stock_mean
        var = np.var(diff_data) / stock_mean

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

