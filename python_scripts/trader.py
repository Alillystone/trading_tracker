import pandas as pd
import datetime
import data_client as DataClient
import file_system as FileSystem
import data_analyser as DataAnalyser
import date_formatter as DateFormatter
import internet_scraper as InternetScraper

class TraderAI:

    def __init__(self):
        self.CLIENT = DataClient.configure_client()
        self.use_scraper  = False
        self.use_analyser = True

    def initialise(self):
        
        self.create_ticker_dataframe()

        if (self.use_scraper):
            self.SCRAPER  = InternetScraper.ScraperAI()
            self.SCRAPER.initialise(self.df_ticker_info)

        if (self.use_analyser):
            self.ANALYSER = DataAnalyser.DataAnalyser(self.CLIENT)
            self.ANALYSER.initialise(self.df_ticker_info)
    
    def create_ticker_dataframe(self):

        self.csv_folder = 'csvs/'
        ticker_file = self.csv_folder + 'tickers.csv'

        if not (FileSystem.file_exists(ticker_file)):
            if not (FileSystem.directory_exists(self.csv_folder)):
                FileSystem.create_directory(self.csv_folder)
            df_ticker_info = pd.DataFrame(DataClient.download_tickers(self.CLIENT))
            df_ticker_info.to_csv(ticker_file)
        else:
            df_ticker_info = pd.read_csv(ticker_file)
        
        df_ticker_info = df_ticker_info.loc[
                        (df_ticker_info.exchange.isin(['NASDAQ',
                                                       'NYSE',
                                                       'NYSE ARCA']))]

        current_date = DateFormatter.current_date()
        week_date    = DateFormatter.convert_date_to_string(
                                     current_date - datetime.timedelta(days=7))
        year_date    = DateFormatter.convert_date_to_string(
                                     current_date - datetime.timedelta(days=365))
        self.df_ticker_info = df_ticker_info[
                             (df_ticker_info.startDate < year_date) &
                             (df_ticker_info.endDate > week_date)]

    def live(self):
        alive = True
        while (alive):
            self.SCRAPER.increment()