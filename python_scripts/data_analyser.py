import datetime
import csv
import pandas as pd
import numpy as np
import data_client as DataClient
import file_system as FileSystem
import date_formatter as DateFormatter

class DataAnalyser:

    def __init__(self, CLIENT):

        self.CLIENT = CLIENT
        if not (FileSystem.directory_exists(FileSystem.csv_folder)):
            FileSystem.create_directory(FileSystem.csv_folder)

        if not (FileSystem.directory_exists(FileSystem.ticker_data_folder)):
            FileSystem.create_directory(FileSystem.ticker_data_folder)

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
        self.current_date = DateFormatter.current_date()
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
                     'date_filename' : [FileSystem.csv_folder + 'current.csv',
                                        FileSystem.csv_folder + 'week.csv',
                                        FileSystem.csv_folder + 'month.csv',
                                        FileSystem.csv_folder + 'quarter.csv',
                                        FileSystem.csv_folder + 'yearly.csv',
                                        FileSystem.csv_folder + 'lustrum.csv']}
        self.df_dates = pd.DataFrame(data=date_data)

    def write_csv_headers(self):
        metric_headers = self.df_metrics.metric_label.values.tolist()
        metric_headers = ['ticker_symbol'] + metric_headers
        files = self.df_dates.date_filename.values
        for filename in files:
            if (FileSystem.file_exists(filename)):
                FileSystem.delete_file(filename)
            with open(filename, 'a') as csvfile:
                writer = csv.writer(csvfile, delimiter=str(',')) 
                writer.writerow(metric_headers)

    def initialise(self, df_ticker_info):
        self.df_ticker_info = df_ticker_info
        for ticker_index, ticker_row in self.df_ticker_info.iterrows():
            ticker_symbol    = ticker_row['ticker']
            stock_start_date = ticker_row['startDate']
            ticker_csv = FileSystem.ticker_data_folder + ticker_symbol + ".csv"
            
            df_ticker = pd.DataFrame()
            if (FileSystem.file_exists(ticker_csv)):
                df_ticker = pd.read_csv(ticker_csv,sep=",",
                                        parse_dates=['date'],
                                        index_col=['date'])
            else:
                successful, df_ticker = DataClient.download_stock_data(self.CLIENT,
                                                                       ticker_symbol,
                                                                       stock_start_date, 
                                                                       self.current_date)
                if not successful:
                    continue

                df_ticker.to_csv(ticker_csv,sep=",")

            for index, row in self.df_dates.iterrows():

                start_date = row['date']
                filename   = row['date_filename']

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
                         self.df_dates.date_label == 'year'].tolist()[0]
        df_all = pd.read_csv(month_filename,sep=",")

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
        ticker_order = self.df_live.ticker_symbol.values

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