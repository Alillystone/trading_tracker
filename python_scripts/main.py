import quandl
import numpy as np
from collections import namedtuple
from requests import get
from requests.exceptions import RequestException
from contextlib import closing
from bs4 import BeautifulSoup

class CompanyAnalysis:

    def __init__(self, tag, confidence, predictability):
        self.ticker = tag
        self.confidence = confidence
        self.predictability = predictability

    def write_properties(self):
        print ('--------------')
        print ('Ticker = ', self.ticker)
        print ('Confidence = ', self.confidence)
        print ('Predictability = ', self.predictability)

APIKEY = "5V48hxsMxjK57i7gsNAT"
quandl.ApiConfig.api_key = APIKEY
company_taglist = ['AAPL', 'MSFT', 'WMT', 'FB', 'MU', 'BAC', 'F', 'GE', 'GM', 'CHK']

data = quandl.get_table('WIKI/PRICES', 
                        qopts = { 'columns': ['ticker', 'date', 'close'] }, 
                        ticker = company_taglist, 
                        date = { 'gte': '2018-01-01', 'lte': '2018-05-01' },
                        paginate=True)

company_analysis_data = []
Company = namedtuple('Company', 'tag data fft_response simple_differential accumulated_differential')

FFT_list = []
sim_list = []
dif_list = []
for company_tag in company_taglist:
    company_data = data[data.ticker == company_tag]

    unmodified_dates = (company_data.date).values.astype('datetime64[D]')
    start_date = unmodified_dates[0]
    date_data  = (unmodified_dates - start_date).astype('float64')

    price_data = (company_data.close).values
    stock_data = np.column_stack((date_data, price_data))

    num_items    = len(stock_data)
    prev_idx     = num_items - 1
    next_idx     = num_items - 2
    buffer_width = 5

    response_accumulation = 0.0
    while (next_idx >= buffer_width):

        prev_stock_slice = price_data[prev_idx - buffer_width:prev_idx]
        next_stock_slice = price_data[next_idx - buffer_width:next_idx]

        prev_fft = np.fft.fft(prev_stock_slice)
        next_fft = np.fft.fft(next_stock_slice)

        prev_response = np.angle(prev_fft)
        next_response = np.angle(next_fft)

        predictability_response  = np.sum(np.abs(prev_response - 
                                                 next_response))
        response_accumulation += predictability_response
        prev_idx -= 1
        next_idx -= 1

    current_price_avg   = (price_data[-1] + 
                           price_data[-2] + 
                           price_data[-3]) / 3

    start_price_avg     = (price_data[0] + 
                           price_data[1] + 
                           price_data[2]) / 3

    simple_differential = ((current_price_avg - start_price_avg) / 
                           (current_price_avg))

    prev_idx = num_items - 1
    next_idx = num_items - 2
    accumulated_differential = 0.0
    while (next_idx > 0):
        prev_stock = price_data[prev_idx]
        next_stock = price_data[next_idx]
        prev_date  = date_data[prev_idx]
        next_date  = date_data[next_idx]

        price_diff = (prev_stock - next_stock) / (prev_date - next_date) 
        accumulated_differential += ((price_diff * prev_idx) / 
                                     (prev_stock))

        prev_idx -= 1
        next_idx -= 1

    company_instance = Company(company_tag,
                               company_data,
                               response_accumulation * -1, 
                               simple_differential, 
                               accumulated_differential)

    FFT_list.append(response_accumulation * -1)
    sim_list.append(simple_differential)
    dif_list.append(accumulated_differential)

    company_analysis_data.append(company_instance)

FFT_min   = min(FFT_list)
FFT_range = max(FFT_list) - min(FFT_list)

sim_min   = min(sim_list)
sim_range = max(sim_list) - min(sim_list)

dif_min   = min(dif_list)
dif_range = max(dif_list) - min(dif_list)

companies = []
for company in company_analysis_data:
    predictability = (company.fft_response - FFT_min) / FFT_range
    sim_confidence = (company.simple_differential - sim_min) / sim_range
    dif_confidence = (company.accumulated_differential - dif_min) / dif_range
    confidence = (sim_confidence + dif_confidence) / 2

    company_object = CompanyAnalysis(company.tag, confidence, predictability)
    companies.append(company_object)

print (companies)

