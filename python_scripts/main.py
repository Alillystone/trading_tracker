import quandl
import numpy as np
from requests import get
from requests.exceptions import RequestException
from contextlib import closing
from bs4 import BeautifulSoup

APIKEY = "5V48hxsMxjK57i7gsNAT"
quandl.ApiConfig.api_key = APIKEY
company_taglist = ['AAPL', 'MSFT', 'WMT']

data = quandl.get_table('WIKI/PRICES', 
                        qopts = { 'columns': ['ticker', 'date', 'close'] }, 
                        ticker = company_taglist, 
                        date = { 'gte': '2018-01-01', 'lte': '2018-05-01' },
                        paginate=True)

predictability_responses = []
for company_tag in company_taglist:
    company_data = data[data.ticker == company_tag]

    unmodified_dates = (company_data.date).values.astype('datetime64[D]')
    start_date = unmodified_dates[0]
    date_data  = (unmodified_dates - start_date).astype('float64')

    price_data = (company_data.close).values
    stock_data = np.column_stack((date_data, price_data))

    num_items      = len(stock_data)
    prev_idx = num_items - 1
    next_idx = num_items - 2
    buffer_width   = 5

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
    
    predictability_responses.append((company_tag, predictability_response))