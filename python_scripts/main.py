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

for company_tag in company_taglist:
    company_data = data[data.ticker == company_tag]

    unmodified_dates = (company_data.date).values.astype('datetime64[D]')
    start_date = unmodified_dates[0]
    date_data  = (unmodified_dates - start_date).astype('float64')

    price_data = (company_data.close).values
    stock_data = np.column_stack((date_data, price_data))

    print (stock_data)
