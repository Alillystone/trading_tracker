from bs4 import BeautifulSoup as HTMLReader
import requests
from PyDictionary import PyDictionary
from googlesearch import search 

class InternetScrape:
    def __init__(self):
        self.google_url_limit = 20

    def get_headlines(self, ticker_symbol):

        search_string = ticker_symbol + " stock news"
        for url in search('AMD stock news', stop=self.google_url_limit):
            try:
                request = requests.get(url)
            except:
                continue
            data = request.text
            HTML = BeautifulSoup(data,'lxml')

            headers   = []
            headers.append(HTML.find_all('h1'))
            headers.append(HTML.find_all('h1', itemprop='name headline'))
            headers.append(HTML.find_all('h1', itemprop='headline'))

            headlines = []
            for header in headers:
                for headline in header:
                    headlines.append(headline.text)

        return headlines