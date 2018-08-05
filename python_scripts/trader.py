import stock_analysis
import language_analysis
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
            self.SCRAPER  = language_analysis.ScraperAI()
            self.SCRAPER.initialise(self.company_info)

        if (self.use_analyser):
            self.ANALYSER = stock_analysis.AnalysisAI(self.CLIENT)
            self.ANALYSER.initialise(self.company_info)
        

    def live(self):
        alive = True
        while (alive):
            self.SCRAPER.increment()