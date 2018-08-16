import wordnik
import urllib
from bs4 import BeautifulSoup as HTMLReader

class ScraperAI():
    def __init__(self):
        self.headline_weight = 5
        self.paragraph_weight = 1

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