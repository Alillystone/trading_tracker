import requests
from PyDictionary import PyDictionary
from bs4 import BeautifulSoup
from googlesearch import search 

dictionary = PyDictionary()
for url in search('AMD news', stop=20):
    request = requests.get(url)
    data = request.text
    HTML = BeautifulSoup(data,'lxml')
    type1_headers = HTML.find_all('h1', itemprop='name headline')
    type2_headers = HTML.find_all('h1', itemprop='headline')
    for header in type1_headers:
        word_array = header.text.split()
        for word in word_array:
            # print (word)
            # print (dictionary.meaning(word))
            meaning = dictionary.meaning('was')
            print (meaning)
            print (len(meaning['Noun']))
            print (len(meaning['Verb']))
            print (len(meaning['Adjective']))
            exit()
    #     try:
    #         print (header.text)
    #     except:
    #         continue
    # for header in type2_headers:
    #     try:
    #         print (header.text)
    #     except:
    #         continue 

