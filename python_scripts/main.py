import trader
import sentiment_analyser
import matplotlib.pyplot as plt
from tiingo import TiingoClient

def main():

    # SentimentAnalyser = sentiment_analyser.SentimentAnalyser(min_neurons=1000,
    #                                                          max_neurons=2000,
    #                                                          num_hidden_layers=3)
    # SentimentAnalyser.prepare_training_data()
    # SentimentAnalyser.train_neural_network()

    config = {}
    config['session'] = True
    config['api_key'] = "4069fe9d132ea2d8cf8b50cedbc4b0d6d764e5eb"

    client = TiingoClient(config)
    TRADER = trader.TraderAI(client)
    TRADER.initialise()

if __name__ == "__main__":
    main()

