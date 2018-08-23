import trader
import sentiment_analyser
import matplotlib.pyplot as plt

def main():

    # SentimentAnalyser = sentiment_analyser.SentimentAnalyser(min_neurons=1000,
    #                                                          max_neurons=2000,
    #                                                          num_hidden_layers=3)
    # SentimentAnalyser.prepare_training_data()
    # SentimentAnalyser.train_neural_network()
    
    TRADER = trader.TraderAI()
    TRADER.initialise()

if __name__ == "__main__":
    main()

