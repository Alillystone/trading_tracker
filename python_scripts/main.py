import trader
import matplotlib.pyplot as plt
from tiingo import TiingoClient

def main():
    config = {}
    config['session'] = True
    config['api_key'] = "4069fe9d132ea2d8cf8b50cedbc4b0d6d764e5eb"

    client = TiingoClient(config)
    TRADER = trader.TraderAI(client)
    TRADER.initialise()

if __name__ == "__main__":
    main()

