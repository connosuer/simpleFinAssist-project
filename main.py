import os
import json
import random
import pickle
import nltk
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader as web
import mplfinance as mpf
import sys
import datetime as dt
import yfinance as yf

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)

class BasicAssistant:
    def __init__(self, intents_file, method_mappings, model_name="basic_model"):
        self.intents = json.load(open(intents_file))
        self.method_mappings = method_mappings
        self.model_name = model_name
        self.words = []
        self.classes = []
        self.documents = []
        self.ignore_letters = ['?', '!', '.', ',']
        self.lemmatizer = nltk.stem.WordNetLemmatizer()

        for intent in self.intents['intents']:
            for pattern in intent['patterns']:
                word_list = nltk.word_tokenize(pattern)
                self.words.extend(word_list)
                self.documents.append((word_list, intent['tag']))
                if intent['tag'] not in self.classes:
                    self.classes.append(intent['tag'])

        self.words = [self.lemmatizer.lemmatize(word) for word in self.words if word not in self.ignore_letters]
        self.words = sorted(set(self.words))
        self.classes = sorted(set(self.classes))

    def _prepare_training_data(self):
        training = []
        output_empty = [0] * len(self.classes)

        for document in self.documents:
            bag = []
            word_patterns = document[0]
            word_patterns = [self.lemmatizer.lemmatize(word.lower()) for word in word_patterns]
            for word in self.words:
                bag.append(1) if word in word_patterns else bag.append(0)

            output_row = list(output_empty)
            output_row[self.classes.index(document[1])] = 1
            training.append([bag, output_row])

        random.shuffle(training)
        training = np.array(training, dtype=object)

        train_x = np.array([item[0] for item in training])
        train_y = np.array([item[1] for item in training])

        return train_x, train_y

    def fit_model(self, epochs=200):
        X, y = self._prepare_training_data()
        self.model = Sequential([
            Dense(128, input_shape=(len(self.words),), activation='relu'),
            Dropout(0.5),
            Dense(64, activation='relu'),
            Dropout(0.5),
            Dense(len(self.classes), activation='softmax')
        ])

        self.model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.01), metrics=['accuracy'])
        self.history = self.model.fit(X, y, epochs=epochs, batch_size=5, verbose=1)

    def save_model(self):
        self.model.save(f"{self.model_name}.keras")
        pickle.dump(self.words, open(f'{self.model_name}_words.pkl', 'wb'))
        pickle.dump(self.classes, open(f'{self.model_name}_classes.pkl', 'wb'))

    def load_model(self):
        self.model = load_model(f'{self.model_name}.keras')
        self.words = pickle.load(open(f'{self.model_name}_words.pkl', 'rb'))
        self.classes = pickle.load(open(f'{self.model_name}_classes.pkl', 'rb'))

    def _predict_class(self, sentence):
        bow = [0] * len(self.words)
        sentence_words = nltk.word_tokenize(sentence)
        sentence_words = [self.lemmatizer.lemmatize(word) for word in sentence_words]
        for w in sentence_words:
            for i, word in enumerate(self.words):
                if word == w:
                    bow[i] = 1

        res = self.model.predict(np.array([bow]))[0]
        ERROR_THRESHOLD = 0.25
        results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
        results.sort(key=lambda x: x[1], reverse=True)
        return_list = [{'intent': self.classes[r[0]], 'probability': str(r[1])} for r in results]
        return return_list

    def process_input(self, message):
        ints = self._predict_class(message)
        if ints:
            tag = ints[0]['intent']
            if tag in self.method_mappings:
                function = self.method_mappings[tag]
                if callable(function):
                    return function()
            
            # If no function output, use the intent response
            for intent in self.intents['intents']:
                if intent['tag'] == tag:
                    return random.choice(intent['responses'])

        return "I'm not sure how to respond to that."

# Load portfolio
try:
    with open('portfolio.pkl', 'rb') as f:
        portfolio = pickle.load(f)
except FileNotFoundError:
    portfolio = {}

def save_portfolio():
    with open('portfolio.pkl', 'wb') as f:
        pickle.dump(portfolio, f)

def add_portfolio():
    ticker = input("Which stock do you want to add: ")
    amount = int(input("How many shares do you want to add: "))

    if ticker in portfolio:
        portfolio[ticker] += amount
    else:
        portfolio[ticker] = amount

    save_portfolio()
    return f"Added {amount} shares of {ticker} to your portfolio."

def remove_portfolio():
    ticker = input("Which stock do you want to sell: ")
    amount = int(input("How many shares do you want to sell: "))

    if ticker in portfolio:
        if amount <= portfolio[ticker]:
            portfolio[ticker] -= amount
            save_portfolio()
            return f"Sold {amount} shares of {ticker} from your portfolio."
        else:
            return "You don't have enough shares!"
    else:
        return f"You don't own any shares of {ticker}"

def show_portfolio():
    output = "Your portfolio:\n"
    for ticker, shares in portfolio.items():
        output += f"You own {shares} shares of {ticker}\n"
    return output.strip()

def portfolio_worth():
    total_worth = 0
    output = ""
    for ticker, shares in portfolio.items():
        try:
            stock = yf.Ticker(ticker)
            data = stock.history(period="1d")
            if data.empty:
                output += f"Unable to fetch data for {ticker}\n"
                continue
            price = data['Close'].iloc[-1]
            worth = price * shares
            total_worth += worth
            output += f"{ticker}: {shares} shares worth ${worth:.2f}\n"
        except Exception as e:
            output += f"Error fetching data for {ticker}: {str(e)}\n"
    output += f"Your total portfolio worth: ${total_worth:.2f}"
    return output

def portfolio_gains():
    starting_date = input("Enter date for a comparison (YYYY-MM-DD): ")
    sum_now = 0
    sum_then = 0

    try:
        for ticker, shares in portfolio.items():
            stock = yf.Ticker(ticker)
            data = stock.history(start=starting_date, end=dt.datetime.now())
            if data.empty:
                return f"Unable to fetch data for {ticker}"
            price_now = data['Close'].iloc[-1]
            price_then = data['Close'].iloc[0]
            sum_now += price_now * shares
            sum_then += price_then * shares

        gains = sum_now - sum_then
        relative_gains = (gains / sum_then) * 100

        return f"Relative Gains: {relative_gains:.2f}%\nAbsolute Gains: ${gains:.2f}"
    except Exception as e:
        return f"An error occurred: {str(e)}"

def plot_chart():
    ticker = input("Choose a ticker symbol: ")
    starting_string = input("Choose a starting date (YYYY-MM-DD): ")

    plt.style.use('dark_background')

    start = dt.datetime.strptime(starting_string, "%Y-%m-%d")
    end = dt.datetime.now()

    try:
        data = yf.download(ticker, start=start, end=end)
        if data.empty:
            return f"No data available for {ticker} in the specified date range."
        
        colors = mpf.make_marketcolors(up='#00ff00', down='#ff0000', wick='inherit', edge='inherit', volume='in')
        mpf_style = mpf.make_mpf_style(base_mpf_style='nightclouds', marketcolors=colors)
        mpf.plot(data, type='candle', style=mpf_style, volume=True)
        return f"Chart for {ticker} from {starting_string} to now has been displayed."
    except Exception as e:
        return f"An error occurred while plotting the chart for {ticker}. Error: {str(e)}"

def get_stock_price():
    ticker = input("Enter the stock symbol: ")
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period="1d")
        if data.empty:
            return f"Unable to fetch the price for {ticker}. Please check the symbol and try again."
        price = data['Close'].iloc[-1]
        return f"The current price of {ticker} is ${price:.2f}"
    except Exception as e:
        return f"An error occurred while fetching the price for {ticker}. Error: {str(e)}"

def bye():
    return "Goodbye", True

mappings = {
    'plot_chart': plot_chart,
    'add_portfolio': add_portfolio,
    'remove_portfolio': remove_portfolio,
    'show_portfolio': show_portfolio,
    'portfolio_worth': portfolio_worth,
    'portfolio_gains': portfolio_gains,
    'stock_price': get_stock_price,
    'bye': bye
}

assistant = BasicAssistant('intents.json', mappings, model_name="fin_assist_model")

# Train or load the model
if not os.path.exists("fin_assist_model.keras"):
    print("Training new model...")
    assistant.fit_model(epochs=50)  # You can adjust the number of epochs
    assistant.save_model()
else:
    print("Loading existing model...")
    assistant.load_model()

print("Financial Assistant is ready. Type 'bye' to exit.")

while True:
    message = input("You: ")
    if message.lower() == 'bye':
        print("Assistant:", bye())
        break
    response = assistant.process_input(message)
    print("Assistant:", response)
    print()  # Add a blank line for better readability