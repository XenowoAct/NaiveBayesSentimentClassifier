# NaiveBayesSentimentClassifier

This code performs sentiment analysis using a Naive Bayes classifier. It predicts the sentiment (positive or negative) of a given document based on a set of labelled tweets.

## Prerequisites

- Python 3.x
- numpy
- nltk

## Installation

1. Clone the repository:

```
git clone https://github.com/XenowoAct/NaiveBayesSentimentClassifier
```

2. Install the required packages:

```
pip install nltk numpy
```

### Usage

1. Format your dataset into a labelled dictionary (As shown in line 16 of the code:)

```
labelledTweets = {'aino is pog': 1, 'frick aino': 0, 'frick you bino': 0, 'I love honey': 1}
```

2. Use the `PredictDoc` function to predict the sentiment of any document. The function takes the document you want to predict and the labelled dictionary as argument and returns a float number.
A possitive prediction indicates that the module predicted positive sentiment, and vice versa. The module automatically deals with words which aren't in it's training set.

