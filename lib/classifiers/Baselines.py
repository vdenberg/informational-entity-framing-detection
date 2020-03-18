from sklearn.feature_extraction.text import TfidfVectorizer
import numpy
import pandas


def get_tfidf_mean(tfidf, sent):
    scores = tfidf.transform([sent])
    for row in scores:
        scores = [cell for cell in row.data if cell > 0]
        scores = scores if scores else [1]
    average = sum(scores) / len(scores)
    return average


class TFIDFBaseline:
    def __init__(self, basil):
        self.basil = basil
        self.tfidf = TfidfVectorizer()
        self.tfidf.fit(basil.tokens)

    def rank_sentences(self):
        low_tfidf = pandas.DataFrame(numpy.zeros(self.basil.bias.shape), index=self.basil.index, columns=['low_tfidf'])
        for n, story in self.basil.groupby('story'):
            for n2, article in story.groupby('source'):
                av_scores = article.tokens.apply(lambda x: get_tfidf_mean(self.tfidf, x))
                picks = av_scores.nsmallest(4)
                low_tfidf.loc[picks.index, 'low_tfidf'] = [1] * 4
        return low_tfidf

    def get_y_pred(self, train_X, train_y, test_X, test_y):
        y_pred = test_X.low_tfidf
        return train_X, train_y, test_X, test_y, y_pred


class Maj:
    def __init__(self):
        self.maj = None

    def train(self, X, y):
        self.maj = y.mode().values[0]

    def predict(self, X):
        return [self.maj] * len(X)

    def get_y_pred(self, train_X, train_y, test_X, test_y):
        self.train(train_X, train_y)
        y_pred = self.predict(test_X)
        return train_X, train_y, test_X, test_y, y_pred

