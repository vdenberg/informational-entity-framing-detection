from helpers import evaluation


class CrossValidation:
    def __init__(self):
        pass

    def average(self, scores):
        scores = zip(*scores)
        return [sum(sc) / len(sc) for sc in scores]

    def cross_validate(self, data, get_y_pred):
        scores = []
        for fold in data:
            (train_X, train_y), (dev_X, dev_y), (test_X, test_y) = fold
            print('\nTrain:', len(train_X), 'Dev:', len(dev_X), 'Test:', len(test_X))

            y_pred = get_y_pred(fold)
            metrics, metrics_string = evaluation(test_y, y_pred)

            print(metrics_string)
            scores.append(metrics)

        scores = self.average(scores)
        print(scores)
        return scores