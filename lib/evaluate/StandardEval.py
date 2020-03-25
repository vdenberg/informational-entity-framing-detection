from sklearn.metrics import matthews_corrcoef, confusion_matrix, f1_score, accuracy_score
from sklearn.metrics import precision_recall_fscore_support
import pandas as pd


def get_metrics(labels, preds):
    assert len(preds) == len(labels)

    #mcc = matthews_corrcoef(labels, preds)
    acc = accuracy_score(labels, preds)
    prec_rec_fscore = precision_recall_fscore_support(labels, preds, labels=[0, 1])
    prec, rec, _ = [el[1] for el in prec_rec_fscore[:-1]]

    f1 = f1_score(labels, preds)

    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()

    #"mcc": mcc
    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "f1": f1,
        'acc': acc,
        'rec': rec,
        'prec': prec
    }


def my_eval(labels, preds, av_loss=None):
    metrics_dict = get_metrics(labels, preds)

    # make string in the style that I like
    acc, f1 = metrics_dict['acc'], metrics_dict['f1']
    select_metrics = [metrics_dict['prec'], metrics_dict['rec']]
    metrics_string = ['{:.4f}'.format(met) for met in select_metrics]
    metrics_string = ['Acc: {:.4f}'.format(acc)] + metrics_string + ['F1: {:.4f}'.format(f1)]
    metrics_string = " ".join(metrics_string)

    #metrics_df = pd.DataFrame(metrics_dict)

    if av_loss:
        metrics_dict['loss'] = av_loss
        #metrics_df['average_loss'] = av_loss
        metrics_string += ' Loss: {:.4f}'.format(av_loss)

    conf_mat = {'tn': metrics_dict['tn'], 'tp': metrics_dict['tp'], 'fn': metrics_dict['fn'], 'fp': metrics_dict['fp']}
    metrics_string += f" {conf_mat}"

    return metrics_dict, metrics_string


# old eval helper

def evaluation(y_true, y_pred):
    # acc and f1
    f1 = f1_score(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)

    # other scores
    prec_rec_fscore = precision_recall_fscore_support(y_true, y_pred, labels=[0,1])
    prec, rec, _ = [round(el[1], 2) for el in prec_rec_fscore[:-1]]
    metrics = [prec, rec]
    metrics_string = ['{:.4f}'.format(met) for met in metrics]

    metrics = [acc] + metrics + [f1]
    metrics_string = ['Acc: {:.4f}'.format(acc)] + metrics_string + ['F1: {:.4f}'.format(f1)]
    metrics_string = " ".join(metrics_string)

    return metrics, metrics_string