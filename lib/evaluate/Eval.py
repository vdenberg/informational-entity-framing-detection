from sklearn.metrics import matthews_corrcoef, confusion_matrix, f1_score, accuracy_score
from sklearn.metrics import precision_recall_fscore_support
import pandas as pd


def convert_bio_to_binary(labels):
    labels = [2 if lab == 3 else lab for lab in labels]
    labels = [1 if lab == 0 else lab for lab in labels]
    labels = [0 if lab == 1 else lab for lab in labels]
    labels = [1 if lab == 2 else lab for lab in labels]
    return labels


def get_metrics(labels, preds):
    assert len(preds) == len(labels)

    if len(set(labels)) > 2:
        preds = convert_bio_to_binary(preds)
        labels = convert_bio_to_binary(labels)

    else:
        preds = [lab - 1 for lab in preds]
        labels = [lab - 1 for lab in labels]

    assert set(labels) == {0, 1}

    #mcc = matthews_corrcoef(labels, preds)
    acc = accuracy_score(labels, preds)
    prec_rec_fscore = precision_recall_fscore_support(labels, preds, labels=[0, 1])
    prec, rec, _ = [el[1] for el in prec_rec_fscore[:-1]]

    f1 = f1_score(labels, preds, average='binary')
    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
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


def my_eval(labels, preds, av_loss=None, set_type="", name=""):
    """
    Compares labels to predictions, Loss can be added to also
    display the loss associated to the model that made those predictions
    bzw. the loss associated with those predictions.
    :param labels: self-explan.
    :param preds: self-explan.
    :param av_loss: loss, e.g. train loss, or val loss
    :param set_type: are the labels train dev or test labels?
    :param name: name of model that's being evaled, e.g. 'bert_epochx'
    :return:
    """
    # METRICS_DICT
    metrics_dict = get_metrics(labels, preds)

    if av_loss:
        metrics_dict['loss'] = av_loss

    if set_type:
        metrics_dict['set_type'] = set_type

    # METRICS_DF
    # metrics_df = pd.DataFrame(metrics_dict)

    # METRICS_STRING
    # select values
    metrics = [metrics_dict['acc'], metrics_dict['prec'], metrics_dict['rec'], metrics_dict['f1']]
    metrics = [str(round(met * 100, 2)) for met in metrics]
    if av_loss:
        av_loss = round(metrics_dict['loss'], 4)
        metrics += [av_loss]

    # make conf_mat
    conf_mat = f"(tn {metrics_dict['tn']} tp {metrics_dict['tp']} fn {metrics_dict['fn']} fp {metrics_dict['fp']})"
    if av_loss:
        metrics_string = f"On {set_type}: loss {metrics[4]} {conf_mat} acc {metrics[0]} prec {metrics[1]} rec {metrics[2]} > {set_type} f1: {metrics[3]} <"
    else:
        metrics_string = f"On {set_type} {conf_mat}: acc {metrics[0]} prec {metrics[1]} rec {metrics[2]} > {set_type} f1: {metrics[3]} <"

    return metrics_dict, metrics_string


'''
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
    metrics_string = [' Acc: {:.4f}'.format(acc)] + metrics_string + ['F1: {:.4f}'.format(f1)]
    metrics_string = " ".join(metrics_string)

    return metrics, metrics_string
'''
