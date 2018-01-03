from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score, recall_score
from sklearn.utils.multiclass import unique_labels
import numpy as np
from operator import itemgetter


def get_eval_metric(y_true, y_pred, eval_metric="f1_macro", eval_labels=None):
    """
    Returns selected evaluation metric based on true and prediction labels.

    :param y_true: array
        Ground truth (correct) target values.
    :param y_pred: array
        Estimated targets as returned by a classifier.
    :param eval_metric: string
        Identifier for eval metric {f1_macro, f1_micro}
    :param eval_labels: array, optional
        Sett of labels to include. For multilabel targets, labels are column indices. By default (None), all labels in
        y_true and y_pred are used in sorted order.
    :return:
    """
    if eval_metric == "f1_macro":
        eval_metric = f1_score(y_true, y_pred, labels=eval_labels, average="macro")
    elif eval_metric == "f1_micro":
        eval_metric = f1_score(y_true, y_pred, labels=eval_labels, average="micro")
    elif eval_metric == "recall_macro":
        eval_metric = recall_score(y_true, y_pred, labels=eval_labels, average="macro")
    else:
        raise ValueError("Unknown eval_metric value in config: {0:s}".format(config["best_metric"]))

    return eval_metric


def get_confustion_matrix(y_gold, y_pred):
    label_names = unique_labels(y_gold, y_pred)
    conf_mat = confusion_matrix(y_gold, y_pred, label_names)
    return conf_mat
    
def print_results(y_gold, y_pred):
    correct_predictions = sum(np.array(y_pred) == np.array(y_gold))
    print("Total number of test examples: {}".format(len(y_gold)))
    label_names = unique_labels(y_gold, y_pred)
    conf_mat = confusion_matrix(y_gold, y_pred, label_names)
    #print("Confustion matrix:\n" + str(conf_mat))
    recall = []
    precision = []
    f1 = []
    print("Label\tRecall\tPrecision\tF1")
    for label_id in range(len(label_names)):
        recall.append(get_recall(conf_mat, label_id))
        precision.append(get_precision(conf_mat, label_id))
        f1.append(get_f1(conf_mat, label_id))
        print(str(label_names[label_id]) + "\t" + str(get_recall(conf_mat, label_id))+ "\t" + str(get_precision(conf_mat, label_id))+ "\t" + str(get_f1(conf_mat, label_id)))
    print("Accuracy: {:g}".format(correct_predictions/float(np.sum(conf_mat))))
    print("Recall (macro-avrg)", np.mean(recall))
    print("Precision (macro-avrg)", np.mean(precision))
    print("F1 (macro-avrg)", np.mean(f1))
    label_names,recall,precision,f1 = zip(*sorted(zip(label_names,recall,precision,f1),
  key=itemgetter(0)))
    #print(" ".join(str(x) for x in label_names))
    #print(" ".join(str(x) for x in recall))
    #print(" ".join(str(x) for x in precision))
    #print(" ".join(str(x) for x in f1))
    return conf_mat
    
def mean_f1(y_gold, y_pred):
    label_names = unique_labels(y_gold, y_pred)
    conf_mat = confusion_matrix(y_gold, y_pred, label_names)
    return conf_mat_f1(conf_mat)

    
def avrg_neg_pos_f1(conf, neg_id, pos_id):
    neg_f1 = f1(conf, neg_id)
    pos_f1 = f1(conf, pos_id)
    avrg_f1 = (neg_f1 + pos_f1) / 2
    return avrg_f1


def get_recall(conf_mat, label_id):
    true_pos = conf_mat[label_id][label_id]
    false_neg = sum(conf_mat[label_id][:])-conf_mat[label_id][label_id]
    return true_pos/(true_pos+false_neg)


def get_precision(conf_mat, label_id):
    true_pos = conf_mat[label_id][label_id]
    false_pos = sum([row[label_id] for row in conf_mat])-conf_mat[label_id][label_id]
    return true_pos/(true_pos+false_pos)


def get_f1(conf_mat, label_id):
    recall = get_recall(conf_mat, label_id)
    precision = get_precision(conf_mat, label_id)
    return 2 * recall * precision / (recall+precision)
    

def conf_mat_f1(conf_mat):
    """
    Return macro averaged F1 of input confusion matrix.

    :param conf_mat: Confusion matrix array as return by sklearn method
    :return:
    """
    f1 = []
    for label_id in range(len(conf_mat)):
        f1.append(get_f1(conf_mat, label_id))
    return np.mean(f1)


def conf_mat_recall(conf_mat):
    """
    Return macro averaged Recall of input confusion matrix.

    :param conf_mat: Confusion matrix array as return by sklearn method
    :return:
    """
    recall = []
    for label_id in range(len(conf_mat)):
        recall.append(get_recall(conf_mat, label_id))
    return np.mean(recall)


def conf_mat_precision(conf_mat):
    """
    Return macro averaged Recall of input confusion matrix.

    :param conf_mat: Confusion matrix array as return by sklearn method
    :return:
    """
    precision = []
    for label_id in range(len(conf_mat)):
        precision.append(get_precision(conf_mat, label_id))
    return np.mean(precision)


if __name__ == "__main__":
    y_gold = [0, 0, 0, 1, 1, 1, 2, 2, 2]
    y_pred = [0, 0, 0, 0, 1, 1, 2, 2, 1]
    print_results(y_gold, y_pred)
    mean_f1(y_gold, y_pred)