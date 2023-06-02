from sklearn.metrics import precision_recall_fscore_support, matthews_corrcoef, f1_score
from transformers import EvalPrediction, set_seed

def calclulate_f1(statics_dict):
    prec, recall, f1 = 0, 0, 0
    if statics_dict["c"] != 0:
        prec = float(statics_dict["c"] / statics_dict["p"])
        recall = float(statics_dict["c"] / statics_dict["g"])
        f1 = float(prec * recall) / float(prec + recall) * 2
    return {"prec": prec, "recall": recall, "f1": f1}

def unirel_metric(p: EvalPrediction):
    tail_labels = p.label_ids
    tail_preds = p.predictions
    tail_acc, tail_recall, tail_f1, _ = precision_recall_fscore_support(
        y_pred=tail_preds.reshape(-1),
        y_true=tail_labels.reshape(-1),
        labels=[1],
        average='micro')

    return {
        "acc": tail_acc,
        "recall": tail_recall,
        "f1": tail_f1,
    }

def unirel_span_metric(p: EvalPrediction):
    head_labels, tail_labels, span_labels = p.label_ids
    head_preds, tail_preds, span_preds = p.predictions
    head_acc, head_recall, head_f1, _ = precision_recall_fscore_support(
        y_pred=head_preds.reshape(-1),
        y_true=head_labels.reshape(-1),
        labels=[1],
        average='micro')
    tail_acc, tail_recall, tail_f1, _ = precision_recall_fscore_support(
        y_pred=tail_preds.reshape(-1),
        y_true=tail_labels.reshape(-1),
        labels=[1],
        average='micro')
    span_acc, span_recall, span_f1, _ = precision_recall_fscore_support(
        y_pred=span_preds.reshape(-1),
        y_true=span_labels.reshape(-1),
        labels=[1],
        average='micro')
    return {
        "head_acc": head_acc,
        "head_recall": head_recall,
        "head_f1": head_f1,
        "tail_acc": tail_acc,
        "tail_recall": tail_recall, 
        "tail_f1": tail_f1,
        "span_acc": span_acc,
        "span_recall": span_recall,
        "span_f1": span_f1,
    }