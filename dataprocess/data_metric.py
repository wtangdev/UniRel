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
    token_len = 102
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
