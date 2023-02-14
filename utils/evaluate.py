"""
Script providing evaluation functions for LGBM and Type2 GNNs.
"""
import logging
import torch
import numpy as np
from sklearn import metrics
from torch_geometric.loader import NeighborLoader, LinkNeighborLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, precision_recall_curve, mean_squared_error

from utils.gcn_utils import remove_reverse, get_batch_size

def metric_per_class(metric, labels, preds, axis=1, **kwargs):
    return [
        metric(l.unsqueeze(1), p.unsqueeze(1), **kwargs) for l,p in zip(torch.unbind(labels, dim=axis), torch.unbind(preds, dim=axis))
    ]

def compute_binary_metrics(preds, labels):
    """
    Computes metrics based on raw/ normalized model predictions
    :param preds: Raw (or normalized) predictions (can vary threshold here if raw scores are provided)
    :param labels: Binary target labels
    :return: Accuracy, illicit precision/ recall/ F1, and ROC AUC scores
    """
    logging.debug(f"preds.shape = {preds.shape}, labels.shape = {labels.shape}")
    if len(preds.shape) > 1 and preds.shape[1] > 1:
        probs = preds[:,1]
        preds = preds.argmax(dim=-1)
    else:
        probs = torch.sigmoid(preds)
        preds = probs.round()
    ap = metrics.average_precision_score(labels, probs, pos_label=1)  # area under precision-recall curve
    fpr, tpr, thresholds = metrics.roc_curve(labels, probs, pos_label=1)
    auc = metrics.auc(fpr, tpr)  # area under roc curve (fpr vs tpr)
    
    precs, recs, thrs = precision_recall_curve(labels, probs, pos_label=1)

    accuracy = accuracy_score(labels, preds)
    logging.debug("evaluate.py/compute_binary_metrics")
    logging.debug(f"labels: dtype = {labels.dtype}, shape = {labels.shape}, max() = {labels.max()}, min() = {labels.min()}")
    logging.debug(f"preds:  dtype = {labels.dtype}, shape = {labels.shape}, max() = {labels.max()}, min() = {labels.min()}")
    precision = precision_score(labels.bool(), preds, zero_division=0)
    recall = recall_score(labels, preds, zero_division=0)
    F1 = f1_score(labels, preds, zero_division=0)

    return accuracy, precision, recall, F1, auc, ap, (fpr, tpr), (precs, recs)

def compute_binary_f1(preds, labels):
    """
    Computes metrics based on raw/ normalized model predictions
    :param preds: Raw (or normalized) predictions (can vary threshold here if raw scores are provided)
    :param labels: Binary target labels
    :return: Accuracy, illicit precision/ recall/ F1, and ROC AUC scores
    """
    if len(preds.shape) > 1 and preds.shape[1] > 1:
        probs = preds[:,1]
        preds = preds.argmax(dim=-1)
    else:
        probs = torch.sigmoid(preds)
        preds = probs.round()
    
    F1 = f1_score(labels, preds, zero_division=0)

    return F1

def compute_continuous_metrics(preds, labels):
    """
    Computes metrics based on raw/ normalized model predictions
    :param preds: Raw (or normalized) predictions (can vary threshold here if raw scores are provided)
    :param labels: Binary target labels
    :return: Accuracy, illicit precision/ recall/ F1, and ROC AUC scores
    """
    probs = preds
    preds = preds.round()

    # Add some defaults - ignore these in the logs!
    y_true = np.array([0, 0, 1, 1])
    y_scores = np.array([0.1, 0.4, 0.35, 0.8])
    ap = 0
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_scores, pos_label=1)
    auc = metrics.auc(fpr, tpr)  # area under roc curve (fpr vs tpr)
    precs, recs, thrs = precision_recall_curve(y_true, y_scores, pos_label=1)

    # accuracy = accuracy_score(labels.round(), preds)
    accuracies = metric_per_class(metrics.accuracy_score, labels.round(), preds)
    accuracy = np.mean(accuracies)
    logging.debug("evaluate.py/compute_continuous_metrics")
    logging.debug(f"labels: dtype = {labels.dtype}, shape = {labels.shape}, max() = {labels.max()}, min() = {labels.min()}")
    logging.debug(f"preds:  dtype = {labels.dtype}, shape = {labels.shape}, max() = {labels.max()}, min() = {labels.min()}")
    precision = 0
    recall = 0
    MSEs = metric_per_class(mean_squared_error, labels, probs)
    F1 = np.exp(-np.mean(MSEs))

    return accuracy, precision, recall, F1, auc, ap, (fpr, tpr), (precs, recs), accuracies

def compute_multiclass_metrics(preds, labels):
    """
    Computes metrics based on raw/ normalized model predictions
    :param preds: Raw (or normalized) predictions (can vary threshold here if raw scores are provided)
    :param labels: Binary target labels
    :return: Accuracy, illicit precision/ recall/ F1, and ROC AUC scores
    """
    probs = torch.sigmoid(preds)
    preds = probs.round()

    ap = metrics.average_precision_score(labels, probs, pos_label=1)  # area under precision-recall curve

    # Add some defaults - ignore these in the logs!
    y_true = np.array([0, 0, 1, 1])
    y_scores = np.array([0.1, 0.4, 0.35, 0.8])
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_scores, pos_label=1)
    auc = metrics.auc(fpr, tpr)  # area under roc curve (fpr vs tpr)
    precs, recs, thrs = precision_recall_curve(y_true, y_scores, pos_label=1)

    accuracies = metric_per_class(metrics.accuracy_score, labels, preds)
    accuracy = np.mean(accuracies)
    # accuracy = accuracy_score(labels, preds)
    precisions = metric_per_class(precision_score, labels.bool(), preds, zero_division=0)
    precision = np.mean(precisions)
    recalls = metric_per_class(recall_score, labels, preds, zero_division=0)
    recall = np.mean(recalls)
    F1s = metric_per_class(f1_score, labels, preds, zero_division=0)
    F1 = np.mean(F1s)

    return accuracy, precision, recall, F1, auc, ap, (fpr, tpr), (precs, recs), F1s

def compute_continuous_f1(preds, labels):
    """
    Computes metrics based on raw/ normalized model predictions
    :param preds: Raw (or normalized) predictions (can vary threshold here if raw scores are provided)
    :param labels: Binary target labels
    :return: Accuracy, illicit precision/ recall/ F1, and ROC AUC scores
    """
    probs = preds
    preds = preds.round()
    MSEs = metric_per_class(mean_squared_error, labels, probs)
    F1 = np.exp(-np.mean(MSEs))
    return F1

def compute_multiclass_f1(preds, labels):
    probs = torch.sigmoid(preds)
    preds = probs.round()
    F1s = metric_per_class(f1_score, labels, preds, zero_division=0)
    F1 = np.mean(F1s)
    return F1



@torch.no_grad()
def evaluate(eval_data, model, config, args, topk=False, y_type='binary', only_f1=False, return_preds=False):
    """
    Computes evaluation metrics for homogeneous/ multirelational GNN models in accordance with config parameters
    :param eval_data: Data to be evaluated (can be a list of Data/ HeteroData instances, a NeighborLoader, or simply a
    Data/ HeteroData instance)
    :param model: GNN model
    :param config: Configuration file
    :param args: Auxiliary command line arguments
    :param micro_avg: Bool to select whether to compute micro-averaging (relevant for lists of datasets or batching)
    :return: Accuracy, precision, recall, F1, and ROC AUC metrics
    """
    tx = 'tx'
    model.eval()
    model.embedding = 0

    preds, targets = [], []
    accs, pres, recs, f1s, aucs, aps = [], [], [], [], [], []

    if isinstance(eval_data, list):
        for data in eval_data:
            # if config.multi_relational:
            if config.model == "type2_hetero_sage":
                eval_pred, _ = model(data)
            # elif not config.multi_relational:
            else:
                eval_pred, _ = model(data)
            preds.append(eval_pred)
            y = data[tx].y if config.multi_relational else data.y
            targets.append(y)
    elif isinstance(eval_data, NeighborLoader) or isinstance(eval_data, LinkNeighborLoader):
        i = 0
        for batch in eval_data:
            if topk:
                if i > topk:
                    break
                i+=1
            if config.reverse_mp: batch = remove_reverse(batch)
            batch.to(args.device)
            _batch_size = get_batch_size(config, args, batch)
            # if config.multi_relational:
            if config.model == "type2_hetero_sage":
                eval_pred, _ = model(batch)
                eval_pred = eval_pred[:_batch_size].detach().cpu()
                target = batch['tx'].y[:_batch_size].cpu()
            # elif not config.multi_relational:
            else:
                eval_pred, _ = model(batch)
                eval_pred = eval_pred[:_batch_size].detach().cpu()
                target = batch.y[:_batch_size].cpu()
            logging.debug(f"eval_pred.shape = {eval_pred.shape}")
            preds.append(eval_pred)
            targets.append(target)
    else:
        # Here we compute for Data or HeteroData object (i.e. no batches or lists)
        # if config.multi_relational:
        if config.model == "type2_hetero_sage":
            eval_pred, _ = model(eval_data)
            eval_pred = eval_pred
        # elif not config.multi_relational:
        else:
            eval_pred, _ = model(eval_data)
            eval_pred = eval_pred
        preds.append(eval_pred)
        y = eval_data[tx].y if config.multi_relational else eval_data.y
        targets.append(y)

    model.embedding = config.generate_embedding

    preds = torch.cat(preds)
    targets = torch.cat(targets)

    if only_f1:
        if y_type == 'binary':
            f1 = compute_binary_f1(preds.cpu(), targets.cpu())
        elif y_type == 'multiclass':
            f1 = compute_multiclass_f1(preds.cpu(), targets.cpu())
        elif y_type == 'continuous':
            f1 = compute_continuous_f1(preds.cpu(), targets.cpu())
    else:
        if y_type == 'binary':
            acc, pre, rec, f1, auc, ap, roc_curve, pr_curve = compute_binary_metrics(preds.cpu(), targets.cpu())
            accuracies_per_class = [acc]
        elif y_type == 'multiclass':
            acc, pre, rec, f1, auc, ap, roc_curve, pr_curve, accuracies_per_class = compute_multiclass_metrics(preds.cpu(), targets.cpu())
        elif y_type == 'continuous':
            acc, pre, rec, f1, auc, ap, roc_curve, pr_curve, accuracies_per_class = compute_continuous_metrics(preds.cpu(), targets.cpu())

    logging.debug(f"preds.shape = {preds.shape}, targets.shape = {targets.shape}")
    logging.debug(f"{torch.cat([preds.reshape((len(targets), -1)), targets.reshape((len(targets), 1))], dim=1)[:10, :]}")
    if only_f1:
        return f1
    else:
        if return_preds:
            return (acc, pre, rec, f1, auc, ap, roc_curve, pr_curve, accuracies_per_class), preds
        else:
            return acc, pre, rec, f1, auc, ap, roc_curve, pr_curve, accuracies_per_class
