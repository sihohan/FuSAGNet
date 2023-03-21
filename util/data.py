import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import iqr, rankdata
from sklearn.metrics import f1_score


def generate_gaussian_noise(x, mean=0.0, std=1.0):
    eps = torch.randn(x.size()) * std + mean
    return eps


def eval_scores(scores, true_scores, th_steps, return_threshold=False):
    padding_list = [0] * (len(true_scores) - len(scores))
    if len(padding_list) > 0:
        scores = padding_list + scores

    scores_sorted = rankdata(scores, method="ordinal")
    th_vals = np.array(range(th_steps)) * 1.0 / th_steps
    fmeas = [None] * th_steps
    thresholds = [None] * th_steps
    for i in range(th_steps):
        cur_pred = scores_sorted > th_vals[i] * len(scores)
        fmeas[i] = f1_score(true_scores, cur_pred)
        score_index = scores_sorted.tolist().index(int(th_vals[i] * len(scores) + 1))
        thresholds[i] = scores[score_index]

    if return_threshold:
        return fmeas, thresholds

    return fmeas


def get_err_median_and_iqr(predicted, groundtruth):
    np_arr = np.subtract(np.array(predicted), np.array(groundtruth))
    err_median = np.median(np_arr)
    err_iqr = iqr(np_arr)
    return err_median, err_iqr


def kl_divergence(p, q):
    p = F.softmax(p, dim=1)
    q = F.softmax(q, dim=1)
    s1 = torch.sum(p * torch.log(p / q))
    s2 = torch.sum((1 - p) * torch.log((1 - p) / (1 - q)))
    return s1 + s2
