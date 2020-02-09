# -*- coding: utf-8 -*-
# @Time    : 2020/2/3 9:59
# @Author  : Haoyi Fan
# @Email   : isfanhy@gmail.com
# @File    : utils.py
import json
import numpy as np


def precision_AT_K(actual, predicted, k, num_anomaly):
    act_set = np.array(actual[:k])
    pred_set = np.array(predicted[:k])
    ll = act_set & pred_set
    tt = np.where(ll == 1)[0]
    prec = len(tt) / float(k)
    rec = len(tt) / float(num_anomaly)
    return round(prec, 4), round(rec, 4)


def save_results(results, export_json):
    """Save results dict to a JSON-file."""
    with open(export_json, 'w') as fp:
        json.dump(results, fp)

def read_results(export_json):
    """Save results dict to a JSON-file."""
    with open(export_json, 'r') as fp:
        results = json.load(fp)
    return results

