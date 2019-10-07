import pandas as pd
import pylab as pl
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc, accuracy_score


def get_threshold_where_fpr_approx_20perc(fpr, thresholds):
    best_threshold = 0
    for i, fpr in enumerate(fpr):
        if 0.17<fpr:
            return thresholds[i]

def get_thresholds_as_first_bad_change(tpr2fpr, thresholds):
    d_tpr2fpr_dt = tpr2fpr[1:]-tpr2fpr[:-1]
    
    best_idx = 0
    for i, change in enumerate(d_tpr2fpr_dt):
        if change<0:
            best_idx = i
            break
            
    return thresholds[best_idx]
        
    
def find_thresholds(y_scores, y_labels):
    # for seeker avoider - NOT undetermined
    
    fpr, tpr, thresholds = roc_curve(y_labels, y_scores)
    tpr2fpr = tpr-fpr
    
    best_threshold = get_threshold_where_fpr_approx_20perc(fpr, thresholds)
    best_prediction = np.array(y_scores)>best_threshold
            
    #print('tpr: ', tpr)
    #print('fpr: ', fpr)
    #print('thresh: ', thresholds)
    #print('tpr2fpr: ', tpr2fpr)
    #print('best thresh: ', best_threshold)
    #print('y_labels: ', y_labels[::3])
    #print('y_scores: ', y_scores[::3])
    
    #raise Exception('boom')
    return best_threshold, best_prediction
    
    

def find_optimal_cutoff(target, predicted):
    """ Find the optimal probability cutoff point for a classification model related to event rate parameters
    ----------
    target    : matrix with dependent or target data, where rows are observations
    predicted : matrix with predicted data, where rows are observations
    returns   : list type, with optimal cutoff value
    """
    fpr, tpr, threshold = roc_curve(target, predicted)
    i = np.arange(len(tpr))
    roc = pd.DataFrame({'tf' : pd.Series(tpr-(1-fpr), index=i), 'threshold' : pd.Series(threshold, index=i)})
    #roc = pd.DataFrame({'tf' : pd.Series(tpr-fpr, index=i), 'threshold' : pd.Series(threshold, index=i)})
    roc_t = roc.ix[(roc.tf-0).abs().argsort()[:1]]

    return list(roc_t['threshold'])



