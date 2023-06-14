import numpy as np
import pandas as pd

def get_sufficiency_gap_quantiles(y,p,races,buckets=500):
    #todo redo assessment with deciles
    sufficiency_gaps = {f : 0 for f in list(races.unique())}
    bins = pd.qcut(p,q=buckets,duplicates='drop')
    for race in races.unique():
        sufficiency_gap = 0
        for q in bins.unique():
            if len(p[(races==race) & (bins==q)]):
                weight = len(bins[bins==q])/len(bins)
                race_weight =  len(bins[(bins==q) & (races==race)])/len(bins)
                sufficiency_gap += weight * (np.mean(y[bins==q])) - race_weight*(np.mean(y[(bins==q) & (races==race)]))
        sufficiency_gaps[race] = sufficiency_gap
    return sufficiency_gaps

def get_separation_gap(y,p,races,label):
    separation_gaps = {f : 0 for f in list(races.unique())}
    p_label = p[y==label]
    for race in races.unique():
        separation_gaps[race] = np.mean(p_label[races==race]) - np.mean(p_label)
    return separation_gaps
