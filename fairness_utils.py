import numpy as np

def get_sufficiency_gap(y,p,races,round_to=1):
    #todo redo assessment with deciles
    if round_to != None:
        p = p.apply(round,args=[round_to])
    sufficiency_gaps = {f : 0 for f in list(races.unique())}
    for race in races.unique():
        sufficiency_gap = 0
        for score in p[races==race].unique(): # prevent nan
            weight = len(p[p==score])/len(p)
            sufficiency_gap += weight * (np.mean(y[p==score]) 
                                         - np.mean(y[(p==score) & (races==race)]))
        sufficiency_gaps[race] = sufficiency_gap
    return sufficiency_gaps

def get_separation_gap(y,p,races,label):
    separation_gaps = {f : 0 for f in list(races.unique())}
    p_label = p[y==label]
    for race in races.unique():
        separation_gaps[race] = np.mean(p_label[races==race]) - np.mean(p_label)
    return separation_gaps
