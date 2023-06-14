from abc import *
from sympy import *

import matplotlib.pyplot as plt
import math 
import numpy as np
import pandas as pd

class Scorer(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def s0(self,p):
        pass

    @abstractmethod
    def s1(self,p):
        pass

    def plot(self):
        x = np.linspace(0,1,200)
        s0 = np.vectorize(self.s0)(x)
        s1 = np.vectorize(self.s1)(x)
        plt.xlabel('Probability')
        plt.ylabel('Score')
        plt.title(self.title)
        plt.plot(x, s0, 'mediumorchid',label='s0')
        plt.plot(x, s1, 'mediumseagreen',label='s1')
        plt.legend(loc='upper right')
        plt.show()

    def score(self,y,p):
        return self.s0(p) if y == 0 else self.s1(p)
        
    def score_many(self,y,p,sensitive_features=[]):
        scores_dict = {f : 0 for f in list(sensitive_features.unique()) + ['all']}
        n = len(y)
        for i in range(n):
            score = self.score(y[i],p[i])
            scores_dict['all']+= score
            if len(sensitive_features):
                scores_dict[sensitive_features[i]]+=score
        
        # Normalize scores
        scores_dict['all'] = scores_dict['all']/n
        for f in sensitive_features.unique():
            scores_dict[f] = scores_dict[f]/len(sensitive_features[sensitive_features==f])
        return  scores_dict

    def score_components(self,y,p,buckets=50):
        calibration_component = refinement_component = 0
        bins = pd.qcut(p,q=buckets,duplicates='drop')
        bins = bins.rename('bins')
        p = p.rename('score')
        bin_means = pd.concat([p,bins],axis=1).groupby('bins').agg(np.mean).reset_index()
        bins = bins.astype('string')
        bin_means['bins'] = bin_means['bins'].astype('string')
        for _, row in bin_means.iterrows():
            x = row['score']
            cat = row['bins']
            if cat in bins.unique():
                nu = len(p[bins==cat])/len(p)
                rho = len(p[(bins==cat) & (y==1)])/len(p[bins==cat])
                calibration_component += nu * (
                    rho * (self.s1(x) - self.s1(rho)) - \
                    (1-rho) * (self.s0(x)-self.s0(rho))
                )
                refinement_component +=  nu * (
                        rho * (self.s1(rho) + \
                        (1-rho)) * (self.s0(rho))
                )
        return calibration_component, refinement_component
    
# TODO(kara): reimplement using the scipy beta function, I think it will be better/faster
class BetaScorer(Scorer):
    def __init__(self,alpha,beta):
        self.alpha = alpha
        self.beta = beta
        self.title = "Beta Scorer (α = %d, β=%d)"
        t = Symbol('t')
        self._s0 = lambdify(t,integrate((t**self.alpha)*((1-t)**(self.beta-1)),t))
        s1_helper = integrate(t**(self.alpha-1)*(1-t)**self.beta,t)
        self._s1 = lambdify(t,s1_helper.subs({t:1}) - s1_helper)
        super().__init__()
    
    def s0(self,p):
        return self._s0(p)

    def s1(self,p):
        return self._s1(p)

class LogScorer(Scorer):
    def __init__(self):
        super().__init__()
        self.title = "Log Score"

    def s0(self,p):
        try:
            return -math.log(1-p)
        except ValueError:
            return float('inf')

    def s1(self,p):
        try:
            return -math.log(p)
        except ValueError:
            return float('inf')

class BrierScorer(Scorer):
    def __init__(self):
        super().__init__()
        self.title = "Brier Score"

    def s0(self,p):
        return p**2

    def s1(self,p):
        return (1-p)**2
    
class CustomScorer(Scorer):
    def __init__(self,s0,s1,title):
        super().__init__()
        self._s0 = s0
        self._s1 = s1
        self.title = title
    
    def s0(self,p):
        return self._s0(p)

    def s1(self,p):
        return self._s1(p)
