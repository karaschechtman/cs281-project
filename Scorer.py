from abc import *
from sympy import *

import matplotlib.pyplot as plt
import math 
import numpy as np

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
        plt.plot(x, s0, 'r')
        plt.plot(x, s1, 'b')
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

    def score_components(self,y,p,round_to=None):
        if round_to != None:
            p = p.apply(round,args=[round_to])
        # Return scores segmented into components representing calibration and refinement
        calibration_component = refinement_component = 0
    
        for x in p.unique():
            nu = len(p[p==x])/len(p)
            rho = len(p[(p==x) & (y==1)])/len(p[p==x])
            calibration_component += nu * (
                rho * (self.s1(x) - self.s1(rho)) - \
                (1-rho) * (self.s0(x)-self.s0(rho))
            )
            refinement_component +=  nu * (
                    rho * (self.s1(rho) + \
                    (1-rho)) * (self.s0(rho))
            )
        return calibration_component, refinement_component
    
class BetaScorer(Scorer):
    def __init__(self,alpha,beta):
        self.alpha = alpha
        self.beta = beta
        t = Symbol('t')
        self._s0 = lambdify(t,integrate((t**self.alpha)*((1-t)**(self.beta-1)),t))
        s1_helper = integrate(t**(self.alpha-1)*(1-t)**self.beta,t)
        self._s1 = lambdify(t,s1_helper.subs({t:1}) - s1_helper)
        super().__init__()
    
    def s0(self,p):
        return self._s0(p)

    def s1(self,p):
        return self._s1(p)

class WeightedLogScorer(Scorer):
    def __init__(self,a=1,b=1):
        self.a = a
        self.b = b
        super().__init__()

    def s0(self,p):
        try:
            return -1**self.a*math.log(p)**self.a
        except:
            return float('inf')

    def s1(self,p):
        try:
            return -1**self.b*math.log(1-p)**self.b
        except:
            return float('inf') 
        
class WeightedBrierScorer(Scorer):
    def __init__(self,a=2,b=2):
        self.a = a
        self.b = b
        super().__init__()

    def s0(self,p):
        return p**self.a

    def s1(self,p):
        return (1-p)**self.b
