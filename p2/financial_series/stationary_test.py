# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
# %% Import libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# %% Get financial price series


# %% ADF and auxiliar functions

def ztcrit(nobs,p):
    '''ztcrit function
    Critical values for adf test (Zt statistics)
    
    Inputs:
        nobs <-- number of observations
        p = order of polinomial in null-hypothesis
        p = -1, no deterministic part
        p = 0, constant term
        p = 1, constant plus time trend
        p > 1, higher order polynomial
        
    Note: only works for p < 5.
    Returns a dictionary containing the critical values for 1% 5% 10% 90% 95% 99%
    quintiles.
    
    Adapted from: https://www.mathworks.com/matlabcentral/fileexchange/45093-time-frequency-generalized-phase-synchrony-for-eeg-signal-analysis/content/TF%20Generalized%20Phase%20Synchrony/ztcrit.m 
    '''
    text_zt ='-2.63467   -1.95254   -1.62044   0.910216    1.30508    2.08088 -3.63993   -2.94935   -2.61560  -0.369306 -0.0116304   0.666745 -4.20045   -3.54490   -3.21450   -1.20773  -0.896215  -0.237604 -4.65813   -3.99463   -3.66223   -1.69214   -1.39031  -0.819931 -5.07175   -4.39197   -4.03090   -2.06503   -1.78329   -1.21830 -5.45384   -4.73277   -4.39304   -2.40333   -2.15433   -1.62357 -5.82090   -5.13053   -4.73415   -2.66466   -2.39868   -1.88193 -2.53279   -1.94976   -1.62656   0.915249    1.31679    2.11787 -3.56634   -2.93701   -2.61518  -0.439283 -0.0498821   0.694244 -4.08920   -3.46145   -3.17093   -1.25839  -0.919533  -0.298641 -4.56873   -3.89966   -3.59161   -1.72543   -1.44513  -0.894085 -4.97062   -4.33552   -4.00795   -2.12519   -1.85785   -1.30566 -5.26901   -4.62509   -4.29928   -2.42113   -2.15002   -1.65832 -5.54856   -4.95553   -4.63476   -2.71763   -2.46508   -1.99450 -2.60249   -1.94232   -1.59497   0.912961    1.30709    2.02375 -3.43911   -2.91515   -2.58414  -0.404598 -0.0481033   0.538450 -4.00519   -3.46110   -3.15517   -1.25332  -0.958071  -0.320677 -4.46919   -3.87624   -3.58887   -1.70354   -1.44034  -0.920625 -4.84725   -4.25239   -3.95439   -2.11382   -1.85495   -1.26406 -5.15555   -4.59557   -4.30149   -2.41271   -2.19370   -1.70447 -5.46544   -4.89343   -4.58188   -2.74151   -2.49723   -2.02390 -2.58559   -1.94477   -1.62458   0.905676    1.30371    2.01881 -3.46419   -2.91242   -2.58837  -0.410558 -0.0141618   0.665034  -4.00090   -3.45423   -3.16252   -1.24040  -0.937658  -0.304433 -4.45303   -3.89216   -3.61209   -1.74246   -1.48280  -0.906047 -4.79484   -4.22115   -3.92941   -2.11434   -1.83632   -1.30274 -5.15005   -4.58359   -4.30336   -2.44972   -2.21312   -1.68330 -5.42757   -4.88604   -4.60358   -2.74044   -2.50205   -2.04008 -2.65229   -1.99090   -1.66577   0.875165    1.27068    2.04414 -3.49260   -2.87595   -2.56885  -0.416310 -0.0488941   0.611200 -3.99417   -3.42290   -3.13981   -1.25096  -0.950916  -0.310521  -4.42462   -3.85645   -3.56568   -1.73108   -1.45873  -0.934604 -4.72243   -4.22262   -3.94435   -2.10660   -1.84233   -1.26702 -5.12654   -4.55072   -4.24765   -2.43456   -2.18887   -1.73081 -5.46995   -4.87930   -4.57608   -2.71226   -2.48367   -2.00597 -2.63492   -1.96775   -1.62969   0.904516    1.31371    2.03286 -3.44558   -2.84182   -2.57313  -0.469204  -0.128358   0.553411 -3.99140   -3.41543   -3.13588   -1.23585  -0.944500  -0.311271 -4.43404   -3.84922   -3.56413   -1.73854   -1.48585  -0.896978 -4.75946   -4.19562   -3.91052   -2.09997   -1.86034   -1.32987 -5.14042   -4.56772   -4.25699   -2.43882   -2.18922   -1.67371 -5.39389   -4.85343   -4.57927   -2.73497   -2.49921   -2.00247 -2.58970   -1.95674   -1.61786   0.902516    1.32215    2.05383 -3.44036   -2.86974   -2.58294  -0.451590 -0.0789340   0.631864 -3.95420   -3.43052   -3.13924   -1.23328  -0.938986  -0.375491 -4.40180   -3.79982   -3.52726   -1.71598   -1.44584  -0.885303 -4.77897   -4.21672   -3.93324   -2.12309   -1.88431   -1.33916 -5.13508   -4.56464   -4.27617   -2.44358   -2.18826   -1.72784 -5.35071   -4.82097   -4.54914   -2.73377   -2.48874   -2.01437 -2.60653   -1.96391   -1.63477   0.890881    1.29296    1.97163 -3.42692   -2.86280   -2.57220  -0.463397 -0.0922419   0.613101 -3.99299   -3.41999   -3.13524   -1.23857  -0.929915  -0.337193 -4.41297   -3.83582   -3.55450   -1.72408   -1.44915  -0.872755 -4.75811   -4.18759   -3.92599   -2.12799   -1.88463   -1.37118 -5.08726   -4.53617   -4.26643   -2.44694   -2.19109   -1.72329 -5.33780   -4.82542   -4.54802   -2.73460   -2.50726   -2.02927 -2.58687   -1.93939   -1.63192   0.871242    1.26611    1.96641 -3.38577   -2.86443   -2.57318  -0.391939 -0.0498984   0.659539 -3.93785   -3.39130   -3.10317   -1.24836  -0.956349  -0.334478 -4.39967   -3.85724   -3.55951   -1.74578   -1.46374  -0.870275 -4.74764   -4.20488   -3.91350   -2.12384   -1.88202   -1.36853 -5.07739   -4.52487   -4.25185   -2.43674   -2.22289   -1.72955 -5.36172   -4.81947   -4.53837   -2.74448   -2.51367   -2.03065 -2.58364   -1.95730   -1.63110   0.903082    1.28613    2.00605 -3.45830   -2.87104   -2.59369  -0.451613  -0.106025   0.536687 -3.99783   -3.43182   -3.16171   -1.26032  -0.956327  -0.305719 -4.40298   -3.86066   -3.56940   -1.74588   -1.48429  -0.914111 -4.84459   -4.23012   -3.93845   -2.15135   -1.89876   -1.39654 -5.10571   -4.56846   -4.28913   -2.47637   -2.22517   -1.79586 -5.39872   -4.86396   -4.58525   -2.78971   -2.56181   -2.14042'
    zt = []
    for i in text_zt.split(' '):
        if i == '':
            continue
        zt.append(np.float(i))
    zt = np.matrix(zt).reshape((int(420/6),6))
    
    i = np.round(nobs/50)+1
    if nobs < 50:
        i = i - 1
    if i > 10:
        i = 10
        
    i = (i-1)*7 + p + 2

    keys = '1% 5% 10% 90% 95% 99%'.split(' ')
    val  = zt[i-1,]
    critical_values = {}
    for k,v in zip(keys,val.tolist()[0]):
        critical_values[k] = v

    return critical_values
    
    
def tdiff(x,k):
    '''tdiff function
    Perform difference in a vector of 1-dim.
    '''
    
    if type(x) == type(pd.Series([])):
        x = x.values
    if type(x) == type([]):
        x = np.array(x)
        
    nobs = np.shape(x)[0]
    if k == 0:
        return x
    elif k == 1:
        return x[1:nobs] - x[:nobs-1]
    else:
        return x[k:nobs] - x[:nobs-k]

def trimr(x,n1,n2):
    '''trimr function
    Returns a matrix or vector striped of specified rows.
    '''
    if type(x) == type(pd.Series([])):
        x = x.values
    
    nobs = np.shape(x)[0]
    if (n1+n2) >= nobs:
        print('Error: Attempting to trim too much.')
        return None
        
    h1, h2 = n1, nobs-n2
    
    if n2 == 0:
        return x[h1:]
    return np.array(x[h1:h2])
    
def lag(x,k):
    '''lag function
    Returns a lag vector (by k positions)
    '''
    nobs = np.shape(x)[0]
    return [np.float('nan')]*k+list(x[:nobs-k])

def ptrend(p, nobs):
    '''ptrend (polynomial trend) function
    Returns explainatory matrix containing polynomial time-trend
    '''
    u = nobs*[1]
    timep = [np.array(u)]
    if p > 0:
        t = np.arange(1,nobs+1) / nobs
        m = 1
        while m <= p:
            timep.append(np.array(list(map(lambda x: x**2, t))))
            m += 1
    return timep
    
def detrend(y, p):
    '''detrend function
    Detrend a time series matrix using regression of y against a polynomial 
    time trend of order p. 
    
    Inputs:
        y <-- input matrix of time-series
        p = 0 substract mean
        p = 1 constant plus trend model
        p > 1 higher order oplynimial model
        p = -1 returns y
    '''
    
    if p == -1:
        return y
        
    nobs = np.shape(y)[0]
    
    xmat = pd.DataFrame([])
    cont = 0
    for i in ptrend(p, nobs):
        xmat[str(cont)] = i
        cont += 1
    
    xmat = np.asmatrix(xmat.values)
    
    beta = np.linalg.inv(xmat.T * xmat) * (xmat.T * y)
    return y - xmat*beta
    

class ols():
    
    desc = 'Ordinary Least Squares Estimator'
    
    def __init__(self, y,variables):
        self.y = y
        self.x = variables
        self.b = np.linalg.inv(self.x.T * self.x) * self.x.T * self.y
        self.res = self.y - self.x * self.b
        self.detrend_res = detrend(self.y,0) - detrend(self.x,0) * self.b
        self.so = np.float((self.detrend_res.T * self.detrend_res) / (len(self.y)-np.shape(self.x)[1]))
        self.var_cov = self.so * np.linalg.inv(self.x.T * self.x)
        self._adf = np.float(self.b[0] / np.sqrt(self.var_cov[0,0]))
        
        
def adf(x,p,nlag,silence=False):
    '''adf function
    Perform a Dickey-Fuller test on a time series vector.
    
    Inputs
    - x: Pandas series vector
    - p: order of time pl¿olynomial in the null-hypothesis
            p = -1; no deterministic part
            p = 0 ; constant term
            p = 1 ; constant plus time-trend
            p > 1 ; higher order polynomial 
    - nlag: number of lagged changes of x
    '''
    from numpy.matlib import ones
    # error checking 
    
    if p < -1:
        print('Error: p cannot be less than -1.')
        return None
    elif len(np.shape(x)) != 1:
        print('Error: x must be a pandas timeseries object with 1 dimension')
        return None
    
    # get number of observations 
    nobs = np.shape(x)[0]
    
    if (nobs - 2*nlag) + 1 < nlag:
        print('Error: nlag too large; negatide dof.')
        return None
    
    dep = trimr(x,1,0)
    ch  = tdiff(x,1)
    #ch  = trimr(ch,1,0)
    
    k = 0
    z = []

    while k < nlag:
        k += 1
        z.append(lag(ch, k))
    
    z   = list(map(lambda x: np.array(trimr(x,k,0)),z)) #trimr(z,k,0)
    dep = trimr(dep,k,0)
    
    if p > -1:
        [z.append(i) for i in ptrend(p,len(z[0]))]
    
    
    ylag = lag(dep,1)
    ylag = np.asmatrix(trimr(ylag,1,0)).T
    z2   = np.asmatrix(list(map(lambda x: trimr(x,1,0), z))).T
    # NOTE: ERROR IN DOCUMENTATION 
    y2   = np.asmatrix(trimr(ch,2,0)).T
    regressor = np.concatenate((ylag, z2),1)
    results   = ols(y2, regressor)
    results_  = ols(y2, np.concatenate((ylag, ones(ylag.shape)), 1))
    halflife  = -np.log(2) / results_.b[0]
    statistical_test = results._adf
    
    critic_d = ztcrit(nobs,p)
    if not silence:
        print('\nAugmented Dickey-Fuller Test for unit root variable.\n')
        print('\t >> ADF t-statistic: {:.4f}'.format(statistical_test))
        print('\t >> Number of lags: {}'.format(nlag))
        print('\t >> AR(1) Estimates: [lambda = {:.4f}, lambda + 1 = {:.4f}]'.format(results.b[0,0],results.b[0,0]+1))
        print('\t >> Half-life (days): {}\n'.format(halflife))
        print('Critical Values:\n')
        [print('\t >> {} Critic Value: {:.4f}'.format(i,critic_d[i])) for i in ['1%', '5%', '10%']]
        print('\nConclusion (90% of confidence): \n')
        if statistical_test > critic_d['10%']:
            print('\t >> The price series is not mean regresive.')
        else:
            print('\t >> The price series is mean regresive.')
        
    if statistical_test > critic_d['10%']:
        return 0
    return 1
    

# %% Test ADF

# %% 


# %%

# %%  