# -*- coding: cp1252 -*-
#
# Created by Pádraig Mac Carron
#
################################
#Import Libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import poisson
################################



"""This script works out the parameters of a compound Poisson model using MLE.

The parameters outputted are the means for each of the Poisson distributions.


---------------------

Note:
When this script is used, if any means it ouptuts are very similar you need
to remove any parameter guesses that are close to each other until it gets unique means.

Look at the AIC or BIC for each number of clusters, if it stabilises and there are
two almost identical means then we can safely remove one

"""

###################
#Create random data

n = 1000

#We create a distribution of 4 Poissons
X = np.array([np.random.poisson(5) for _ in range(n)] 
             + [np.random.poisson(15) for _ in range(n)]
             + [np.random.poisson(50) for _ in range(n)]
             + [np.random.poisson(150) for _ in range(n)])

#We don't want any negatives
X[X<=0] = 1

x,f = np.unique(X,return_counts=True)
p = f/n


###################

def lnL(x0):
    """The log-likelihood function.

    This uses the global variable X for the data

    Parameters
    ----------
    x0 : array like
        The parameter guesses for the compound Poisson distribution
        (i.e. the mean for each Poisson). 
  
    Returns
    -------
    Log-likelihood of the distribution for the estimates
    """
    
    n_dists = len(x0)
    
    d1 = poisson.pmf(X,x0[0])
    for i in range(1,n_dists):
        d1 += poisson.pmf(X,x0[i])
    d1 = d1[np.nonzero(d1)]

    return -1*np.sum(np.log(d1))


#####################


# These are the initial estimates
#    Note from the data above these are "bad" estimates
#    Also put parameter guesses in increasing order!
x0 = [3,20,40,200,300,500]


#This loop tries to work out the optimal number of distributions by checking
#    the Akaike Information Criterion while consecutively adding estimates
#   (can also do BIC instead of AIC, they generally yield the same results)
l_max = []
for i in range(len(x0)):
    l = minimize(lnL,x0[:i+1],method='SLSQP')['fun']
    if np.isnan(l):
        l = np.inf
    AIC = 2*l + 2*(i+1) + 2*(i+1)*(i+2)/(X.size-i-2)
    print(i+1,AIC)
    #BIC = 2*l + (i+1)*np.log(X.size) 
    #print(i+1,BIC)
    l_max += [AIC]


#This is the "optimal" number of clusters (see above for issues)
n_c = l_max.index(min(l_max)) + 1
print("\nn_clusts =", n_c)


#This computes the maximum likelihood estimates for the optimal number of clusters
#    -- kind of inefficient as I calculate this in the loop above....
est_poiss = minimize(lnL,x0[:n_c],method='SLSQP')

means = est_poiss['x']
#log_L = -1*est_poiss['fun']
print("means = ", np.round(means,3))
print("AIC =", min(l_max))


########################
#Plot

x_n = range(0,int(X.max()))
y_est = 0
for l in means:
    y_est += poisson.pmf(x_n,l)


plt.figure(figsize=(9,7))
plt.scatter(x,p,6,'k')
plt.plot(x_n,y_est)


plt.show() 

