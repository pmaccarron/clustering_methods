# -*- coding: cp1252 -*-
#
# Created by Pádraig Mac Carron

########################
#Import Libraries
from __future__ import print_function
import numpy as np
from MLE_func import MLE
################################

'''
    This uses the MLE function in the imported script and test for various distributions

    Currently the error calculations do not work well and produce a vast amount of Warnings.
    For some distributions I have not implemented the error at all.



    If two or more AIC or BIC values are close to each other then either distribution is supported



    Note
    ----

    This was initially developed for testing network degree distributions using the discrete
    parts of Clauset et al (2007). All these test for discrete (i.e. not continuous approximations)
    distributions. Hence this only works for integers.

    Therefore the normal and log-normal are not the true normal and log-normal we all know,
    as they are discrete versions the normalisation is different as it is now summing over
    the distribution rather than integrating from -ininity to +infinity.

    


    Examples
    --------
    
    The first example below generates log-normal data and then (should) correctly identify
    the distribution. Including the argument print_params=True will output the error estimates
    for the parameters however the sigma error will probably have got a warning will output 0


    The second example uses the group size data and test the different distributions.
    I lowered the AIC threshold for the AIC weights but it still vastly prefers one distribution



'''


####################
#lognormal test

print("MLE for log-normal data\n\n")

n = 1000
X = np.array([np.random.lognormal(5,1) for _ in range(n)])

#These have to be integers as we are testing for discrete distributions 
X = np.rint(X)


est = MLE(X)
#While running the MLE function it will output values, est is a tuple of two arrays,
# the first is all the parameters and likelihoods, the second is just the information
# for the most likely distributions

print("\n\n\nParameters and their associated errors:")
print(est[-1])

###################
#read data

print("\n\n--------------------------\n\nMLE for group size data\n\n")

X = []
with open('groups.csv') as f:
    for line in f:
        if len(line.strip())>0:
            X.append(float(line.strip()))    
X = np.array(X)

#These have to be integers as we are testing for discrete distributions
# While the group size should of course be positive, there are some cases where
#  it was not exact so the mean observed size over the study period was reported
X = np.rint(X)

est = MLE(X,AIC_thres=0.001)


