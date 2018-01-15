# -*- coding: cp1252 -*-
#
# Created by Pádraig Mac Carron
#
################################
#Import Libraries
from __future__ import print_function
import numpy as np
from jenks import jenks
################################


'''
    This script will perform Jenks natural breaks algorithm on the dataset read below.

    It requires jenks from   https://github.com/perrygeo/jenks
    which uses Cython so is more efficient than pure Python methods

    --------

    
    The goodness of fit threshold chosen can be changed from 0.85 below

    Coulson (1987) recommends 0.85, can increase/decrease for different percision
    As it approaches 1, the number of clusters will approach the system size

    In mmost datasets I've used increasing it to a stricter 0.9 makes little difference
'''

##############
#Parameters


#Goodness of fit threshold
thres = 0.85


#The maximum number of clusters you want to test for
end = 15


##############
#Functions
def goodness_of_variance_fit(array, classes):
    '''This and the next function were written by camdenl:
        https://stats.stackexchange.com/questions/143974/jenks-natural-breaks-in-python-how-to-find-the-optimum-number-of-breaks/144075
    '''
    
    # get the break points
    classes = jenks(array, classes)

    # do the actual classification
    classified = np.array([classify(i, classes) for i in array])

    # max value of zones
    maxz = max(classified)

    # nested list of zone indices
    zone_indices = [[idx for idx, val in enumerate(classified) if zone + 1 == val] for zone in range(maxz)]

    # sum of squared deviations from array mean
    sdam = np.sum((array - array.mean()) ** 2)

    # sorted polygon stats
    array_sort = [np.array([array[index] for index in zone]) for zone in zone_indices]

    # sum of squared deviations of class means
    sdcm = sum([np.sum((classified - classified.mean()) ** 2) for classified in array_sort])

    # goodness of variance fit
    gvf = (sdam - sdcm) / sdam

    return gvf


def classify(value, breaks):
    for i in range(1, len(breaks)):
        if value < breaks[i]:
            return i
    return len(breaks) - 1


###################
#read data

X = []
with open('groups.csv') as f:
    for line in f:
        if len(line.strip())>0:
            X.append(float(line.strip()))
        
X = np.array(X)

###########
#Jenks

gvf = 0.0
for k in range(2,end):
    gvf = goodness_of_variance_fit(X, k)
    if round(gvf,2) >= thres:
        break


print(k)
breaks = jenks(X,k)
print("breaks =",breaks)

print("\nmean    clust_n")
means = []
for i in range(1,k+1):
    if i == 1:
        X_ = X[X<= breaks[i]]
        means += [X_.mean()]
        print(round(X_.mean(),2)," ",X_.size)
    else:
        x = X[X > breaks[i-1]]
        X_ = x[x<=breaks[i]]
        means += [X_.mean()]
        print(round(X_.mean(),2)," ",X_.size)



