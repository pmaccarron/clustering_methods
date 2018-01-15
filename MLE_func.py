# -*- coding: cp1252 -*-
#
# Created by Pádraig Mac Carron

########################
#Import Libraries
import numpy as np
from scipy.optimize import minimize
from scipy.special import zeta
from scipy.misc import factorial
from scipy.stats import poisson
import math
########################


########################
#MLE function


def MLE(degree,k_min=2,k_max=None,stretch=False,cum=True,AIC_thres=0.01,print_params=False):
    """Maximum Likelihood Estimates for a list of discrete integers from a distribution.

    Parameters
    ----------
    degree : array like
        The list/array of degree values, must be integers.
    k_min : int (optional, default: 2)
        The minimum value for which to estimate the parameters for.
    k_max : int (optional, default: None)
        The maximum value to fit to in case there's some sprurious outliers.
    stretch : bool (optional, default: False)
        If True this includes a stretched expontential
    AIC_thres : float (optional, default = 0.01)
        If the AIC weight is less than this it will not report it
    print_params : bool (optional, default: False)
        If true it will output the parameters of each model

    Returns
    -------
    MLE : tuple
        ( list for each candidate : [[paramters, Likelihood, name, error]],
        list of most likely candidates : [(name, AIC,parameters]))


    """

    degree = np.array(degree)
    if k_max == None:
        k_max = degree.max()
        
    k, n_k = np.unique(degree, return_counts=True)
    k_new = np.array([i for i in degree if i >= k_min and i <= k_max])

    #Length
    n = len(k_new)
    sum_k = np.sum(k_new)   
    sum_log = np.sum(np.log(k_new))
    k_mean = float(sum_k)/n
    stdev = k_new.std()

    if k_max < 1000:
        length = 1000
    else:
        length = k_max + 1000

    inf = np.arange(length)

    #x axis
    x = np.arange(k_min,k_max+0.25,0.25)


    ###########
    #log-normal

    def norm_ln(mu,sigma):
        return np.sum( (1.0/(inf+k_min)) * np.exp(-((np.log(inf+k_min)-mu)**2)/(2*sigma**2) ) )

    lognorm = lambda x: -1*( - n*np.log(norm_ln(x[0],x[1])) - sum_log - np.sum( ((np.log(k_new)-x[0])**2)/(2*x[1]**2) ) )

    est_lognorm = minimize(lognorm,[(np.log(k_new)).mean(),(np.log(k_new)).std()],method='TNC',bounds=[(0.,np.log(k_mean)+10),(0.01,np.log(k_new.std())+10)])

    mu = est_lognorm['x'][0]
    sigma = est_lognorm['x'][1]


    #Error is rough as it doesn't include normalisation
    err_mu = sigma/(n**0.5)
    err_sig = sigma**2/( (3*n*mu - n*sigma**2 + 6*mu*sum_log - 3*((np.log(k_new))**2).sum() )**0.5)

    if print_params == True:
        print("log-normal: mu =", round(mu,2), u"±", round(err_mu,2), " sigma =",round(sigma,2), u"±", round(err_sig,2))

    L_lognormal = [est_lognorm['x'],-1*est_lognorm['fun'],'log-normal',(err_mu,err_sig)]


    ############
    #power

    power = lambda g: (n*np.log(zeta(g[0],k_min) )) + g[0]*sum_log

    #bounds work for 'SLSQP','TNC' and 'L-BFGS-B'
    est_power = minimize(power,[2.],method='SLSQP',bounds=[(0.5,4)])
    gamma = est_power['x'][0]

    #Error
    zeta_kmin = zeta(gamma,k_min)

    first_deriv = np.sum(-(np.log(k_min+inf))*(k_min+inf)**(-gamma))
    second_deriv = np.sum(((np.log(k_min+inf))**2)*(k_min+inf)**(-gamma))

    error_power = (n*( (second_deriv/zeta_kmin) - ((first_deriv/zeta_kmin)**2)))**(-0.5)
    if print_params == True:
        print("Power: exponent =", round(gamma,2), u"±", round(error_power,2))

    L_power = [est_power['x'],-1*est_power['fun'],'power',error_power]


    ############
    #Exponential


    expon = lambda kappa: -n*(np.log(1-np.exp(-1/kappa[0]))) + (1/kappa[0])*(sum_k - n*k_min)

    est_exp = minimize(expon,k_mean,method='SLSQP',bounds=[(0.5,k_mean+20)])
    kappa = est_exp['x'][0]
        
    #Error
    error_exp = (-n*( 2*(k_min - sum_k)/(kappa**3)
                      + ( ( 2*(np.exp(1/kappa) - 1)*kappa - np.exp(1/kappa) )
                          / ( ((np.exp(1/kappa) - 1)**2 )*(kappa**4) ) ) ))**(-0.5)

    if print_params == True:
        print("exponential: kappa=", round(kappa,2), u"±", round(error_exp,2))

    L_exponential = [est_exp['x'],-1*est_exp['fun'],'exponential',error_exp]

    ###########
    #Truncated

    #Hurwitz Zeta function with exponential
    def zet(k,gamma,kappa,length):
        if kappa == 0:
            kappa += 0.001
        return np.sum( ((k+np.arange(length))**-gamma)*np.exp(-np.arange(length)/kappa) )


    trunc = lambda x: -1*(n*k_min/x[0] - n*np.log(zet(k_min,x[1],x[0],k_max)) - x[1]*sum_log - sum_k/x[0])

    est_trunc = minimize(trunc,[k_mean,2.],method='SLSQP',bounds=[(0.5,k_mean+30),(0.5,4)])
    gamma = est_trunc['x'][1]
    kappa = est_trunc['x'][0]


    #Error
    zeta_kmin = zet(k_min,gamma,kappa,length)
    S_dev_g1 = np.sum(-(np.log(((k_min+inf)**(-gamma))))*((k_min+inf)**(-gamma))*(np.exp(-inf/kappa)))
    S_dev_g2 = np.sum(((np.log(((k_min+inf)**(-gamma))))**2)*((k_min+inf)**(-gamma))*(np.exp(-inf/kappa)))
    S_dev_k1 = np.sum((-inf)*((k_min+inf)**(-gamma))*(np.exp(-inf/kappa)))
    S_dev_k2 = np.sum(((-inf)**2)*((k_min+inf)**(-gamma))*(np.exp(-inf/kappa)))
        
    error_tg = (n * ( (S_dev_g2/zeta_kmin)  -  (S_dev_g1/zeta_kmin)**2) )**(-0.5)
    error_tk = (n * (  -(2/(kappa**3)*(k_min - sum_k)) +  (S_dev_k2/zeta_kmin)  -  (S_dev_k1/zeta_kmin)**2) )**(-0.5)

    if print_params == True:
        print("truncated: gamma =", round(gamma,2), u"±", round(error_tg,2), " kappa =", round(kappa,2), u"±", round(error_tk,2))

    L_trunc = [est_trunc['x'],-1*est_trunc['fun'],'truncated',(error_tg,error_tk)]

    ###########
    #Weibull


    def norm_w(b,kap):
        return np.sum((((k_min+inf)/kap)**(b-1))*np.exp(-((k_min+inf)/kap)**b ))
    weibull = lambda x: -1*(-n*np.log(norm_w(x[0],x[1])) - np.sum( (k_new/x[1])**x[0] ) -n*(x[0]-1)*np.log(x[1])+(x[0]-1)*sum_log)

    est_weibull = minimize(weibull,[1.,k_mean],method='SLSQP',bounds=[(0.05,4.),(0.5,None)])
    beta = est_weibull['x'][0]
    kappa = est_weibull['x'][1]


    #Error  (These assume k_min = 1...)
    err_beta = (np.sum(((np.log( np.array(k_new)/kappa ))**2) * (np.array(k_new)/kappa)**beta))**(-0.5)
    err_kap = (kappa*(1-beta+beta*(beta+1)*(np.sum((np.array(k_new)/kappa)**beta))))**(-0.5)

    if print_params == True:
        print("Weibull: beta =", round(beta,2), u"±", round(err_beta,2), " kappa =", round(kappa,2), u"±", round(err_kap,2))

    L_weibull = [est_weibull['x'],-1*est_weibull['fun'],'Weibull',(err_beta,err_kap)]

    ###########
    #Stretched

    #Stretched exponential gives similar fit to Weibull
    # and it is also the cumulative of the Weibull

    if stretch == True:
        def norm_s(b,kap):
            return np.sum( np.exp(-((k_min+inf)/kap)**b ))


        stretched = lambda x: -1*( -n*np.log(norm_s(x[0],x[1])) - np.sum((k_new/x[1])**x[0]))

        est_stretch = minimize(stretched,[1.,k_mean],method='SLSQP',bounds=[(0.05,4.),(0.5,None)])
        beta = est_stretch['x'][0]
        kappa = est_stretch['x'][1]

        if print_params == True:
            print("Stretch: beta =", round(beta,2), " kappa =", round(kappa,2))

        L_stretch = [est_stretch['x'],-1*est_stretch['fun'],'stretched exponential',0.0]

    #########
    #Poisson
        
    def lnL_poisson(x0):
        d1 = poisson.pmf(k_new,k_new.mean())
        #d1[d1==0] = 1
        d1 = d1[np.nonzero(d1)]
        return -1*np.sum(np.log(d1))
    est_poisson = minimize(lnL_poisson,k_new.mean(),method='SLSQP')

    L_poisson = [est_poisson['x'],-1*est_poisson['fun'],'Poisson']
    #    print("Poisson error")


    #######
    #Normal

    def norm_n(mu,sigma):
        return np.sum( np.exp( -((inf-mu)**2)/(2*sigma**2) ))

    normal = lambda x: n*np.log(norm_n(x[0],x[1])) + np.sum(((k_new-x[0])**2)/(2*x[1]**2))

    est_normal = minimize(normal,[k_mean,stdev],method='SLSQP',bounds=[(0.,k_mean+10),(0.1,None)])
    mu = est_normal['x'][0]
    sigma = est_normal['x'][1]
    
    #Error is rough as it doesn't include normalisation
    err_mu = sigma/(n**0.5)
    err_sig = sigma**2/( (-3*n*mu+n*(sigma**2) +6*mu*sum_k -3*np.sum(k_new**2))**0.5)

    if print_params == True:
        print("Normal: mu =", round(mu,2), " sigma =", round(sigma,2))

    L_normal = [est_normal['x'],-1*est_normal['fun'],'Gaussian',(err_mu,err_sig)]


    #######
    #Geometric
   
    p_est = n/(n+np.sum(k_new))
    #lnL_geom = -1*np.log(1-p_est)*np.sum(k_new) + n*(p_est-np.log(1-p_est))

    lnL = lambda p: -1*np.sum(np.log( p* (1-p)**(k_new-1)))
    est_geom = minimize(lnL,[n/(n+np.sum(k_new))],method='SLSQP',bounds=[(0.,1.)])
    p_est = est_geom['x'][0]
    lnL_geom = est_geom['fun']
    if print_params == True:
        print("Geometric: p =", round(p_est,4))
    
    L_geom = [[p_est],-1*lnL_geom,'Geometric']


    #######
    #Negative Binomial

    '''Again consider implementing the normalisation with regard to k_min'''

    n_nb = lambda m,s: (m**2) / ((s**2) - m)
    p_nb = lambda m,s: 1 - m/(s**2)
    
    from scipy.special import binom
    def lnL_negbin(pars):
        d = binom(k_new+pars[0]-1,k_new)*(pars[1]**k_new)*((1-pars[1])**pars[0])
        ###This makes the 0 values arbitrarily small
        d[d==0] = 1e-323
        ###Instead can use the following that removes the 0 values
        #d = d[np.nonzero(d)]
        return -1*np.sum(np.log(d))

    est_negbin = minimize(lnL_negbin,[n_nb(k_new.mean(),k_new.std()),p_nb(k_new.mean(),k_new.std())],method='SLSQP')
    L_negbin = [est_negbin['x'],-1*est_negbin['fun'],'Negative Binomial']

  
    ######
    #Compound poisson

    '''Again the normalisation should consider k_min'''

    def lnL(x0):
        n_dists = len(x0)
        d1 = poisson.pmf(k_new,x0[0])
        for i in range(1,n_dists):
            d1 += poisson.pmf(k_new,x0[i])
        d1[d1==0] = 1e-323
        #d1 = d1[np.nonzero(d1)]
        return -1*np.sum(np.log( d1 ))

    x0 = [5,15,50,150,500,1500]
    l_max = []
    for i in range(len(x0)):
        l = minimize(lnL,x0[:i+1],method='SLSQP')['fun']
        if np.isnan(l):
            l = np.inf
        bic = 2*l + len(x0[:2*(i+1)])*np.log(k_new.size) 
        l_max += [bic]

    n_c = l_max.index(min(l_max)) + 1
    

    est_poiss = minimize(lnL,x0[:n_c],method='SLSQP')

    if print_params == True:
        print("compound Poisson:",n_c, np.around(est_poiss['x'],3))

    L_cpoiss = [est_poiss['x'], -1*est_poiss['fun'],'Compound Poisson',]
   
    
    ####################
    #AIC Weights and BIC

    lnL = [L_power,L_exponential,L_trunc,L_weibull,L_normal,L_lognormal,
           L_geom,L_negbin,L_poisson,
           #L_cln,#L_cnorm,
           L_cpoiss,#L_nb
           ]
    if k_max <= 170:
        lnL += [L_poisson]
    if stretch == True:
        lnL += [L_stretch]

    AIC = []
    for i in lnL:
        if math.isnan(i[1]):
            i[1] = -1e+10
        AIC += [-2*i[1] + 2*(len(i[0])-2) + (2*(len(i[0])-2)*((len(i[0])-2)+1))/(n-1-(len(i[0])-2))]
        print("AIC",i[2],AIC[-1])
    AIC_min = min(AIC)

    w_total = 0.0
    for i in AIC:
        w_total += np.exp(- (i- AIC_min)/2)

    weights = []
    for i in AIC:
        weights += [np.exp( -(i-AIC_min)/2 ) / w_total]


    most_likely = []
    for i in range(len(weights)):
        if weights[i] > AIC_thres:
            most_likely += [(lnL[i][2],weights[i],lnL[i][0],lnL[i][-1])]


    print("\nAIC, highest weights: (closer to 1 is best)")
    for i in most_likely:
        print(i[0],round(i[1],3),"; parameters:",np.around(np.array(i[2]),2))#,u"±",np.around(np.array(i[3]),2))

    print("AIC most likely:", lnL[weights.index(max(weights))][2])


    #####
    #BIC


    BIC = []
    for i in lnL:
        if math.isnan(i[1]):
            i[1] = -1e+10
        BIC += [-2*i[1] + (len(i[0])-2)*np.log(n)]

    BIC_min = min(BIC)

    BIC_likely = []
    BIC_weights = []
    b = 0.0
    for i in range(len(BIC)):
        b = BIC[i] - BIC_min
        BIC_weights += [b]
        if b < 10:
            BIC_likely += [(lnL[i][2],b)]


    print("\nBIC, best candidates: (closer to 0 is best)")
    for i,j in BIC_likely:
        print(i,round(j,3))

    print("BIC most likely:", lnL[BIC_weights.index(min(BIC_weights))][2])

    return(lnL,most_likely)




