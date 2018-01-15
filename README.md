# clustering_methods
MLE and Jenks algorithms


This repository contains the data and scripts for analysing the dataset groups.csv used in Dunbar & Mac Carron "The Tradeoff Between Fertility and Predation Risk Drives a Geometric Sequence in the Pattern of Group Sizes in Baboons" submitted to Biology Letters. 


The Jenks script uses https://github.com/perrygeo/jenks and should be self-explanatory.


The MLE_func script is more complicated. This tests for Power, Exponential, Truncated exponential, Weibull, Normal (Gaussian), Log-normal, geometric, negative binomial and between 1 and n Poisson distributions.

Initially this was developed for testing degree distrutions in networks and the compound Poisson was not there. 
However after finding a multiple-peaked distribution a referee made me test for compound distributions also.
Only Poisson is used as that has a single parameter, therefore two Poisson and a log-normal have the same number of parameters and even using a third we are less likely to overfit. If it was a compound of two negative binomials this would already have four parameters which is getting a quite high and likely to result in over-fitting.

One must take care when interpretting the Compound Poisson as it can output two (or more) Poisson's with the same mean in which case one will need to be removed.
There is a script with an example for just the Compound Poisson which should be studied before applying the full MLE function to get a better idea of what it's doing. It is a good idea to plot the distribution first too to see if it really could have multiple peaks.


Finally the MLE_func does not do a great job with errors and will output many warnings.
The error estimates are rough and some distributions and I have not implimented errors in some distriubtions yet.
Therefore if you need highly accurate parameters (if you're working on critical exponents for example) then this script is not for you, if you want the most likely distribution though and just need a good idea of the paramter values this script should be fine however.
