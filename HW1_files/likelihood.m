function l = likelihood(b,X,Y)
%This function compute the log-likelihood of b
l = -(sum(log(normcdf(X*b)).*Y) + sum(log(1 - normcdf(X*b)).* (1 -Y)));
