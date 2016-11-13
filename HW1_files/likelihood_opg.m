function l = likelihood_opg(b,X,Y)

l = -(sum(log(normcdf(X*b)).*Y) + sum(log(1 - normcdf(X*b)).* (1 -Y)));
