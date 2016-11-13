function avg_par_IQ = funIQ(theta,X)
nobs = length(X);
avg_par_IQ = normpdf(X*theta)'*(theta(3)*ones(nobs,1))/nobs;