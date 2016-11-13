function L = likelihood_probit(Y,theta)
%This function takes in data Y and theta=[alpha,rho,gamma_L,pi]
%and return the likelihood calculated as L = sum(ln(li))

alpha  = theta(1);
rho    = theta(2);

[nN,nT] = size(Y);
nT = nT-1;
%The tmp is gamma(alpha+rho*y_t-1 + c)
tmp  = gamma_val(rho*Y+alpha);
tmp  = [zeros(nN,1),tmp(:,1:nT)];
likelihood = Y .* log(tmp) + (1 -Y) .* log(1-tmp);

likelihood(:,1) = 0;
L = -sum(sum(likelihood));