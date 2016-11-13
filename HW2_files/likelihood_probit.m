function L = likelihood_probit(Y,theta)
%This function takes in data Y and theta=[alpha,rho,gamma_L,pi]
%and return the likelihood calculated as L = sum(ln(li))

alpha  = theta(1);
rho    = theta(2);
gamma_L= theta(3);
tao    = theta(4);
pi     = abs(tao)/(1+abs(tao));
gamma_H=-pi*gamma_L/(1-pi);


[nN,nT] = size(Y);
nT = nT-1;
l_1    = ones(nN,1);

%The tmp is gamma(alpha+rho*y_t-1 + c)
tmp     = gamma_val(rho*Y+(gamma_L+alpha)*repmat(l_1,1,nT+1));
tmp_L   = [zeros(nN,1),tmp(:,1:nT)];
tmp     = gamma_val(rho*Y+(gamma_H+alpha)*repmat(l_1,1,nT+1));
tmp_H   = [zeros(nN,1),tmp(:,1:nT)];

likelihood_L = (tmp_L.^Y).* ((1-tmp_L).^(1-Y));
likelihood_H = (tmp_H.^Y).* ((1-tmp_H).^(1-Y));
y_0 = Y(:,1);
prob_y0_L = l_1 * gamma_val(alpha+gamma_L)/...
    (1-gamma_val(alpha+gamma_L+rho)+gamma_val(alpha+gamma_L));
prob_y0_H = l_1 * gamma_val(alpha+gamma_H)/...
    (1-gamma_val(alpha+gamma_H+rho)+gamma_val(alpha+gamma_H));

likelihood_L(:,1) = (prob_y0_L.^(y_0)).* ((1-prob_y0_L).^(1-y_0));
likelihood_H(:,1) = (prob_y0_H.^(y_0)).* (1-prob_y0_H).^(1-y_0);

likelihood = pi * prod(likelihood_L,2) + (1 - pi) * prod(likelihood_H,2);
like = -sum(log(likelihood));
L = max(like,-realmax);