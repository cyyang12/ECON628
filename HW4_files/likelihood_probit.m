function L = likelihood_probit(Y,Q,theta)
%This function takes in data Y and theta=[alpha,rho,gamma_L,pi]
%and return the likelihood calculated as L = sum(ln(li))

alpha  = theta(1);
rho    = theta(2);
sigma  = theta(3);

[nN,nT] = size(Y);
nT      = nT-1;
R       = size(Q(:,1));
l_1     = ones(nN,1);
y_0     = Y(:,1);

%Calculate the likelihood for each c
likelihood = zeros(size(Q));
for r = 1:size(Q(:,1))
    c = Q(r,:)*sigma;
    prob_y0   = l_1 * gamma_val(alpha+c)/...
        (1-gamma_val(alpha+rho+c)+gamma_val(alpha+c));
    tmp       = gamma_val(rho*Y+ repmat(c,nT+1,1)' +alpha *repmat(l_1,1,nT+1));
    tmp       = [zeros(nN,1),tmp(:,1:nT)];

    prob      = (tmp.^Y).* ((1-tmp).^(1-Y));
    prob(:,1) = (prob_y0.^(y_0)).* ((1-prob_y0).^(1-y_0));
    likelihood(r,:)= prod(prob,2); %Calculate for each round of simulation
                                   %contain nN firms
end
likelihood = mean(likelihood);
likelihood = -log(likelihood);
L          = sum(likelihood);

