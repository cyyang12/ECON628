%%

N     = 10000;
alpha =  0;
beta  =  2;
c_0   = -2;
c_1   =  2;
p     = 0.5; %probability z = 1
epsilon  = randn(N,1); 
xi = 1/2 * randn(N,1);  
z  = binornd(1,p,N,1); 

%%
C  = exp(-c_0 - c_1*z - xi);
d  = [alpha+c_0 + log(exp(beta)-1) + c_1*z+epsilon+xi > 0];
pi = exp(alpha + beta * d + epsilon);

%%
%(2.B)
X_ols = [ones(N,1),d];
beta_ols = (X_ols'*X_ols)^(-1)*(X_ols'*log(pi))

%(2.C)
%The OLS estimator is upward biased for beta. 
%This is because d_i is positively correlated with beta.

%(2.D)
%IV can consistently estimate beta, since z as an IV 
%only effect through d.

%(2.E)
X_iv = [ones(N,1),z];
beta_iv = (X_iv'*X_ols)^(-1)*(X_iv'*log(pi))

%%
%(3.A)

N     = 10000;
alpha =  0;

c_0   = -2;
c_1   =  2;
p     = 0.5; %probability z = 1
epsilon  = randn(N,1); 
xi = 1/2 * randn(N,1);  

beta  = 1 + randn(N,1);

%The first data set
z_1  = binornd(1,p,N,1); 
C_1  = exp(-c_0 - c_1*z_1 - xi);
d_1  = [alpha+c_0 + log(exp(beta)-1) + c_1*z_1+epsilon+xi > 0];
pi_1 = exp(alpha + beta .* d_1 + epsilon);


%The second data set
c_0   = -4;
z_2  = binornd(1,p,N,1); 
C_2  = exp(-c_0 - c_1*z_2 - xi);
d_2  = [alpha+c_0 + log(exp(beta)-1) + c_1*z_2+epsilon+xi > 0];
pi_2 = exp(alpha + beta .* d_2 + epsilon);


%%
%(3.B)
X_ols_1 = [ones(N,1),d_1];
beta_ols_1 = (X_ols_1'*X_ols_1)^(-1)*(X_ols_1'*log(pi_1))


X_ols_2 = [ones(N,1),d_2];
beta_ols_2 = (X_ols_2'*X_ols_2)^(-1)*(X_ols_2'*log(pi_2))


%(3.C)
%The OLS bias is upward. But since beta has variation
%It will add more bias compare to the static case.

%%
%(3.D)
X_iv_1 = [ones(N,1),z_1];
beta_iv_1 = (X_iv_1'*X_ols_1)^(-1)*(X_iv_1'*log(pi_1))

X_iv_2 = [ones(N,1),z_2];
beta_iv_1 = (X_iv_2'*X_ols_2)^(-1)*(X_iv_2'*log(pi_2))

%%(3.F)

