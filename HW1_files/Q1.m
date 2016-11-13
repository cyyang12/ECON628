clear all;
%ECON 628
%Problem Set 1
%Question 1

cd('C:\Users\haoja\Dropbox\Dropbox\ECON628\HW1_files\'); 
addpath('C:\Users\haoja\Dropbox\Dropbox\ECON628\HW1_files\');

load('Data_HW1'); 
%(a) Draw e_i from i = 1,..., N.
nobs = size(data,1);    % the number of observations = 1457 
% read the variables from ``data''
college = data(:,1);
nearc4 = data(:,2);
IQ = data(:,3);
motheduc = data(:,4);
fatheduc = data(:,5);
reg662 = data(:,6);
reg663 = data(:,7);
reg664 = data(:,8);
reg665 = data(:,9);
reg666 = data(:,10);
reg667 = data(:,11);
reg668 = data(:,12);
reg669 = data(:,13);
lwage = data(:,14); % this variable is not used in this exercise
exper = data(:,15); % this variable is not used in this exercise
expersq = data(:,16); % this variable is not used in this exercise
smsa = data(:,17); % this variable is not used in this exercise
south = data(:,18);  % this variable is not used in this exercise
nparm = 13;             % the number of parameters to be estiamted
Y = zeros(nobs,1);      % allocate memory  
X = zeros(nobs,nparm);  % allocate memory   
Y(:) = college;       % dummy variable for college attendence
X(:,1) = ones(nobs,1);  % the first column is constant term
X(:,2:nparm) = [nearc4,IQ,motheduc,fatheduc,reg662,reg663,reg664,reg665,reg666,reg667,reg668,reg669]; 


e = randn(nobs,1);
theta = [-6.76, 0.34, 0.04, 0.07, 0.11, 0.3, 0.36, 0.51, 0.67, 0.88, 0.83, 1.18, 0.56]';
Y_tilde = (X*theta + e > 0);

%(b) The function is written in likelihood.m

%(c) Use minifuc to estimate ML estimate
b = zeros(nparm,1); 
% using fminunc
options = optimset('Display','iter','TolX',1e-6,'TolFun',1e-6, 'MaxIter', 10000,'MaxFunEvals',10000);
f_ml_1  = @(b)likelihood(b,X,Y_tilde); 
[theta_hat_1,fval] = fminunc(f_ml_1,b,options);

%(d)Proof written.

%(e)Compute the asymptotic variance
Y_hat  = normcdf(X*theta_hat_1,0,1);
Y_hat_d= normpdf(X*theta_hat_1,0,1);
Y_hat_dd=Y_hat_d.*(X*theta_hat_1);

%grad   = X'*[(Y_tilde.*Y_hat_d)./Y_hat - ((1- Y_tilde).*Y_hat_d)./Y_hat]/nobs;
grad   = gradient(f_ml_1,theta_hat_1);
temp_1 = repmat([(Y_tilde.*Y_hat_d)./Y_hat - ((1- Y_tilde).*Y_hat_d)./Y_hat],1,nparm);
grad_2 = (temp_1.*X)'*(temp_1.*X)/nobs;
temp_2 = repmat([(Y_tilde.*(Y_hat.*Y_hat_dd - Y_hat_d.*Y_hat_d)./(Y_hat.^(2)))-...
    ((1-Y_tilde).*((1-Y_hat).*Y_hat_dd + Y_hat_d.^(2))./((1-Y_hat).^(2)))],1,13);
hessian= X'*(temp_2.*X)/nobs;

avar_1 = (-1)*hessian^(-1)/nobs;
avar_2 = grad_2/nobs;
avar_3 = [hessian]^(-1)*(grad_2)*[hessian]^(-1)/nobs;

std_1  = sqrt(diag(avar_1));
std_2  = sqrt(diag(avar_2));
std_3  = sqrt(diag(avar_3));

t_90   = norminv(0.95);

for i = 1:nparm
    fprintf('[%d,%d],%d \n',...
        theta_hat_1(i)-t_90*std_1(i), theta_hat_1(i)+t_90*std_1(i),...
        theta(i));
end

for i = 1:nparm
    fprintf('[%d,%d],%d \n',...
        theta_hat_1(i)-t_90*std_1(i), theta_hat_1(i)+t_90*std_2(i),...
        theta(i));
end

for i = 1:nparm
    fprintf('[%d,%d],%d \n',...
        theta_hat_1(i)-t_90*std_1(i), theta_hat_1(i)+t_90*std_3(i),...
        theta(i));
end

%(f)Asymptotic variance

%(2)Generate psuedo data using e~N(0,2)
e = sqrt(2) * e;
Y_tilde_2 = (X*theta + e > 0);
f_ml_2    = @(b)likelihood(b,X,Y_tilde_2); 
[theta_hat_2] = fminunc(f_ml_2,b,options);

%The estimate is different from the previous, because
%the variance of error term is larger, thus the variance 
%of the estimated parameter is larger.

%(3)Apply the estimation to Y
options = optimset('Display','iter','TolX',1e-6,'TolFun',1e-6, 'MaxIter', 10000,'MaxFunEvals',10000);
f_ml  = @(b)likelihood(b,X,Y); 
[theta_hat,fval] = fminunc(f_ml,b,options);

%Calculate asymptotic variance
Y_hat  = normcdf(X*theta_hat,0,1);
Y_hat_d= normpdf(X*theta_hat,0,1);
Y_hat_dd=Y_hat_d.*(X*theta_hat);
grad   = X'*[(Y.*Y_hat_d)./Y_hat - ((1- Y).*Y_hat_d)./Y_hat]/nobs;
temp_1 = repmat([(Y.*Y_hat_d)./Y_hat - ((1- Y).*Y_hat_d)./Y_hat],1,nparm);
grad_2 = (temp_1.*X)'*(temp_1.*X)/nobs;
temp_2 = repmat([(Y.*(Y_hat.*Y_hat_dd - Y_hat_d.*Y_hat_d)./(Y_hat.^(2)))-...
    ((1-Y).*((1-Y_hat).*Y_hat_dd + Y_hat_d.^(2))./((1-Y_hat).^(2)))],1,13);
hessian= X'*(temp_2.*X)/nobs;
avar   = [hessian]^(-1)*(grad_2)*[hessian]^(-1)/nobs;
std    = sqrt(diag(avar));

%(4)The average effect of IQ
avg_par_IQ = funIQ(theta_hat,X);
fun        = @(b)funIQ(b,X);
G          = gradient(fun,theta_hat);
avar_par_IQ= G'*avar*G;

%Compare with OLS
theta_ols=(inv(X'*X))*X'*Y;
e_ols=Y-X*theta_ols;
se_ols=sqrt(diag((1/1457)*(e_ols'*e_ols)*(inv(X'*X))));

avg_par_eff_IQ_ols=theta_ols(3);
se_ols_IQ=se_ols(3);

%(5)The average effect of proximity
X_near4c  = X(X(:,2)==0,:);
X_near4c_1= X_near4c;
X_near4c_1(:,2)=1;
par_nearc4 = sum(normcdf(X_near4c_1 * theta_hat,0,1) - ...
              normcdf(X_near4c * theta_hat,0,1))/length(X_near4c);
fun      = @(b) sum(normcdf(X_near4c_1 * b,0,1) - ...
              normcdf(X_near4c * b,0,1))/length(X_near4c);
grad_near4c = gradient(fun,theta_hat);
avar_near4c = grad_near4c' * avar * grad_near4c;

par_nearc4_ols=theta_ols(2);
se_ols_nearc4=se_ols(2);