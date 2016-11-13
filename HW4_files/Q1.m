%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%    Problem Set 4
%    Q1
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all;
B = 1000;

cd('C:\Users\haoja\Dropbox\Dropbox\ECON 628\HW1_files\'); 
addpath('C:\Users\haoja\Dropbox\Dropbox\ECON 628\HW1_files\');

load('Data_HW1'); 
%Draw e_i from i = 1,..., N.
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

theta = zeros(nparm,1); 
options = optimset('Display','iter','TolX',1e-6,'TolFun',1e-6, 'MaxIter', 10000,'MaxFunEvals',10000);
f_ml  = @(b)likelihood(b,X,Y); 
[theta_hat,fval,exitflag,output,grad,hessian] = fminunc(f_ml,theta,options);
hessian = hessian / nobs;
avar    = hessian^(-1);
se   = sqrt(diag(avar));
%This is for 1.3
ape_iq = normpdf(X*theta_hat)'*(theta_hat(3)*ones(nobs,1))/nobs;;
fun        = @(b)funIQ(b,X);
G          = gradient(fun,theta_hat);
avar_ape_iq= G'*avar*G;
se_ape_iq = sqrt(avar_ape_iq);
%%1.1 - 1.2
%Bootstrap
%for CI of thetas
T1_1 = zeros(B,1); 
T2_1 = zeros(B,1); 

T1_2 = zeros(B,1); 
T2_2 = zeros(B,1); 

T1_3 = zeros(B,1); 
T2_3 = zeros(B,1); 

%for CI of APE

T_ape_1 = zeros(B,1);
T_ape_2 = zeros(B,1); 
T_ape_3 = zeros(B,1); 

T_ape_par_1 = zeros(B,1);
T_ape_par_2 = zeros(B,1); 
T_ape_par_3 = zeros(B,1); 
for k = 1: B
    u = ceil(rand(nobs,1) * nobs);
    epsilon = randn(nobs,1);
    X_b = zeros(size(X));
    Y_b = zeros(size(Y));

    for i = 1:nobs
        bi = u(i);
        X_b(bi,:) = X(bi,:);
        Y_b(bi,:) = Y(bi,:);
    end
    
    f_ml_b  = @(theta_0)likelihood(theta_0,X_b,Y_b); 
    [theta_hat_b,fval,exitflag,output,grad,hessian] =...
        fminunc(f_ml_b,theta,options);
    hessian = hessian / nobs;
    avar_b    = hessian^(-1);
    se_b   = sqrt(diag(avar_b));
    ape_iq_b = normpdf(X*theta_hat_b)'*(theta_hat_b(3)*ones(nobs,1))/nobs;
    fun        = @(b)funIQ(b,X_b);
    G          = gradient(fun,theta_hat_b);
    avar_ape_iq_b= G'*avar_b*G;
    se_ape_iq_b = sqrt(avar_ape_iq_b);
    
    T1_1(k) = theta_hat_b(2);
    T2_1(k) = theta_hat_b(3);
    T_ape_1(k) = ape_iq_b;
    
    T1_2(k) = theta_hat_b(2) - theta_hat(2);
    T2_2(k) = theta_hat_b(3) - theta_hat(3);
    T_ape_2(k) = ape_iq_b - ape_iq;
    
    T1_3(k) = (theta_hat_b(2) - theta_hat(1))/se_b(2);
    T2_3(k) = (theta_hat_b(3) - theta_hat(2))/se_b(3);
    T_ape_3(k) = (ape_iq_b - ape_iq)/se_ape_iq_b;
    
    %%%%Parametric Bootstrap%%%
    
    Y_par_b = (X_b * theta_hat + epsilon>0);
    f_ml_par_b  = @(theta_0)likelihood(theta_0,X_b,Y_par_b); 
    [theta_hat_par_b,fval,exitflag,output,grad,hessian] =...
        fminunc(f_ml_b,theta,options);
    hessian = hessian / nobs;
    avar_par_b    = hessian^(-1);
    se_par_b   = sqrt(diag(avar_par_b));
    ape_iq_par_b = normpdf(X*theta_hat_par_b)'*(theta_hat_par_b(3)*ones(nobs,1))/nobs;
    fun        = @(b)funIQ(b,X_b);
    G          = gradient(fun,theta_hat_par_b);
    avar_ape_iq_par_b= G'*avar_b*G;
    se_ape_iq_par_b = sqrt(avar_ape_iq_par_b);
    T_ape_par_1(k) = ape_iq_par_b;
    T_ape_par_2(k) = ape_iq_par_b - ape_iq; 
    T_ape_par_3(k) = (ape_iq_par_b - ape_iq)/se_ape_iq_par_b; 
end

T1_1 = sort(T1_1,'ascend');
T2_1 = sort(T2_1,'ascend');

T1_2 = sort(T1_2,'ascend');
T2_2 = sort(T2_2,'ascend');

T1_3 = sort(T1_3,'ascend');
T2_3 = sort(T2_3,'ascend');

T_ape_1 = sort(T_ape_1,'ascend');
T_ape_2 = sort(T_ape_2,'ascend'); 
T_ape_3 = sort(T_ape_3,'ascend'); 

T_ape_par_1 = sort(T_ape_par_1,'ascend'); 
T_ape_par_2 = sort(T_ape_par_2,'ascend'); 
T_ape_par_3 = sort(T_ape_par_3,'ascend'); 


C1_low_1 = T1_1(ceil(B*0.025));
C1_high_1 = T1_1(ceil(B*0.975));
C2_low_1 = T2_1(ceil(B*0.025));
C2_high_1 = T2_1(ceil(B*0.975));

C1_low_2 = theta_hat(1) - T1_2(ceil(B*0.975));
C1_high_2 = theta_hat(1) - T1_2(ceil(B*0.025));
C2_low_2 = theta_hat(2) - T2_2(ceil(B*0.975));
C2_high_2 = theta_hat(2) - T2_2(ceil(B*0.025));

C1_low_3 = theta_hat(1) - se(2)*T1_3(ceil(B*0.975));
C1_high_3 = theta_hat(1) - se(2)*T1_3(ceil(B*0.025));
C2_low_3 = theta_hat(2) - se(3)*T2_3(ceil(B*0.975));
C2_high_3 = theta_hat(2) - se(3)*T2_3(ceil(B*0.025));


%%1.1 - 1.2 Print
fprintf('Confidence Interval 1 for nearc4 [%f , %f]\n',C1_low_1,C1_high_1);
fprintf('Confidence Interval 1 for IQ [%f , %f]\n',C2_low_1,C2_high_1);
fprintf('Confidence Interval 2 for nearc4 [%f , %f]\n',C1_low_2,C1_high_2);
fprintf('Confidence Interval 2 for IQ [%f , %f]\n',C2_low_2,C2_high_2);
fprintf('Confidence Interval 3 for nearc4 [%f , %f]\n',C1_low_3,C1_high_3);
fprintf('Confidence Interval 3 for IQ [%f , %f]\n',C2_low_3,C2_high_3);

%%1.3 Confidence interval for average partial effect
fprintf('Confidence Interval 1 for Average Partial Effect of IQ [%f, %f]\n',...
    T_ape_1(ceil(B*0.025)), T_ape_1(ceil(B*0.975)));
fprintf('Confidence Interval 2 for Average Partial Effect of IQ [%f, %f]\n',...
    ape_iq - T_ape_2(ceil(B*0.975)), ape_iq - T_ape_2(ceil(B*0.025)));
fprintf('Confidence Interval 3 for Average Partial Effect of IQ [%f, %f]\n',...
    ape_iq - se_ape_iq*T_ape_3(ceil(B*0.975)), ...
    ape_iq - se_ape_iq*T_ape_3(ceil(B*0.025)));

%%1.4Parametric Bootstrap
fprintf('Confidence Interval 1 for parametric Average Partial Effect of IQ [%f, %f]\n',...
    T_ape_par_1(ceil(B*0.025)), T_ape_par_1(ceil(B*0.975)));
fprintf('Confidence Interval 2 for parametric Average Partial Effect of IQ [%f, %f]\n',...
    ape_iq - T_ape_par_2(ceil(B*0.975)), ape_iq - T_ape_par_2(ceil(B*0.025)));
fprintf('Confidence Interval 3 for parametric Average Partial Effect of IQ [%f, %f]\n',...
    ape_iq - se_ape_iq*T_ape_par_3(ceil(B*0.975)), ...
    ape_iq - se_ape_iq*T_ape_par_3(ceil(B*0.025)));
