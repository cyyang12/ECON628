%%ECON 628
%ASSIGNMENT 2
%Jasmine Hao

%%Q1.2
clear all;
%Parameters
alpha   = -1.0;
rho     = 0.5;
gamma_L = -1.0;
gamma_H = 1.0;
pi      = 0.5;

%%Draw Y_1
nN      = 20000;
nT      = 2;

%Draw u and calculate realization of c
u  = rand(nN,1);
c  = ones(nN,1)*gamma_L + (u>pi)* (gamma_H-gamma_L);

%Draw probability p_i0
p_0 = rand(nN,1);
threshold = gamma_val(c+alpha)./...
    (1-gamma_val(c+alpha+rho)+gamma_val(c+alpha));
y_0 = p_0 <= threshold;

%Generate y_it
Y_1= zeros(nN,nT+1);
y_lag = y_0;
Y_1(:,1) = y_lag;
for t=1:nT
   p_t = rand(nN,1);
   threshold = gamma_val(c + alpha + rho*y_lag);
   y_lag = p_t <= threshold; 
   Y_1(:,t+1) = y_lag;
end

%%Draw Y_2
nN      = 4000;
nT      = 10;

%Draw u and calculate realization of c
u  = rand(nN,1);
c  = ones(nN,1)*gamma_L + (u>pi)* (gamma_H-gamma_L);

%Draw probability p_i0
p_0 = rand(nN,1);
threshold = gamma_val(c+alpha)./...
    (1-gamma_val(c+alpha+rho)+gamma_val(c+alpha));
y_0 = p_0 <= threshold;

%Generate y_t
Y_2= zeros(nN,nT+1);
y_lag = y_0;
Y_2(:,1) = y_lag;
for t=1:nT
   p_t = rand(nN,1);
   threshold = gamma_val(c + alpha + rho*y_lag);
   y_lag = p_t <= threshold; 
   Y_2(:,t+1) = y_lag;
end


%save data file
filename='C:\Users\haoja\Dropbox\Dropbox\ECON 628\HW2_files\data2.mat';
save(filename)

%%Estimation
theta_0 = [-1,0.5,-1,-1];

%%Q1.4 
%Write a function m-file that compute the likelihood value 
%given theta

%First data set
func = @(b)likelihood_probit(Y_1,b);
[theta_1_1,fval,exitflag,output,grad,hessian]= fminunc(func,theta_0);
hessian = hessian /nN;
se_theta_1_1 = sqrt(diag(hessian));

fprintf('Theta 1 using random effect model with (T=2,N=20000)\n');
for i = 1:length(theta_1_1)
    fprintf('Estimation=%.4f, S.D=%.4f\n',theta_1_1(i),se_theta_1_1(i));
end

%Second data set
func = @(b)likelihood_probit(Y_2,b);
[theta_1_2,fval,exitflag,output,grad,hessian]= fminunc(func,theta_0);
hessian = hessian /nN;
se_theta_1_2 = sqrt(diag(hessian));

fprintf('Theta 1 using random effect model with (T=10,N=4000)\n');
for i = 1:length(theta_1_2)
    fprintf('Estimation=%.4f, S.D=%.4f\n',theta_1_2(i),se_theta_1_2(i));
end

%%Q1.5
%Suppose initial y_i0 are independent of c_i

%First data set
func = @(b)likelihood_probit_2(Y_1,b);
[theta_2_1,fval,exitflag,output,grad,hessian]= fminunc(func,theta_0);
hessian = hessian /nN;
se_theta_2_1 = sqrt(diag(hessian));

fprintf('Theta 2 with (T=2,N=20000)\n');
for i = 1:length(theta_2_1)
    fprintf('Estimation=%.4f, S.D=%.4f\n',theta_2_1(i),se_theta_2_1(i));
end

%Second data set
theta_01 = theta_0 = [-1,0.5,-1,-1];

func = @(b)likelihood_probit_2(Y_2,b);
[theta_2_2,fval,exitflag,output,grad,hessian]= fminunc(func,theta_0);
hessian = hessian /nN;
se_theta_2_2 = sqrt(diag(hessian));

fprintf('Theta 2 with (T=10,N=4000)\n');
for i = 1:length(theta_2_2)
    fprintf('Estimation=%.4f, S.D=%.4f\n',theta_2_2(i),se_theta_2_2(i));
end

%%Q1.6
%Suppose a researcher thought that there is no permanent unobserved heterogeneity so that
% c_i = gamma_L = gamma_H = 0 for all i.

%First data set
func = @(b)likelihood_probit_3(Y_1,b);
[theta_3_1,fval,exitflag,output,grad,hessian]= fminunc(func,theta_0);
hessian = hessian /nN;
se_theta_3_1 = sqrt(diag(hessian));

fprintf('Theta 3 with (T=2,N=20000)\n');
for i = 1:2
    fprintf('Estimation=%.4f, S.D=%.4f\n',theta_3_1(i),se_theta_3_1(i));
end

%Second data set
func = @(b)likelihood_probit_3(Y_2,b);
[theta_3_2,fval,exitflag,output,grad,hessian]= fminunc(func,theta_0);
hessian = hessian /nN;
se_theta_3_2 = sqrt(diag(hessian));

fprintf('Theta 3 with (T=10,N=4000)\n');
for i = 1:2
    fprintf('Estimation=%.4f, S.D=%.4f\n',theta_3_2(i),se_theta_3_2(i));
end

%%Q1.7
%Construct a data set from the subset of observations i's in the data set 
%(T;N) = (10; 4000) wim sum(y_it) = 3

index = (sum(Y_2,2)==3);
Y_3   = Y_2(index,:);
func = @(b)likelihood_probit_3(Y_3,b);
[theta_4_1,fval,exitflag,output,grad,hessian]= fminunc(func,theta_0);
hessian = hessian /nN;
se_theta_4_1 = sqrt(diag(hessian));

fprintf('Theta with sum(y)=3 \n');
for i = 1:2
    fprintf('Estimation=%.4f, S.D=%.4f\n',theta_4_1(i),se_theta_4_1(i));
end

index = (sum(Y_2,2)==4);
Y_3   = Y_2(index,:);
func = @(b)likelihood_probit_3(Y_3,b);
[theta_4_2,fval,exitflag,output,grad,hessian]= fminunc(func,theta_0);
hessian = hessian /nN;
se_theta_4_2 = sqrt(diag(hessian));

fprintf('Theta with sum(y)=4 \n');
for i = 1:2
    fprintf('Estimation=%.4f, S.D=%.4f\n',theta_4_2(i),se_theta_4_2(i));
end


index = (sum(Y_2,2)==5);
Y_3   = Y_2(index,:);
func = @(b)likelihood_probit_3(Y_3,b);
[theta_4_3,fval,exitflag,output,grad,hessian]= fminunc(func,theta_0);
hessian = hessian /nN;
se_theta_4_3 = sqrt(diag(hessian)); 

fprintf('Theta with sum(y)=5 \n');
for i = 1:2
    fprintf('Estimation=%.4f, S.D=%.4f\n',theta_4_3(i),se_theta_4_3(i));
end


index = (sum(Y_2,2)==6);
Y_3   = Y_2(index,:);
func = @(b)likelihood_probit_3(Y_3,b);
[theta_4_4,fval,exitflag,output,grad,hessian]= fminunc(func,theta_0);
hessian = hessian /nN;
se_theta_4_4 = sqrt(diag(hessian)); 

fprintf('Theta with sum(y)=6 \n');
for i = 1:2
    fprintf('Estimation=%.4f, S.D=%.4f\n',theta_4_4(i),se_theta_4_4(i));
end


%%Q1.8
%Conterfactual
%(a) Compute a fraction of exporters
alpha_2   = theta_1_2(1);
rho_2     = theta_1_2(2);
gamma_L_2 = theta_1_2(3);
tao_2     = theta_1_2(4);
pi_2      = abs(tao_2)/(1+abs(tao_2));
gamma_H_2 = -pi*gamma_L_2/(1-pi_2);

%generate Y_3
nT        = 10;
nN        = 50000;

%Draw u and calculate realization of c
u  = rand(nN,1);
c  = ones(nN,1)*gamma_L_2 + (u>pi)* (gamma_H_2-gamma_L_2);

%Draw probability p_i0
p_0 = rand(nN,1);
threshold = gamma_val(c+alpha_2)./...
    (1-gamma_val(c+alpha_2+rho_2)+gamma_val(c+alpha_2));
y_0 = p_0 <= threshold;

%Generate y_it
Y_3= zeros(nN,nT+1);
y_lag = y_0;
Y_3(:,1) = y_lag;
for t=1:nT
   p_t = rand(nN,1);
   threshold = gamma_val(c + alpha_2 + rho*y_lag);
   y_lag = p_t <= threshold; 
   Y_3(:,t+1) = y_lag;
end

Y_3_mean = mean(Y_3);
for i = 1:length(Y_3_mean)
    fprintf('Export percentage in time %d : %.2f\n',i-1,Y_3_mean(i)*100);
end

%(b)Generate contrafactual fraction
p_0 = rand(nN,1);
threshold = gamma_val(c+alpha_2)./...
    (1-gamma_val(c+alpha_2+rho_2)+gamma_val(c+alpha_2));
y_0 = p_0 <= threshold;

%Generate y_it
Y_3_c= zeros(nN,nT+1);
y_lag = y_0;
Y_3_c(:,1) = y_lag;
for t=1:nT
   p_t = rand(nN,1);
   threshold = gamma_val(c + alpha_2+1 + rho*y_lag);
   y_lag = p_t <= threshold; 
   Y_3_c(:,t+1) = y_lag;
end

Y_3_c_mean = mean(Y_3_c);
for i = 1:length(Y_3_c_mean)
    fprintf('Export percentage in time %d : %.2f\n',i-1,Y_3_c_mean(i)*100);
end

%(c)Use the mispecified model to predict contrafactual
alpha_3   = theta_3_2(1);
rho_3     = theta_3_2(2);
gamma_L_3 = 0;
gamma_H_3 = 0;

%Use the new coefficient to generate data
u  = rand(nN,1);
c  = ones(nN,1)*gamma_L_3 + (u>pi)* (gamma_H_3-gamma_L_3);

Y_3_m= zeros(nN,nT+1);
y_lag = y_0;
Y_3_m(:,1) = y_lag;
for t=1:nT
   p_t = rand(nN,1);
   threshold = gamma_val(c + alpha_3+1 + rho*y_lag);
   y_lag = p_t <= threshold; 
   Y_3_m(:,t+1) = y_lag;
end

% Report counterfactual fraction under mispecified model
Y_3_m_mean = mean(Y_3_m);
for i = 1:length(Y_3_m_mean)
    fprintf('Export percentage in time %d : %.2f\n',i-1,Y_3_m_mean(i)*100);
end

%%Q1.9

%Make a table to report the simulated fraction

Y_mean = mean(Y_2);

fprintf('time \t (a) \t (b) \t (c) \t true\n');

for i = 1:length(Y_3_m_mean)
   fprintf('%d    \t %.2f \t %.2f \t %.2f \t %.2f\n',...
       i-1,Y_3_mean(i)*100,Y_3_c_mean(i)*100,...
       Y_3_m_mean(i)*100,Y_mean(i)*100);
end

index = linspace(0,10,11);
plot(index,Y_3_mean,index,Y_3_c_mean,...
    index,Y_3_m_mean,index,Y_mean);
legend('Estimate','Contrafactual','Mispecified','True');
%%Q1.10
%To get 90% confidence interval from Q1.4, can do the simulation
%many times and get the approximate distribution of theta.
%Then use the 0.95 and 0.05 quantile as the confidence interval.


