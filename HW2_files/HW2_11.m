clear all;
load('HW2_data.mat');

nN = 718;
nT = 6;
Y  = zeros(nN,nT);
%Organize data
for i = 1:length(data)
    t = data(i,2);
    n = data(i,1);
    Y(n,t) = data(i,3);
end
theta_0 = [-1,0.5,-1,-1];

%%
%Random effect model
func = @(b)likelihood_probit(Y,b);
[theta_1,fval,exitflag,output,grad,hessian]= fminunc(func,theta_0);
hessian = hessian /nN;
se_theta_1 = sqrt(diag(hessian));

fprintf('Theta 1 using random effect model with (T=2,N=20000)\n');
for i = 1:length(theta_1)
    fprintf('Estimation=%.4f, S.D=%.4f\n',theta_1(i),se_theta_1(i));
end

%%
%Y_i0 independent of c_i
func = @(b)likelihood_probit_2(Y,b);
[theta_2,fval,exitflag,output,grad,hessian]= fminunc(func,theta_0);
hessian = hessian /nN;
se_theta_2 = sqrt(diag(hessian));

fprintf('Theta 2 with (T=2,N=20000)\n');
for i = 1:length(theta_2)
    fprintf('Estimation=%.4f, S.D=%.4f\n',theta_2(i),se_theta_2(i));
end

%%
%c_i = gamma_H = gamma_L = 0
func = @(b)likelihood_probit_3(Y,b);
[theta_3,fval,exitflag,output,grad,hessian]= fminunc(func,theta_0);
hessian = hessian /nN;
se_theta_3 = sqrt(diag(hessian));

fprintf('Theta 3 with (T=2,N=20000)\n');
for i = 1:2
    fprintf('Estimation=%.4f, S.D=%.4f\n',theta_3(i),se_theta_3(i));
end

%%
%Construct data set with sum(y_it)=3,4,5,6
index = (sum(Y,2)==3);
Y_3   = Y(index,:);
func = @(b)likelihood_probit_3(Y_3,b);
[theta_4_1,fval,exitflag,output,grad,hessian]= fminunc(func,theta_0);
hessian = hessian /nN;
se_theta_4_1 = sqrt(diag(hessian));

fprintf('Theta with sum(y)=3 \n');
for i = 1:2
    fprintf('Estimation=%.4f, S.D=%.4f\n',theta_4_1(i),se_theta_4_1(i));
end

index = (sum(Y,2)==4);
Y_3   = Y(index,:);
func = @(b)likelihood_probit_3(Y_3,b);
[theta_4_2,fval,exitflag,output,grad,hessian]= fminunc(func,theta_0);
hessian = hessian /nN;
se_theta_4_2 = sqrt(diag(hessian));

fprintf('Theta with sum(y)=4 \n');
for i = 1:2
    fprintf('Estimation=%.4f, S.D=%.4f\n',theta_4_2(i),se_theta_4_2(i));
end


index = (sum(Y,2)==5);
Y_3   = Y(index,:);
func = @(b)likelihood_probit_3(Y_3,b);
[theta_4_3,fval,exitflag,output,grad,hessian]= fminunc(func,theta_0);
hessian = hessian /nN;
se_theta_4_3 = sqrt(diag(hessian)); 

fprintf('Theta with sum(y)=5 \n');
for i = 1:2
    fprintf('Estimation=%.4f, S.D=%.4f\n',theta_4_3(i),se_theta_4_3(i));
end


index = (sum(Y,2)==6);
Y_3   = Y(index,:);
func = @(b)likelihood_probit_3(Y_3,b);
[theta_4_4,fval,exitflag,output,grad,hessian]= fminunc(func,theta_0);
hessian = hessian /nN;
se_theta_4_4 = sqrt(diag(hessian)); 

fprintf('Theta with sum(y)=6 \n');
for i = 1:2
    fprintf('Estimation=%.4f, S.D=%.4f\n',theta_4_4(i),se_theta_4_4(i));
end


%%Counterfactual
%
%(a) Compute a fraction of exporters
alpha   = theta_1(1);
rho     = theta_1(2);
gamma_L = theta_1(3);
tao     = theta_1(4);
pi      = abs(tao)/(1+abs(tao));
gamma_H = -pi*gamma_L/(1-pi);

%generate Y_3
%Draw u and calculate realization of c
u  = rand(nN,1);
c  = ones(nN,1)*gamma_L + (u>pi)* (gamma_H-gamma_L);

%Draw probability p_i0
p_0 = rand(nN,1);
threshold = gamma_val(c+alpha)./...
    (1-gamma_val(c+alpha+rho)+gamma_val(c+alpha));
y_0 = p_0 <= threshold;

%Generate y_it
Y_3= zeros(nN,nT+1);
y_lag = y_0;
Y_3(:,1) = y_lag;
for t=1:nT
   p_t = rand(nN,1);
   threshold = gamma_val(c + alpha + rho*y_lag);
   y_lag = p_t <= threshold; 
   Y_3(:,t+1) = y_lag;
end

Y_3_mean = mean(Y_3);
for i = 1:length(Y_3_mean)
    fprintf('Export percentage in time %d : %.2f\n',i-1,Y_3_mean(i)*100);
end

%(b)Generate contrafactual fraction
p_0 = rand(nN,1);
threshold = gamma_val(c+alpha)./...
    (1-gamma_val(c+alpha+rho)+gamma_val(c+alpha));
y_0 = p_0 <= threshold;

%Generate y_it
Y_3_c= zeros(nN,nT+1);
y_lag = y_0;
Y_3_c(:,1) = y_lag;
for t=1:nT
   p_t = rand(nN,1);
   threshold = gamma_val(c + alpha+1 + rho*y_lag);
   y_lag = p_t <= threshold; 
   Y_3_c(:,t+1) = y_lag;
end

Y_3_c_mean = mean(Y_3_c);
for i = 1:length(Y_3_c_mean)
    fprintf('Export percentage in time %d : %.2f\n',i-1,Y_3_c_mean(i)*100);
end

%(c)Use the mispecified model to predict contrafactual
alpha_3   = theta_3(1);
rho_3     = theta_3(2);
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

Y_mean = mean(Y);

fprintf('time \t (a) \t (b) \t (c) \t true\n');

for i = 1:length(Y_3_m_mean)
   fprintf('%d    \t %.2f \t %.2f \t %.2f \t %.2f\n',...
       i-1,Y_3_mean(i)*100,Y_3_c_mean(i)*100,...
       Y_3_m_mean(i)*100,Y_mean(i)*100);
end

index = linspace(0,6,7);
plot(index,Y_3_mean,index,Y_3_c_mean,...
    index,Y_3_m_mean,index,Y_mean);
legend('Estimate','Contrafactual','Mispecified','True');