%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%    Problem Set 4
%    Q2
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%Read Data
clear all;
load('C:\Users\haoja\Dropbox\Dropbox\ECON 628\HW2_files\HW2_data.mat');

nN = 718;
nT = 6;
R  = 500;
Y  = zeros(nN,nT);
%Organize data
for i = 1:length(data)
    t = data(i,2);
    n = data(i,1);
    Y(n,t) = data(i,3);
end
theta_0 = [-1,0.5,1];

%%
%Draw eta
Q_1 = randn(R,nN);
%Try
Q_2 = randn(R/2,nN);
Q_2 = vertcat(Q_2,Q_2);
%%
%Random effect model
func     = @(b)likelihood_probit(Y,Q_1,b);
[theta_1,fval,exitflag,output,grad,hessian]= fminunc(func,theta_0);
hessian  = hessian /nN;
se_theta = sqrt(diag(hessian));


for i = 1:length(theta_1)
    fprintf('Estimation=%.4f, S.D=%.4f\n',theta_1(i),se_theta(i));
end

func     = @(b)likelihood_probit(Y,Q_2,b);
[theta_1,fval,exitflag,output,grad,hessian]= fminunc(func,theta_0);
hessian  = hessian /nN;
se_theta = sqrt(diag(hessian));


for i = 1:length(theta_1)
    fprintf('Estimation=%.4f, S.D=%.4f\n',theta_1(i),se_theta(i));
end