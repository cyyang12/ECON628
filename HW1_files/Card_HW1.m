% Card_HW1.m
% This program estimates the linear probability model of college decision
% using the data set from Card (2005). 
%
% Written by Hiroyuki Kasahara, UBC
%
% Last Updated: Sept 2012
%
% This program calls the following function m-file: g_ols, likelihood,
% likelihood_opg, matrix2latex
 
clear all;

% Go to the directory where the matlab data set is stored
% Note: you need to change the following line 
cd('C:\Users\haoja\Desktop\ECON628\HW1_files\'); 
addpath('C:\Users\haoja\Desktop\ECON628\HW1_files\');
% ``stat'' directory contains StatBox 4.2 written by Gordon K Smyth
% you can download StatBox 4.2 at http://www.statsci.org/matlab/statbox.html

% Load the matlab data set `Card_data.m'
% The original data set is from Card (1995). 
% The sameple selection: (i) non-black, (ii) IQ, motheduc, fatheduc are
% reported
load('Data_HW1'); 

% Try profile
help profile 
profile on 
 
% (0) Generate relevant parameters and variables 
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

% (1) Linear Probability Model --- just OLS

% (1.a) use the OLS matrix formula
b_ols = (X'*X)\(X'*Y);
e = Y-X*b_ols;
% Compute the variance-covariance matrix 
% homoskedasticity case
sigma_hat = (1/nobs)*e'*e;
V_hom = sigma_hat*inv(X'*X);
se_hom = sqrt(diag(V_hom));
% heteroskedasticity case
V_het = (X'*X)\(X'*diag(e.^2)*X)/(X'*X);
se_het = sqrt(diag(V_het)); 

% (1.b) least-squares minimization via fminunc or csminwel
%  Chris Sims' ``csminwel'' is available at http://sims.princeton.edu/yftp/optimize/
b = zeros(nparm,1); 
% using fminunc
options = optimset('Display','iter','TolX',1e-6,'TolFun',1e-6, 'MaxIter', 10000,'MaxFunEvals',10000);

f_ols = @(b)g_ols(b,Y,X); 
[b_ols2,SSR] = fminunc(f_ols,b_ols,options);
% Please learn how to use function handle by typing ``help
% function_handle''  The following two commands are also valid.  
%
% The following also works
% [b_ols2,SSR] = fminunc(@g_ols,b_ols,options,Y,X);  
% [b_ols2,SSR] = fminunc('g_ols',b_ols,options,Y,X);
%
% Try also ``fminsims.m'' which is a wrapper function by Kirill Evdokimov for the 'csminwel' optimization routine 
% written by Chris Sims. The most recent version of Chris Sims' optimization codes 
% can be found at http://sims.princeton.edu/yftp/optimize/. 
% Note: to use ``fminsims'', you need to define a function handle as in
% ``f_like'' because, otherwise, variable arguments cannot be passed
% through ``fminsims''
[b_ols2_sims,SSR] = fminsims(f_ols,b,options);   
% [b_ols2_sims,SSR] = fminsims(@g_ols,b,options,Y,X);   % This DOES NOT work

% profile viewer
% ``fminsims'' seems to be much faster than ``fminunc''
% stop


% (2) Probit 
b0 = [-6.76;0.34;0.04;0.07;0.11;0.3;0.36;0.51;0.67;0.88;0.83;1.18;0.56];
b = zeros(13,1);  
f_like = @(b)likelihood(b,X,Y); % write likelihood.m by yourself
[b_probit,fval] = fminunc(f_like,b,options);  
%[b_probit,fval] = fminunc(@likelihood,b,options,X,Y); % This also works 
% Try also ``fminsims.m''  
[b_probit_sims,fval] = fminsims(f_like,b,options);  
%[b_probit_sims,fval] = fminsims(@likelihood,b,options,X,Y)  % This DOES NOT work
[b_probit, b_probit_sims]
f_opg =  @(b)likelihood_opg(b,X,Y);  % write likelihood_opg.m by yourself
Sigma = OPG(f_opg,b_probit); 
%Sigma = OPG(@likelihood_opg,b_probit,X,Y);  % This also works 
[b_probit,sqrt(diag(inv(Sigma)))] 

 % Latex output 
disp(' '); 
disp(' ');  
disp('Linear Probability Model Estimate of College Decision:');
disp(' ');
columnLabels = {'      ','b_ols  ','b_ols2 ', 'se_hom ', 'se_het '};   
rowLabels = {'const  ','nearc4 ','IQ     ','mothedu','fathedu','reg662 ','reg663 ','reg664 ','reg665 ','reg666 ','reg667 ','reg668 ','reg669 '};  
table =[b_ols,b_ols2,se_hom,se_het];
for COL=1:size(table,2)
    fprintf('%s & ',columnLabels{COL}); 
end
fprintf('%s \\\\ \n', columnLabels{end});
for ROW=1:size(table, 1)  
    fprintf('%s & ',rowLabels{ROW}); 
    for COL=1:size(table,2)-1
        fprintf('%4.3f & ', table(ROW,COL));
    end
    fprintf('%4.3f \\\\ \n', table(ROW,end));
end 
 
% You can also download ``matrix2latex.m'' to print out Latex output from http://www.mathworks.com/matlabcentral/fileexchange/4894-matrix2latex 
columnLabels = {'b_ols  ','b_ols2 ', 'se_hom ', 'se_het '};   % need to redefine columnLabels
matrix2latex(table, 'HW1_out.tex', 'rowLabels', rowLabels, 'columnLabels', columnLabels, 'alignment', 'c', 'format', '%3.1f', 'size', 'small'); 
% output is saved in a latex file named 'HW1_out.tex'
     
profile viewer
    
    
