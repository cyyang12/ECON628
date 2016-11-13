function result = OPG(f1,x,varargin)
% PURPOSE: Computes Hessian based on the outer product of gradients estimator
% -------------------------------------------------------
% Usage:  H = OPG(func,x,varargin)
% Where: func = function name which returns N by 1 vector (N is the number of obs.)
%        fval = func(x,varargin)
%           x = vector of parameters (k by 1)
%    varargin = optional arguments passed to the function
% -------------------------------------------------------
% RETURNS:
%           H = OPG hessian
% -------------------------------------------------------

%eps = 1e-8;

k = size(x,1);
fx = feval(f1,x,varargin{:}); %N by 1 vector
n = size(fx,1);
 
% Compute the stepsize (h)
h = eps.^(1/3)*max(abs(x),1e-8);
xh = x+h; 
h = xh-x;    
ee = sparse(1:k,1:k,h,k,k);
 
% Compute forward step 
G = zeros(n,k); 
for i=1:k
   g = (feval(f1,x+ee(:,i),varargin{:})-feval(f1,x-ee(:,i),varargin{:}))./(2*h(i));
   G(:,i) = g;
end %N by k

H = G'*G; % k by k
result =  H;