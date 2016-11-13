function SSR = g_ols(b,Y,X)
% This function m-file computes the sum of squared residuals sum_{i=1}^N
% e(i)^2, where e(i) = Y(i)-X(i,:)*b

e = Y-X*b;
SSR = e'*e;