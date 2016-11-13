function [A,B,r,U,V,stats] = canoncorr(X,Y)
%CANONCORR Canonical correlation analysis.
%   [A,B] = CANONCORR(X,Y) computes the sample canonical coefficients for
%   the N-by-D1 and N-by-D2 data matrices X and Y.  X and Y must have the
%   same number of observations (rows) but can have different numbers of
%   variables (cols).  A and B are D1-by-D and D2-by-D matrices, where D =
%   min(rank(X),rank(Y)).  The jth columns of A and B contain the canonical
%   coefficients, i.e. the linear combination of variables making up the
%   jth canonical variable for X and Y, respectively.  Columns of A and B
%   are scaled to make COV(U) and COV(V) (see below) the identity matrix.
%   If X or Y are less than full rank, CANONCORR gives a warning and
%   returns zeros in the rows of A or B corresponding to dependent columns
%   of X or Y.
%
%   [A,B,R] = CANONCORR(X,Y) returns the 1-by-D vector R containing the
%   sample canonical correlations.  The jth element of R is the correlation
%   between the jth columns of U and V (see below).
%
%   [A,B,R,U,V] = CANONCORR(X,Y) returns the canonical variables, also
%   known as scores, in the N-by-D matrices U and V.  U and V are computed
%   as
%
%      U = (X - repmat(mean(X),N,1))*A and
%      V = (Y - repmat(mean(Y),N,1))*B.
%
%   [A,B,R,U,V,STATS] = CANONCORR(X,Y) returns a structure containing
%   information relating to the sequence of D null hypotheses H0_K, that
%   the (K+1)st through Dth correlations are all zero, for K = 0:(D-1).
%   STATS contains three fields, each a 1-by-D vector with elements
%   corresponding to values of K:
%
%      DFE:   the error degrees of freedom == (D1-K)*(D2-K)
%      CHISQ: Bartlett's approximate chi-squared statistic for H0_K
%      P:     the right-tail significance level for H0_K
%
%   Example:
%
%      load carbig;
%      X = [Displacement Horsepower Weight Acceleration MPG];
%      nans = sum(isnan(X),2) > 0;
%      [A B r U V] = canoncorr(X(~nans,1:3), X(~nans,4:5));
%
%      plot(U(:,1),V(:,1),'.');
%      xlabel('0.0025*Disp + 0.020*HP - 0.000025*Wgt');
%      ylabel('-0.17*Accel + -0.092*MPG')
%
%   See also PRINCOMP, MANOVA1.

%   References:
%     [1] Krzanowski, W.J., Principles of Multivariate Analysis,
%         Oxford University Press, Oxford, 1988.
%     [2] Seber, G.A.F., Multivariate Observations, Wiley, New York, 1984.

%   Copyright 1993-2002 The MathWorks, Inc.
%   $Revision: 1.2 $  $Date: 2002/03/21 22:06:30 $

if nargin < 2
    error('Requires two arguments.');
end

[n,d1] = size(X);
if size(Y,1) ~= n
    error('X and Y must have the same number of rows.');
elseif n == 1
    error('X and Y must have more than one row.');
end
d2 = size(Y,2);

% center the variables
X = X - repmat(mean(X,1), n, 1);
Y = Y - repmat(mean(Y,1), n, 1);

% factor the inputs, and find a full rank set of columns if necessary
[Q1,T11,e1] = qr(X,0);
k1 = sum(abs(diag(T11)) > max(abs(diag(T11))*eps^(3/4)));
if k1 == 0
    error('X must contain at least one non-constant column');
elseif k1 < d1
    warning('X is not full rank.');
    Q1 = Q1(:,1:k1); T11 = T11(1:k1,1:k1);
end
[Q2,T22,e2] = qr(Y,0);
k2 = sum(abs(diag(T22)) > max(abs(diag(T22))*eps^(3/4)));
if k2 == 0
    error('Y must contain at least one non-constant column');
elseif k2 < d2
    warning('Y is not full rank.');
    Q2 = Q2(:,1:k2); T22 = T22(1:k2,1:k2);
end

% canonical coefficients and canonical correlations.  for k1 > k2, the
% economy-size version ignores the extra columns in L and rows in D.  for
% k1 < k2, need ignore extra columns in M and D explicitly.  normalize
% A and B to give U and V unit variance
d = min(k1,k2);
[L,D,M] = svd(Q1' * Q2,0);
A = T11 \ L(:,1:d) * sqrt(n-1);
B = T22 \ M(:,1:d) * sqrt(n-1);
r = min(max(diag(D(:,1:d))', 0), 1); % remove dangerous roundoff errs

% put back to full size and correct order
A = [A; zeros(d1-k1,d)];
B = [B; zeros(d2-k2,d)];
einv1(e1) = 1:d1; A = A(einv1,:);
einv2(e2) = 1:d2; B = B(einv2,:);

% canonical variates
if nargout > 3
    U = X * A;
    V = Y * B;
end

% chi-squared statistic for H0k: rho_(k+1) == ... = rho_d == 0
if nargout > 5
    k = 0:(d-1);
    LL = repmat(-Inf, 1, d);
    wh = find(r < 1);
    LL(wh) = fliplr(cumsum(fliplr(log(1-r(wh).^2))));
    stats.dfe = (d1-k) .* (d2-k);
    stats.chisq = -(n - .5*(d1+d2+3)) .* LL;
    stats.p = 1 - chi2cdf(stats.chisq, stats.dfe);
end