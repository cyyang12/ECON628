function p=normcdf(z);
%NORMP	NORMP(Z) is the cumulative normal distribution function at Z.
 

%	GKS 8 Oct 93 
%z = (z-mu)./sqrt(sigma);
p=(1+erf(z/sqrt(2)))./2;
