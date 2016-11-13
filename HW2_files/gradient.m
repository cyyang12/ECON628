function G = gradient(f1,x)

eps = 1e-8;

k = size(x,1);
k
% Compute the stepsize (h)
h = eps.^(1/3)*max(abs(x),1e-8);
xh = x+h; 
h = xh-x;    
ee = sparse(1:k,1:k,h,k,k);
 
% Compute forward step 
G = zeros(k,1); 
for i=1:k
   g = (feval(f1,x+ee(:,i))-feval(f1,x-ee(:,i)))./(2*h(i));
   G(i) = g;
end

