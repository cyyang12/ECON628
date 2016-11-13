%%

N     = 10000;
alpha =  0;
beta  =  2;
c_0   = -2;
c_1   =  2;
p     = 0.5; %probability z = 1
epsilon  = randn(N,1); 
xi = 1/2 * randn(N,1);  
z  = binornd(1,p,N,1); 

%%59+
C  = exp(-c_0 - c_1*z - xi);
d  = [alpha+c_0 + log(exp(beta)-1) + c_1*z+epsilon+xi > 0];
pi = exp(alpha + beta * d + epsilon);


