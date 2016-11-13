function val = gamma_val(x)
if (length(x)== 1)
   val = exp(x)/(1 + exp(x));
else
   val = exp(x)./(1 + exp(x));
end
        
