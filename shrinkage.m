function [y ,y_t] = shrinkage(a, kappa, S, n)
m=n;
for t= 1 : S
    for s= t+1 : S
        
        for j= 1 : 2
        prox{t,s}(1:m ,j) = a{t,s}(1:m ,j) - (((a{t,s}(1:m ,1)-a{t,s}(1:m ,2)) / norm(a{t,s}(1:m ,1)-a{t,s}(1:m ,2))) * min(kappa, norm(a{t,s}(1:m ,1)-a{t,s}(1:m ,2)) /2 ));
        end
        
    end
end
     for t=1:S
        x_t{t,1}(1:m ,1)=0; 
      
     end

for t=1:S
    l=t-1;
    for s=t+1:S
    x_t{t,1}(1:m, 1) = x_t{t,1}(1:m, 1)+ prox{t,s}(1:m, 1);
    end
    while l > 0
        x_t{t,1}(1:m ,1)=x_t{t,1}(1:m ,1)+prox{l,t}(1:m ,2);
        l=l-1;
    end
    
    x_new{t,1}=(1/(S-1))*x_t{t,1};
end
