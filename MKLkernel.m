function sum = MKLkernel(theta,K,t)
%this function produces the final Kernel matrix K=sum(theta(m)*K(m)) in MKL
%method

sum=0;
m=length(theta);
for i=1:m
sum=sum+(theta(i,1)*K{t,i});
end
end
