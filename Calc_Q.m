function Q_f = Calc_Q (alpha,Y,K,m_f,T_f,l)

for t=1:T_f
Q_prime{t,1}=zeros(m_f,1); %preallocating memory for vector T which is a function of \alpha; 
for i=1:m_f
    Q_prime{t,1}(i,1)=(-1/2)*alpha{t,l}'*Y{t}*K{t,i}*Y{t}*alpha{t,l}; %  this vector is defined as: \sum(-\frac{1}{2}\alpha_{k}^{'}Y K_{m} Y \alpha_{k})
end
end
Q_f=Q_prime ;
end
