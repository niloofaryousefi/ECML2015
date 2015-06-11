function  [K_1, K_2, K_3, K_4, K_5]= Train_Test_K (X_f, m_f, N_f, N_train_f, N_train_C_f)

T_f=size(X_f ,1);




for t=1:T_f
    
 G{t,1}=X_f{t}*X_f{t}';  
 D{t,1}=(diag(G{t,1})*ones(size(G{t,1},1),1)')+(ones(size(G{t,1},1),1)*diag(G{t,1})')-(2*G{t,1});
 
% for i=1:m_f
%     K_t{t,i}=zeros(N_f(t),N_f(t)); % preallocation for Kernels K_{m} in \sum \theta_{m}K_{m}
%     K_Nor{t,i}=zeros(N_f(t),N_f(t)); %preallocation for Kernels K_{m} in \sum \theta_{m}K_{norm,m}
%     K_Nor_train{t,i}=zeros(N_train_f(t),N_train_f(t));
% end
   
sigma=zeros(m_f-2,1);
for i=1:m_f-2
    sigma(i)=2^(i-1);
end


for k=1:m_f-2
 K_t{t,k}= exp ( (-D{t,1}) / (2*sigma(k)*sigma(k)));  
end


% for k=1:m_f-2   
% for i=1:N_f(t)
% for j=1:N_f(t)
% K_t{t,k}(i,j)=exp((-(norm(X_f{t}(i,:)-X_f{t}(j,:))^2))/(2*sigma(k)*sigma(k))); %Gaussian Kernek function 1 w/ spread parameter \sigma_{1}: K_{Gau_{1}}(x,x{'})=\exp(-\left | x-x{'} \right |/2\sigma_{1}^2)
% 
% end
% end
% end

%  for i=1:N_f(t)
%     for j=1:N_f(t)
% K_t{t,m_f-1}(i,j)=(X_f{t}(i,:)*X_f{t}(j,:)');
% K_t{t,m_f}(i,j)=((X_f{t}(i,:)*X_f{t}(j,:)'))^2;
%     end
%  end
 
 K_t{t,m_f-1}=G{t,1}.^1;
 K_t{t,m_f}=G{t,1}.^2;
 

X_f{t}=[];

for k=1:m_f
for i=1:N_f(t)
    for j=1:N_f(t)
        K_Nor{t,k}(i,j)=K_t{t,k}(i,j)/((K_t{t,k}(i,i)*K_t{t,k}(j,j))^0.5); %Normalized kernels in \sum \theta_{m}K_{norm,m} K_{norm}=\frac{K(i,j)}{K(i,i)K(j,j)}
    end
end
end

% LAMDA{t,1}=diag(1./(sqrt(diag(G{t,1}))));
% 
% for k=1:m_f
% K_Nor_Jadid{t,k}=LAMDA{t,1}*K{t,k}*LAMDA{t,1};
% end

for i=1:m_f
    K_Nor_train{t,i}=K_Nor{t,i}(1:N_train_f(t),1:N_train_f(t));
    K_Nor_train_C{t,i}=K_Nor{t,i}(1:N_train_C_f(t),1:N_train_C_f(t));
    K_Nor_valid_C{t,i}=K_Nor{t,i}(N_train_C_f(t)+1:N_train_f(t),1:N_train_C_f(t));
    K_Nor_test{t,i}=K_Nor{t,i}(N_train_f(t)+1:N_f(t),1:N_train_C_f(t));
    
end

X_f{t}=[];
end
K_1=K_Nor;
K_2=K_Nor_train;
K_3=K_Nor_train_C;
K_4=K_Nor_valid_C;
K_5=K_Nor_test;
   
end
