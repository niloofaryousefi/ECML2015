load 'X'  % load data
size_X=size(X);
T=size_X(1);
m=10;
MAX_ITER=1000; % max iteration for ADMM

for t=1:T
size_X=size(X{t});
N(t)=size_X(1);
d(t)=size_X(2);
y{t}=X{t}(:,1);
X{t}(:,1)=[];
end

CV=20;
train_per=[0.1,0.2,0.5];

ACCU_train_per=zeros(CV,numel(train_per));

for ctr_tr_per=1:numel(train_per)
  
ACCU_cv=zeros(CV,1);
for cv=1:CV
%%    
[X, y, N_train, N_train_C, N_valid_C, N_test, y_train, y_train_C, y_valid_C, y_test] = Train_Test (X, T, N, y, train_per(1,ctr_tr_per));
N_valid=N_valid_C;

[K_Nor, K_Nor_train, K_Nor_train_C, K_Nor_valid_C, K_Nor_test]= Train_Test_K (X, m, N, N_train, N_train_C);

clearvars -except MAX_ITER T m N  K_t K_Nor X y  y_train y_test N_train N_train_C N_valid N_test K_Nor  K_Nor_train  K_Nor_train_C  K_Nor_valid_C  K_Nor_test N_train_C y_train_C y_valid_C  ACCU_cv  cv  CV  ctr_tr_per train_per  ACCU_train_per  theta
 %%

for t=1:T
    y_train_C{t}=y_train{t}(1:N_train_C(t));
    Y_t{t}=diag(y_train_C{t});  %diagonal matrix of y

    for i=1:m
          theta {t,1}(i,1)=1/m; %initialize \theta: \tehta_{0}=[1/m,1/m,1/m]
    end
end

     log_mu=-10:10;
   for i=1:numel(log_mu)
      mu(i)=2^log_mu(i);
   end

ACCU_k=zeros(1,3);
ACCU_valid_mu=zeros(numel(mu),1);

for ctr_mu=1:numel(mu)

for k=1:10


             
      for t=1:T
          K_theta_t{t}=MKLkernel(theta{t,1}(:,k),K_Nor_train,t);
      end
    
  alpha= Calc_alpha (T, N_train, N_train_C , K_theta_t, y_train) ; 
  
  for t=1:T
      alpha_t{t,k}=alpha{t,1};
  end
     
 
%% 
Q = Calc_Q (alpha_t,Y_t,K_Nor_train_C,m,T,k) ;

theta_ADMM = Calc_theta_OPT (m, T, Q, mu, ctr_mu, MAX_ITER, theta, k ) ;

     for t=1:T
        theta {t,1}(:,k+1)=theta_ADMM{t,1}; %assigning \hat{\theta_{star}} to the k-th element of array \hat{\theta}
     end

%% create model
for t=1:T

N_valid(t)=numel(y_valid_C{t});

K{t}=MKLkernel(theta{t,1}(:,k+1),K_Nor_valid_C,t);
valid=zeros(N_valid(t),N_valid(t)+1);
train_t=zeros(N_train(t),N_train(t)+1);
for i=1:N_train(t)
    train_t(i,1)=i;
end
train_t(1:N_train(t),2:N_train(t)+1)=K_theta_t{t};
train{t}=train_t;
train_C{t}=train_t(1:N_train_C(t),1:N_train_C(t)+1);
y_valid_C{t}=y_train{t}(N_train_C(t)+1:N_train(t));
 
for i=1:N_valid(t)
    valid(i,1)=i;
end

valid(1:N_valid(t),2:N_train_C(t)+1)=K{t};

    %model = svmtrain(y_train, train,['-q -t 4 -c ' num2str(svm_c)]); 
    model = svmtrain(y_train_C{t}, train_C{t},'-q -t 4'); 
    % test on the testset
    [lbl, acc, dec] = svmpredict(y_valid_C{t}, valid, model, []);


ACCU_t(t,k)=acc(1);
end

ACCU_k(1,k)=sum(ACCU_t(:,k))/T;
clearvars -except MAX_ITER T m N X y  N_train N_train_C N_valid  N_test y_train y_train_C  y_valid_C   y_test  K_Nor  K_Nor_train  K_Nor_train_C  K_Nor_valid_C  K_Nor_test  theta  alpha_t Y_t  ACCU_t ACCU_k k   ACCU_valid_mu  ctr_mu  ctr_tr_per train_per  mu  ACCU_cv  cv  CV ACCU_t  ACCU_train_per
end
ACCU_valid_mu(ctr_mu)=max(ACCU_k);
clearvars -except MAX_ITER T m N X y  N_train N_train_C N_valid  N_test y_train y_train_C  y_valid_C   y_test  K_Nor  K_Nor_train  K_Nor_train_C  K_Nor_valid_C  K_Nor_test  theta  alpha_t Y_t   ACCU_t ACCU_k k   ACCU_valid_mu  ctr_mu  ctr_tr_per train_per mu  ACCU_cv ACCU_t cv  CV  ACCU_train_per 
end

[~, ctr_mu_opt] = max(ACCU_valid_mu);
mu_opt=mu(ctr_mu_opt);

%%
clearvars -except MAX_ITER T m N X y N_train N_train_C N_valid  N_test y_train y_train_C  y_valid_C   y_test  K_Nor  K_Nor_train  K_Nor_train_C  K_Nor_valid_C  K_Nor_test   theta  alpha_t  Y_t  ACCU_cv  cv ctr_mu_opt  mu  CV  ctr_tr_per train_per  ACCU_train_per  


 
    for t=1:T
        y_train_C{t}=y_train{t}(1:N_train_C(t));
        Y_t{t}=diag(y_train_C{t});  %diagonal matrix of y
        for i=1:m
             theta {t,1}(i,1)=1/m; %initialize \theta: \tehta_{0}=[1/m,1/m,1/m]
        end
    end


ACCU_k=zeros(1,15);

for k=1:15
             
     for t=1:T
          K_theta_t{t}=MKLkernel(theta{t,1}(:,k),K_Nor_train,t);
     end
    
alpha= Calc_alpha (T, N_train, N_train_C , K_theta_t, y_train) ; 
  
  for t=1:T
      alpha_t{t,k}=alpha{t,1};
  end
     

%% 
Q = Calc_Q (alpha_t,Y_t,K_Nor_train_C,m,T,k) ;

theta_ADMM = Calc_theta_OPT (m, T, Q, mu, ctr_mu_opt, MAX_ITER, theta, k ) ;

     for t=1:T
        theta {t,1}(:,k+1)=theta_ADMM{t,1}; %assigning \hat{\theta_{star}} to the k-th element of array \hat{\theta}
     end
     
%% create model

for t=1:T
lbl=[];
K{t}=MKLkernel(theta{t,1}(:,k),K_Nor_test,t);

K_theta_t{t}=MKLkernel(theta{t,1}(:,k),K_Nor_train,t);

train_t=zeros(N_train(t),N_train(t)+1);
for i=1:N_train(t)
    train_t(i,1)=i;
end
train_t(1:N_train(t),2:N_train(t)+1)=K_theta_t{t};
train{t}=train_t;
train_C{t}=train_t(1:N_train_C(t),1:N_train_C(t)+1);

 
for i=1:N_test(t)
    test{t}(i,1)=i;
end

test{t}(1:N_test(t),2:N_train_C(t)+1)=K{t};

    %model = svmtrain(y_train, train,['-q -t 4 -c ' num2str(svm_c)]); 
    model = svmtrain(y_train_C{t}, train_C{t},'-q -t 4'); 
    % test on the testset
    [lbl, acc, dec] = svmpredict(y_test{t}, test{t}, model, []);
    prd_lbl{k,t}=lbl;



ACCU_t(t,k)=acc(1);
end

ACCU_k(1,k)=sum(ACCU_t(:,k))/T;
clearvars -except MAX_ITER T m N X y N_train N_train_C N_valid  N_test y_train y_train_C  y_valid_C   y_test  K_Nor  K_Nor_train  K_Nor_train_C  K_Nor_valid_C  K_Nor_test   theta  alpha_t Y_t   ACCU_t ACCU_k k ctr_mu_opt  mu  ACCU_cv cv CV ctr_tr_per train_per  ACCU_train_per  theta

end

ACCU_cv(cv)=max(ACCU_k);   
clearvars -except MAX_ITER T N m d X y ACCU_cv cv CV ctr_tr_per train_per  ACCU_train_per theta 

end
ACCU_train_per(:,ctr_tr_per)=ACCU_cv(:,1);
save 'ACCU_train_per.mat'
clearvars -except MAX_ITER T N m d X y CV  ctr_tr_per train_per  ACCU_train_per  theta
end
