function alpha_f = Calc_alpha (T_f, N_train_f, N_train_C_f , K_theta_t_f, y_train_f) 

for t=1:T_f
    
train_t=zeros(N_train_f(t),N_train_f(t)+1);
for i=1:N_train_f(t)
    train_t(i,1)=i;
end
train_t(1:N_train_f(t),2:N_train_f(t)+1)=K_theta_t_f{t};
train{t}=train_t;
train_C{t}=train_t(1:N_train_C_f(t),1:N_train_C_f(t)+1);
valid_C{t}=train_t(N_train_C_f(t)+1:N_train_f(t),1:N_train_C_f(t)+1);
size_valid_C{t}=size(valid_C{t});
N_valid(t)=size_valid_C{t}(1);

 for i=1:size_valid_C{t}(1)
     valid_C{t}(i,1)=i;
 end
y_train_C{t}=y_train_f{t}(1:N_train_C_f(t));
y_valid_C{t}=y_train_f{t}(N_train_C_f(t)+1:N_train_f(t));
n = -17:17;
    accuracy{t} = nan(size(n));
    for i=1:numel(n);   % n = {-17,...,17}
        svm_c=2^n(i); 
        % create model
        %options=['-t 4 -c ' num2str(svm_c)];
        model = svmtrain(y_train_C{t}, train_C{t},['-q -t 4 -c ' num2str(svm_c)]);
        
        % option: -t 4 -> precomputed kernel
          [lbl, acc, dec] = svmpredict(y_valid_C{t}, valid_C{t}, model);
        accuracy{t}(i) = acc(1);
    end
    % output the accuracy vs the chosen parameter c
    plot(accuracy{t});
    xlabel('svm_c'), ylabel('Accuracy'); title('Accuracy vs. svm_c');

 %% test optimal c on the test set
    [~, i] = max(accuracy{t}); % find the best value
    svm_c = 2^n(i);             % this is the optimal c

 % create model
model_alpha_star = svmtrain(y_train_C{t}, train_C{t},['-t 4 -c ' num2str(svm_c)]);
nSV=model_alpha_star.nSV;
sv_coef=model_alpha_star.sv_coef;
sv_indices=model_alpha_star.sv_indices;
totalSV=model_alpha_star.totalSV;
for i=(nSV(1)+1):nSV(1)+nSV(2)
  sv_coef(i)=-1*sv_coef(i);
end
alpha_star=zeros(N_train_C_f(t),1);

for i=1:totalSV
    alpha_star(sv_indices(i))=sv_coef(i);
end

alpha_t_f{t,1}=alpha_star; %assigning \alpha^{star} to the k-th element of array \alpha 

end
alpha_f = alpha_t_f;
end

