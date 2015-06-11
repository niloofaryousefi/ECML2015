function [X_f_2, y_f_2, N_train_f, N_train_C_f, N_valid_C_f, N_test_f, y_train_f, y_train_C_f, y_valid_C_f, y_test_f] = Train_Test (X_f_1, T_f, N_f, y_f_1, train_per_f)


for t=1:T_f
fold(t)=round((train_per_f)*N_f(t));
Index=randperm(N_f(t));
X_new{t}=X_f_1{t}(Index(1:N_f(t)),:);
y_new{t}=y_f_1{t}(Index(1:N_f(t)),:);


N_train_C(t)=fold(t);
N_valid_C(t)=round((N_f(t)-fold(t))/2);
N_train(t)=N_train_C(t)+N_valid_C(t);
N_test(t)=N_f(t)-N_train(t);

y_train{t}=y_new{t}(1:N_train(t));
y_train_C{t}=y_train{t}(1:N_train_C(t));
y_valid_C{t}=y_train{t}(N_train_C(t)+1:N_train(t));
y_test{t}=y_new{t}(N_train(t)+1:N_f(t));
end
X_f_2=X_new';
y_f_2=y_new';
N_train_f = N_train ;
N_train_C_f = N_train_C ;
N_valid_C_f = N_valid_C ;
N_test_f = N_test ;
y_train_f = y_train ;
y_train_C_f = y_train_C ;
y_valid_C_f=y_valid_C;
y_test_f = y_test ;

end
