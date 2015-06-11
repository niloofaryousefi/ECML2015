function theta_f = Calc_theta_OPT ( m, T, Q, mu, ctr_mu , MAX_ITER, theta, k)

for t=1:T
    theta_admm{t,1}(1:m ,1)= theta{t,1}(1:m ,k);
end

rho=10;
lamda=1/rho;
theta_hat=[];
ABSTOL   = 1e-4;
RELTOL   = 1e-2;
for t=1:T
    for s= t+1 : T
        theta_hat{t,s}(1:m, 1)=theta_admm{t,1}(1:m ,1);
        theta_hat{t,s}(1:m, 2)=theta_admm{s,1}(1:m ,1);
      for j=1:2
        u{t,s}(1:m ,j)=0;
      end
    end
%     theta{t,1}(1:m, 1)=1/m;
    z{t,1}(1:m ,1)=0;
    u_t{t,1}(1:m ,1)=0;
    v{t,1}(1:m ,1)=0;
    y{t,1}(1:m ,1)=0;
    w{t,1}(1:m ,1)=0;
end 

for admm_iter=1:MAX_ITER
        
    %x-update
     for t=1:T 
        for s=t+1:T
         ThetaMinusU{t,s}(1:m ,j)=theta_hat{t,s}(1:m ,j)-u{t,s}(1:m ,j);
        end
     end
     
     [x , x_t]= shrinkage( ThetaMinusU,lamda,T,m);
     
     
     
    %theta update
    theta_old=theta_admm;
    
    for t=1:T
        theta_admm{t,1} = (1/(T-1))*((x_t{t,1}+u_t{t,1}) + (z{t,1}+v{t,1}) - ((lamda/mu(ctr_mu)*Q{t,1})));
    end
   
    
    %theta_hat update (local variables)
    theta_hat_old=theta_hat;
    for t= 1 : T
       for s= t+1 : T
        
            theta_hat{t,s}(1:m, 1)=theta_admm{t,1}(1:m ,1);
            theta_hat{t,s}(1:m, 2)=theta_admm{s,1}(1:m ,1);
         
       end
   end
    
    
    %z update
    zold=z;
     for t=1:T
        z{t,1}=(1/2)* pos((theta_admm{t,1}-v{t,1})+(y{t,1}-w{t,1}));
     end
     
    %y update
    yold=y;
    P=eye(m,m)-((1/m)*(ones(m,1)*ones(1,m)));
    for t=1:T
        y{t,1}=P * (z{t,1}+w{t,1}) + ((1/m)*ones(m,1));
    end
    
    %u update
    for t= 1 : T
       for s= t+1 : T
        
          for j= 1 : 2
            u{t,s}(1:m, j)=u{t,s}(1:m, j) + x{t,s}(1:m, j) - theta_hat{t,s}(1:m, j);
          end
        
       end
    end
   
    for t=1:T
        u_t{t,1}=(1/(T-1))*(((lamda/mu(ctr_mu))*Q{t,1})-(zold{t,1}+v{t,1}));
    end
    
    %v update
    for t=1:T
        v{t,1}= v{t,1}+ z{t,1}- theta_admm{t,1};
    end
    
    %w update
    for t=1:T
        w{t,1}=w{t,1}+z{t,1}-y{t,1};
    end

    % diagnostics, reporting, termination checks
    %history.objval(k)  = objective(A, b, lambda, cum_part, x, z);

    %primal/dual residual
    
 for t=1:T
     r_1_norm{t,1} = norm(x_t{t,1}-theta_admm{t,1});
     r_2_norm{t,1} = norm(z{t,1}-theta_admm{t,1}); 
     r_3_norm{t,1} = norm(z{t,1}-y{t,1});  
     s_1_norm{t,1} = norm((-rho)*(theta_old{t,1}-theta_admm{t,1}));
     s_2_norm{t,1} = norm(-rho * (z{t,1} - zold{t,1} ));
     s_3_norm{t,1} = norm(-rho * (y{t,1} - yold{t,1} ));
 end
 
   r_1_norm_mat = cell2mat(r_1_norm);  
   r_2_norm_mat = cell2mat(r_2_norm); 
   r_3_norm_mat = cell2mat(r_3_norm);
   s_1_norm_mat = cell2mat(s_1_norm);
   s_2_norm_mat = cell2mat(s_2_norm);
   s_3_norm_mat = cell2mat(s_3_norm);
     
   history.r_1_norm(admm_iter)  = sum(r_1_norm_mat);
   history.r_2_norm(admm_iter)  = sum(r_2_norm_mat);
   history.r_3_norm(admm_iter)  = sum(r_3_norm_mat);
   history.s_1_norm(admm_iter)  = sum(s_1_norm_mat);
   history.s_2_norm(admm_iter)  = sum(s_2_norm_mat);
   history.s_3_norm(admm_iter)  = sum(s_3_norm_mat);
    
        
    for t=1:T
        x_norm{t,1} = norm(x_t{t,1});
        theta_norm{t,1}= norm(-theta_admm{t,1});
        z_norm{t,1} = norm(z{t,1});  
        y_norm{t,1} = norm(-y{t,1});
        u_norm{t,1} = norm(u_t{t,1});
        v_norm{t,1} = norm(v{t,1});
        w_norm{t,1} = norm(w{t,1});
    end
  
    x_norm_mat = cell2mat(x_norm); 
    theta_norm_mat = cell2mat(theta_norm);
    z_norm_mat = cell2mat(z_norm); 
    y_norm_mat = cell2mat(y_norm); 
    u_norm_mat = cell2mat(u_norm);
    v_norm_mat = cell2mat(v_norm);
    w_norm_mat = cell2mat(w_norm);
  
    history.eps_pri_1(admm_iter)  = sqrt(T)*ABSTOL + RELTOL*max(sum(x_norm_mat), sum(theta_norm_mat));
    history.eps_pri_2(admm_iter)  = sqrt(T)*ABSTOL + RELTOL*max(sum(z_norm_mat), sum(theta_norm_mat));
    history.eps_pri_3(admm_iter)  = sqrt(T)*ABSTOL + RELTOL*max(sum(z_norm_mat), sum(y_norm_mat));
    history.eps_dual_1(admm_iter) = sqrt(T)*ABSTOL + RELTOL*norm(rho*sum(u_norm_mat));
    history.eps_dual_2(admm_iter) = sqrt(T)*ABSTOL + RELTOL*norm(-rho*(sum(u_norm_mat)+sum(v_norm_mat)));
    history.eps_dual_3(admm_iter) = sqrt(T)*ABSTOL + RELTOL*norm(rho*(sum(v_norm_mat)+sum(w_norm_mat)));
   
     
     if (history.r_1_norm(admm_iter) < history.eps_pri_1(admm_iter)   &&  history.s_1_norm(admm_iter) < history.eps_dual_1(admm_iter)    &&  history.r_2_norm(admm_iter) < history.eps_pri_2(admm_iter)   &&  history.s_2_norm(admm_iter) < history.eps_dual_2(admm_iter)    &&  history.r_3_norm(admm_iter) < history.eps_pri_3(admm_iter)    &&  history.s_3_norm(admm_iter) < history.eps_dual_3(admm_iter) )
     
     break; 
     end
end

theta_f=theta_admm;
end
     
 
   
      
