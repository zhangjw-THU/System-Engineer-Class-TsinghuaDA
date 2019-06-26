function lammax_A = ConsistencyTest(A)
%% 一致性检验
RI=[0 0 0.58 0.90 1.12 1.24 1.32 1.41 1.45 1.49 1.51];
lammax_A = max(eig(A));
CI=(lammax_A-length(A))/(length(A)-1);
CR=CI/RI(length(A));
if CR<0.1
    disp('对比矩阵通过一致性检验:');
    CR
else
    disp('对比矩阵未通过一致性检验，需对对比矩阵重新构造:');
    CR
end
end
    

