function lammax_A = ConsistencyTest(A)
%% һ���Լ���
RI=[0 0 0.58 0.90 1.12 1.24 1.32 1.41 1.45 1.49 1.51];
lammax_A = max(eig(A));
CI=(lammax_A-length(A))/(length(A)-1);
CR=CI/RI(length(A));
if CR<0.1
    disp('�ԱȾ���ͨ��һ���Լ���:');
    CR
else
    disp('�ԱȾ���δͨ��һ���Լ��飬��ԶԱȾ������¹���:');
    CR
end
end
    

