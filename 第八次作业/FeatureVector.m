function [lamda,W] = FeatureVector(A)
% ���һ������������
[X,Y] = eig(A);
eigenvalue = diag(Y);     %����ֵ
lamda = max(eigenvalue);  %�����������ֵ
X_lamda = X(:, 1);        %�������ֵ��Ӧ����������
W = X(:,1)/sum(X(:,1));   %��һ����������