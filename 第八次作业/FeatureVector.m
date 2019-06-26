function [lamda,W] = FeatureVector(A)
% 求归一化的特征向量
[X,Y] = eig(A);
eigenvalue = diag(Y);     %特征值
lamda = max(eigenvalue);  %矩阵最大特征值
X_lamda = X(:, 1);        %最大特征值对应的特征向量
W = X(:,1)/sum(X(:,1));   %归一化特征向量