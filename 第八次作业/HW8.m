%% 判断矩阵假设

% 三原则
A = [1 2 7;1/2 1 5;1/7 1/5 1];
ConsistencyTest(A);%一致性检验（下同）
[lamda_A,W_A] = FeatureVector(A);%特征值与特征向量求解（下同）

% 发展的三项目
B = [1 1/3 2;3 1 5;1/2 1/5 1];
ConsistencyTest(B);
[lamda_B,W_B] = FeatureVector(B);

% 出国，读硕，直博，工作
% 成绩和能力
C1 = [1 1/6 1/3 1/4;6 1 5 5;3 1/5 1 2;4 1/5 1/2 1];
ConsistencyTest(C1);
[lamda_C1,W_C1] = FeatureVector(C1);

% 性格和以往的经验适合与否
C2 = [1 2 6 4;1/2 1 6 4;1/6 1/6 1 1/3;1/4 1/4 3 1];
ConsistencyTest(C2);
[lamda_C2,W_C2] = FeatureVector(C2);

% 找工作的难度
C3 = [1 2 6 4;1/2 1 6 4;1/6 1/6 1 1/3;1/4 1/4 3 1];
ConsistencyTest(C3);
[lamda_C3,W_C3] = FeatureVector(C3);

% 工作得到的待遇
C4 = [1 5 3 5;1/5 1 2 2;1/3 1/2 1 1/2;1/5 1/2 2 1];
ConsistencyTest(C4);
[lamda_C4,W_C4] = FeatureVector(C4);

% 学位和履历对自己长期发展影响
C5 = [1 1/5 1/3 1;5 1 3 5;3 1/3 1 3;1 1/5 1/3 1];
ConsistencyTest(C5);
[lamda_C5,W_C5] = FeatureVector(C5);

%% 根据模型评价
% 方案
% 计算各个策略得分
 Score = StructuralModel(W_A,W_B,W_C1,W_C2,W_C3,W_C4,W_C5);
 % 求得分最大的策略
 [a,b] = max(Score);
 % 对应策略
b
