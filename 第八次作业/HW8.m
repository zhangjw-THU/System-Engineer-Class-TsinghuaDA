%% �жϾ������

% ��ԭ��
A = [1 2 7;1/2 1 5;1/7 1/5 1];
ConsistencyTest(A);%һ���Լ��飨��ͬ��
[lamda_A,W_A] = FeatureVector(A);%����ֵ������������⣨��ͬ��

% ��չ������Ŀ
B = [1 1/3 2;3 1 5;1/2 1/5 1];
ConsistencyTest(B);
[lamda_B,W_B] = FeatureVector(B);

% ��������˶��ֱ��������
% �ɼ�������
C1 = [1 1/6 1/3 1/4;6 1 5 5;3 1/5 1 2;4 1/5 1/2 1];
ConsistencyTest(C1);
[lamda_C1,W_C1] = FeatureVector(C1);

% �Ը�������ľ����ʺ����
C2 = [1 2 6 4;1/2 1 6 4;1/6 1/6 1 1/3;1/4 1/4 3 1];
ConsistencyTest(C2);
[lamda_C2,W_C2] = FeatureVector(C2);

% �ҹ������Ѷ�
C3 = [1 2 6 4;1/2 1 6 4;1/6 1/6 1 1/3;1/4 1/4 3 1];
ConsistencyTest(C3);
[lamda_C3,W_C3] = FeatureVector(C3);

% �����õ��Ĵ���
C4 = [1 5 3 5;1/5 1 2 2;1/3 1/2 1 1/2;1/5 1/2 2 1];
ConsistencyTest(C4);
[lamda_C4,W_C4] = FeatureVector(C4);

% ѧλ���������Լ����ڷ�չӰ��
C5 = [1 1/5 1/3 1;5 1 3 5;3 1/3 1 3;1 1/5 1/3 1];
ConsistencyTest(C5);
[lamda_C5,W_C5] = FeatureVector(C5);

%% ����ģ������
% ����
% ����������Ե÷�
 Score = StructuralModel(W_A,W_B,W_C1,W_C2,W_C3,W_C4,W_C5);
 % ��÷����Ĳ���
 [a,b] = max(Score);
 % ��Ӧ����
b
