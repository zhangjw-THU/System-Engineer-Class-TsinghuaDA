function Score = StructuralModel(W_A,W_B,W_C1,W_C2,W_C3,W_C4,W_C5)

W_C1 = W_A(1)*W_C1;
W_C2 = W_A(2)*W_C2;

W_B = W_A(3)*W_B;
W_C3 = W_B(1)*W_C3;
W_C4 = W_B(2)*W_C4;
W_C5 = W_B(3)*W_C5;

W_C = cat(2,W_C1,W_C2,W_C3,W_C4,W_C5);

% �����÷�
Score(1) = sum(W_C(1,:));

% ˶ʿ�÷�
Score(2) = sum(W_C(2,:));

% ��ʿ�÷�
Score(3) = sum(W_C(3,:));

% �����÷�
Score(4) = sum(W_C(4,:));


