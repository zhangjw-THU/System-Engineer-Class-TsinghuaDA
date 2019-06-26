clc;
close all;
%% ��һС��
load data.mat;
figure(1)
hold on;
plot(data,'r');
grid on
xlabel('ʱ��');
ylabel('����������/Сʱ��');
title('��һС��');
legend('ԭʼ����');
hold off;

%% �ڶ�С�⣺�ƶ�ƽ����

N = 30;
L_MA= length(data)-N;
MoveAvg = zeros(L_MA,1);
for ii = 1:L_MA
    MoveAvg(ii) = sum(data(ii:ii+N-1))/N;
end
figure(2);
hold on
plot(data,'r');
grid on;
x_MA = linspace(N/2,L_MA+N/2,L_MA);
plot(x_MA,MoveAvg,'b')
xlabel('ʱ��');
ylabel('����������/Сʱ��');
title('�ڶ�С�⣺�ƶ�ƽ����');
legend('ԭʼ����','�ƶ�ƽ����:N='+string(N));
hold off;

N = 10;
L_MA= length(data)-N;
MoveAvg = zeros(L_MA,1);
for ii = 1:L_MA
    MoveAvg(ii) = sum(data(ii:ii+N-1))/N;
end
figure(3);
hold on
plot(data,'r');
grid on;
x_MA = linspace(N/2,L_MA+N/2,L_MA);
plot(x_MA,MoveAvg,'b')
xlabel('ʱ��');
ylabel('����������/Сʱ��');
title('�ڶ�С�⣺�ƶ�ƽ����');
legend('ԭʼ����','�ƶ�ƽ����:N='+string(N));
hold off;

%% �ϲ�
figure(7)
hold on;
subplot(3,1,1)
plot(data,'r');
grid on;
xlabel('ʱ��');
ylabel('����������/Сʱ��');
title('ԭʼ����');
legend('ԭʼ����');

N = 30;
L_MA= length(data)-N;
MoveAvg = zeros(L_MA,1);
for ii = 1:L_MA
    MoveAvg(ii) = sum(data(ii:ii+N-1))/N;
end
subplot(3,1,2)
grid on;
x_MA = linspace(N/2,L_MA+N/2,L_MA);
plot(x_MA,MoveAvg,'b')
xlabel('ʱ��');
ylabel('����������/Сʱ��');
title('�ڶ�С�⣺�ƶ�ƽ����');
legend('�ƶ�ƽ����:N='+string(N));


N = 10;
L_MA= length(data)-N;
MoveAvg = zeros(L_MA,1);
for ii = 1:L_MA
    MoveAvg(ii) = sum(data(ii:ii+N-1))/N;
end
subplot(3,1,3)
plot(data,'r');
grid on;
x_MA = linspace(N/2,L_MA+N/2,L_MA);
plot(x_MA,MoveAvg,'b')
xlabel('ʱ��');
ylabel('����������/Сʱ��');
title('�ڶ�С�⣺�ƶ�ƽ����');
legend('�ƶ�ƽ����:N='+string(N));
hold off;


%% �����ʣ�ָ��ƽ����

a = 0.2;
% ���ʼֵΪ���������ֵ��ƽ��ֵ
S_last = mean(data(1:3));
L_Index = length(data);
Index_Sm = zeros(L_Index,1);
for ii = 1:L_Index
    Index_Sm(ii) = a*data(ii)+(1-a)*S_last;
    S_last = Index_Sm(ii);
end
figure(4);
hold on
plot(data,'r');
grid on;
plot(Index_Sm,'g')
xlabel('ʱ��');
ylabel('����������/Сʱ��');
title('����С��:ָ��ƽ����');
legend('ԭʼ����','ָ��ƽ����:��='+string(a));
hold off;

a = 0.05;
% ���ʼֵΪ���������ֵ��ƽ��ֵ
S_last = mean(data(1:3));
L_Index = length(data);
Index_Sm = zeros(L_Index,1);
for ii = 1:L_Index
    Index_Sm(ii) = a*data(ii)+(1-a)*S_last;
    S_last = Index_Sm(ii);
end
figure(5);
hold on
plot(data,'r')
plot(Index_Sm,'g');
grid on;
xlabel('ʱ��');
ylabel('����������/Сʱ��');
title('����С��:ָ��ƽ����');
legend('ԭʼ����','ָ��ƽ����:��='+string(a));
hold off;
%% ��ͼ
figure(6)
hold on;
a = 0.2;
% ���ʼֵΪ���������ֵ��ƽ��ֵ
S_last = mean(data(1:3));
L_Index = length(data);
Index_Sm = zeros(L_Index,1);
for ii = 1:L_Index
    Index_Sm(ii) = a*data(ii)+(1-a)*S_last;
    S_last = Index_Sm(ii);
end

subplot(3,1,1)
plot(data,'r');
grid on
xlabel('ʱ��');
ylabel('����������/Сʱ��');
title('ԭʼ����');
legend('ԭʼ����');

subplot(3,1,2)

grid on;
plot(Index_Sm,'g')
grid on;
xlabel('ʱ��');
ylabel('����������/Сʱ��');
title('����С��:ָ��ƽ����');
legend('ָ��ƽ����:��='+string(a));


a = 0.05;
% ���ʼֵΪ���������ֵ��ƽ��ֵ
S_last = mean(data(1:3));
L_Index = length(data);
Index_Sm = zeros(L_Index,1);
for ii = 1:L_Index
    Index_Sm(ii) = a*data(ii)+(1-a)*S_last;
    S_last = Index_Sm(ii);
end

subplot(3,1,3)

plot(Index_Sm,'g');
grid on;
xlabel('ʱ��');
ylabel('����������/Сʱ��');
title('����С��:ָ��ƽ����');
legend('ָ��ƽ����:��='+string(a));
hold off;
    