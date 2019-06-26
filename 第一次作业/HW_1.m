clc;
close all;
%% 第一小题
load data.mat;
figure(1)
hold on;
plot(data,'r');
grid on
xlabel('时间');
ylabel('车流量（辆/小时）');
title('第一小题');
legend('原始数据');
hold off;

%% 第二小题：移动平均法

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
xlabel('时间');
ylabel('车流量（辆/小时）');
title('第二小题：移动平均法');
legend('原始数据','移动平均法:N='+string(N));
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
xlabel('时间');
ylabel('车流量（辆/小时）');
title('第二小题：移动平均法');
legend('原始数据','移动平均法:N='+string(N));
hold off;

%% 合并
figure(7)
hold on;
subplot(3,1,1)
plot(data,'r');
grid on;
xlabel('时间');
ylabel('车流量（辆/小时）');
title('原始数据');
legend('原始数据');

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
xlabel('时间');
ylabel('车流量（辆/小时）');
title('第二小题：移动平均法');
legend('移动平均法:N='+string(N));


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
xlabel('时间');
ylabel('车流量（辆/小时）');
title('第二小题：移动平均法');
legend('移动平均法:N='+string(N));
hold off;


%% 第三问：指数平均法

a = 0.2;
% 设初始值为最初的三个值的平均值
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
xlabel('时间');
ylabel('车流量（辆/小时）');
title('第三小题:指数平滑法');
legend('原始数据','指数平滑法:α='+string(a));
hold off;

a = 0.05;
% 设初始值为最初的三个值的平均值
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
xlabel('时间');
ylabel('车流量（辆/小时）');
title('第三小题:指数平滑法');
legend('原始数据','指数平滑法:α='+string(a));
hold off;
%% 合图
figure(6)
hold on;
a = 0.2;
% 设初始值为最初的三个值的平均值
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
xlabel('时间');
ylabel('车流量（辆/小时）');
title('原始数据');
legend('原始数据');

subplot(3,1,2)

grid on;
plot(Index_Sm,'g')
grid on;
xlabel('时间');
ylabel('车流量（辆/小时）');
title('第三小题:指数平滑法');
legend('指数平滑法:α='+string(a));


a = 0.05;
% 设初始值为最初的三个值的平均值
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
xlabel('时间');
ylabel('车流量（辆/小时）');
title('第三小题:指数平滑法');
legend('指数平滑法:α='+string(a));
hold off;
    