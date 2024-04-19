% 读取CSV文件
data = readmatrix('data/1/咽食.csv'); % 假设CSV文件名为'signal.csv'
time_ms = data(:, 1); % 假设第一列是时间，单位是毫秒
signal = data(:, 2); % 假设第二列是信号值

% 将时间转换为秒
time_s = time_ms / 1000;

% 设置带通滤波器的参数
fs = 1000; % 采样频率，假设是1000 Hz
f_lo = 100; % 低频截止频率
f_hi = 300; % 高频截止频率
order = 4; % 滤波器阶数

%filtered_signal = bandpass(signal,[f_lo f_hi],fs);
% filtered_signal = lowpass(signal,200,fs);
% filtered_signal = highpass(filtered_signal,0.01,fs);
filtered_signal = highpass(signal,2,1000);
filtered_signal = bandstop(filtered_signal,[49.5 50.5],1000);

% 设计带通滤波器
%[b, a] = butter(order, [f_lo, f_hi]/(fs/2), 'bandpass');

% 应用带通滤波器
%filtered_signal = filter(b, a, signal);

% 绘制原始信号和滤波后的信号
figure;
subplot(2,1,1);
plot(time_s, signal);
title('原始信号');
xlabel('时间 (s)');
ylabel('信号值');

subplot(2,1,2);
plot(time_s, filtered_signal);
title('带通滤波后的信号');
xlabel('时间 (s)');
ylabel('信号值');
