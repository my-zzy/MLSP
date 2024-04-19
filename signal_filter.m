inputFolderPath = "D:\\清华\\2023春\\MLSP\\code\\data\\1";
outputFolderPath = "D:\\清华\\2023春\\MLSP\\code\\data\\filtered_1";

fileList = dir(fullfile(inputFolderPath, '*.csv'));

for i = 1:length(fileList)
    % 构建完整的文件路径
    inputFile = fullfile(inputFolderPath, fileList(i).name);
    outputFile = fullfile(outputFolderPath, fileList(i).name);
    
    % 读取CSV文件
    data = readmatrix(inputFile);
    
    time_ms = data(:, 1); % 假设第一列是时间，单位是毫秒
    signal = data(:, 2);

    time_s = time_ms / 1000;
    
    
    filtered_signal = highpass(signal,2,1000);
    filtered_signal = bandstop(filtered_signal,[49.5 50.5],1000);

    figure;
    subplot(2,1,1);
    plot(time_ms, signal);
    title(fileList(i).name + "原始信号");
    xlabel('时间 (s)');
    ylabel('信号值');
    
    subplot(2,1,2);
    plot(time_ms, filtered_signal);
    title(fileList(i).name + "滤波后的信号");
    xlabel('时间 (s)');
    ylabel('信号值');


    % 创建一个新的表格来保存滤波后的数据，不包含列名称
    % filteredData = table(time_ms, filtered_signal, VariableNames={'Time', 'FilteredValue'});
    
    % 保存滤波后的数据到新的CSV文件，不包含header
    % writetable(filteredData, outputFile, 'WriteVariableNames', false);

end

