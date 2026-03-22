%% 1. 載入數據
[file, path] = uigetfile('*.mat', '請選擇實驗數據 (.mat)');
if isequal(file, 0), return; end
load(fullfile(path, file));

%% 2. 還原座標軸 (物理單位轉換)
tau_min  = double(tau_min);
tau_max  = double(tau_max);
tau_step = double(tau_step);

theta_min  = double(theta_min);
theta_max  = double(theta_max);
theta_step = double(theta_step);
% 角度軸 (Degree)
theta_axis = theta_min : theta_step : theta_max;

% 時間軸 (從 秒 轉為 納秒 ns，方便閱讀)
tau_axis = (tau_min : tau_step : tau_max) * 1e9; 

%% 3. 
% P_all 的形狀通常是 (Frames, Theta_len, Tau_len)
% 我們取第一幀 (Frame 1) 並轉置以符合 imagesc 的 (Y, X) 慣例
current_frame = 1;
P_data = squeeze(P_all(current_frame, :, :));

figure('Color', 'w', 'Position', [100, 100, 800, 600]);

% 使用 dB 尺度顯示能量分佈，動態範圍會更清楚
imagesc(tau_axis, theta_axis, abs(P_data)); 

%% 4. 設定「每一格都有線」但「特定間隔顯示數字」

% --- X 軸 (ToF) ---
set(gca, 'XTick', tau_axis); 
x_labels = cell(1, length(tau_axis));
for i = 1:length(tau_axis)
    % 使用 round 處理浮點數精度，確保 5, 10, 15 能被抓到
    val = round(tau_axis(i), 2); 
    if mod(val, 5) == 0
        x_labels{i} = num2str(val);
    else
        x_labels{i} = '';
    end
end
set(gca, 'XTickLabel', x_labels);

% --- Y 軸 (AoA) ---
set(gca, 'YTick', theta_axis);
y_labels = cell(1, length(theta_axis));
for i = 1:length(theta_axis)
    val = round(theta_axis(i), 2);
    if mod(val, 15) == 0
        y_labels{i} = num2str(val);
    else
        y_labels{i} = '';
    end
end
set(gca, 'YTickLabel', y_labels);

%% 5. 優化網格外觀
grid on;
set(gca, 'Layer', 'top'); % 讓網格線在最上層
set(gca, 'TickLength', [0 0]);
set(gca, 'GridColor', [1 1 1], 'GridAlpha', 0.4); % 深灰色細線


% 設置色彩映射
colormap('jet');
colorbar;
h = colorbar;
ylabel(h, 'Spectrum Power (dB)');

% 標籤與標題
xlabel('ToF (ns)');
ylabel('AoA (degree)');
%title(['MUSIC Heatmap: ', strrep(file_name, '_', '\_')], 'FontSize', 12);
title(current_frame);

% 優化顯示
grid on;
axis tight;
set(gca, 'YDir', 'normal'); % 確保角度是由下往上遞增