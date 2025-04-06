import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import matplotlib
import matplotlib.font_manager as fm

np.random.seed(28)

matplotlib.rc("font", family='Microsoft YaHei')
def calculate_mae(y_true, y_pred):
    """计算平均绝对误差"""
    return np.mean(np.abs(y_true - y_pred))

def process_data(df, startDate, endDate,set_col):
    """加载并预处理数据"""
    # 将日期列转换为datetime类型（确保列名正确，假设日期列名为'Date'）
    df['Date'] = pd.to_datetime(df['Date'], format="%Y/%m/%d")
    # 转换输入日期为datetime
    start = pd.to_datetime(startDate, format="%Y/%m/%d")
    end = pd.to_datetime(endDate, format="%Y/%m/%d")
    # 筛选数据
    filtered_df = df[(df['Date'] >= start) & (df['Date'] <= end)]
    data_df = filtered_df[['Date', 'Time', set_col]]
    # 处理缺失值
    data_df[set_col] = data_df[set_col].replace(-200, np.nan)
    data_df[set_col] = data_df[set_col].interpolate(method='linear')
    return data_df


def create_inout_sequences(input_data, tw, pre_len):
    inout_seq = []
    L = len(input_data)
    for i in range(L - tw):
        train_seq = input_data[i:i + tw]
        if (i + tw + pre_len) > len(input_data):
            break
        train_label = input_data[i + tw:i + tw + pre_len]
        inout_seq.append((train_seq, train_label))
    return inout_seq


def get_confidence_intervals(preds, n_bootstraps=1000, confidence=0.95):
    """计算预测的置信区间"""
    n_preds = len(preds)
    bootstrap_samples = np.zeros((n_bootstraps, n_preds))
    for i in range(n_bootstraps):
        # 展平数组后再采样（关键修改）
        resampled = np.random.choice(preds.flatten(), size=(n_preds,))  # 新增flatten()
        bootstrap_samples[i] = resampled
    ci = np.quantile(bootstrap_samples, [0.025, 0.975], axis=0)
    return ci

class LSTM(nn.Module):
    """LSTM模型定义"""

    def __init__(self, input_dim=1, hidden_dim=350, output_dim=1):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = x.unsqueeze(1)
        h0_lstm = torch.zeros(1, self.hidden_dim).to(x.device)
        c0_lstm = torch.zeros(1, self.hidden_dim).to(x.device)
        out, _ = self.lstm(x, (h0_lstm, c0_lstm))
        out = out[:, -1]
        out = self.fc(out)
        return out

def train(train_inout,lstm_model,name,y_name):
    # 初始化模型
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(lstm_model.parameters(), lr=0.001)
    # 模型训练
    losss = []
    lstm_model.train()
    start_time = time.time()
    for epoch in range(epochs):
        for seq, labels in train_inout:
            optimizer.zero_grad()
            y_pred = lstm_model(seq)
            loss = loss_function(y_pred, labels)
            loss.backward()
            optimizer.step()
            losss.append(loss.item())
            print(f'Epoch: {epoch + 1:3} Loss: {loss.item():10.8f}')
    torch.save(lstm_model.state_dict(), 'model_param/1-2_lstm_model.pth')
    print(f"训练完成，耗时：{(time.time() - start_time) / 60:.2f}分钟")

    plt.style.use('seaborn')  # 使用更美观的绘图风格
    plt.figure(figsize=(12, 5))
    plt.plot(losss, label='Training Loss', color='#3498DB', linewidth=2)  # 更换为深蓝色，增加线宽
    #plt.title('训练损失曲线', fontsize=18, fontdict={'fontname': 'SimHei'})  # 美观的标题
    plt.xlabel('迭代次数', fontsize=14, fontdict={'fontname': 'SimSun'})
    plt.ylabel('MSE Loss', fontsize=14, fontdict={'fontname': 'Times New Roman'})
    plt.grid(linestyle='--', alpha=0.6)  # 优化网格样式
    plt.legend(loc='upper right', fontsize=12)  # 图例位置和字体大小
    plt.tight_layout()  # 自动调整子图参数，让布局更紧凑美观
    plt.savefig(f'pictures/LSTM/{name}_训练损失.svg')
    #plt.show()

def predict(test_data,scaler,name,y_name):
    # 模型预测
    lstm_model = LSTM(input_dim=1, output_dim=pre_len, hidden_dim=train_window)
    lstm_model.load_state_dict(torch.load('model_param/1-2_lstm_model.pth'))
    lstm_model.eval()

    preds = lstm_model(test_data)
    # 反标准化 - 修正预测值
    preds_np = preds.detach().numpy().reshape(-1, 1)  # 使用detach()处理Tensor
    all_preds_denorm = scaler.inverse_transform(preds_np)
    test_np_reshaped = test_np.reshape(-1, 1)
    all_reals_denorm = scaler.inverse_transform(test_np_reshaped)
    mae = calculate_mae(all_reals_denorm, all_preds_denorm)
    rmse = np.sqrt(np.mean((all_reals_denorm - all_preds_denorm) ** 2))
    print(f'MAE={mae:.2f}, RMSE={rmse:.2f}')

    # 计算置信区间
    ci = get_confidence_intervals(all_preds_denorm)
    plt.style.use('seaborn-darkgrid')
    plt.figure(figsize=(12, 5))
    # 绘制训练数据
    plt.plot(range(len(train_np)), train_np, label='train data', color='#0D47A6')
    # 绘制测试数据
    test_start_index = len(train_np)
    test_end_index = test_start_index + len(test_np)
    plt.plot(range(test_start_index, test_end_index), test_np, label='true data', color='#009688')
    # 绘制预测数据
    pred_start_index = test_start_index
    pred_end_index = pred_start_index + len(all_preds_denorm)
    plt.plot(range(pred_start_index, pred_end_index), all_preds_denorm, label='predict data', color='#D4AF37')
    # 绘制置信区间
    plt.fill_between(range(pred_start_index, pred_end_index), ci[0], ci[1], color='gray', alpha=0.2, label='95% CI')
    #plt.title(f'MAE={mae:.2f}, RMSE={rmse:.2f}')
    plt.xlabel('时间',fontdict={'fontname': 'SimSun'},fontsize=14)
    plt.ylabel(f'{y_name}',fontdict={'fontname': 'SimSun'},fontsize=14)
    plt.xticks()
    plt.grid(True)

    plt.legend(loc='upper right')
    plt.legend()
    plt.savefig(f'pictures/LSTM/{name}_预测曲线.svg')
    #plt.show()


if __name__ == '__main__':
    train_start = '2004/3/11'
    train_end = '2004/3/17'
    test_start = '2004/3/17'
    test_end = '2004/3/17'
    y_labels = ['CO的浓度（微克/立方米）', 'CO浓度的传感器读数', 'NMHC的浓度（微克/立方米）', 'C6H6的浓度（微克/立方米）',
                'NMHC浓度的传感器读数', 'NOx的浓度（微克/立方米）', 'NOx浓度的传感器读数',
                'NO2的浓度（微克/立方米）', 'NO2浓度的传感器读数', 'O3浓度的传感器读数',
                '温度（°C）', '相对湿度（%）','绝对湿度（克/立方米）']


    df =pd.read_excel('../data/AirQualityUCI.xlsx',engine='openpyxl',)

    col_list = df.columns.drop(['Date', 'Time'])

    y_name=y_labels[2]
    set_col = col_list[2]

    train_df = process_data(df,train_start,train_end,set_col)
    test_df = process_data(df,test_start,test_end,set_col)

    # 提取目标变量
    col_list1 = df.columns.drop(['Date', 'Time'])

    name=set_col
    train_np = np.array(train_df[set_col])
    test_np = np.array(test_df[set_col])

    # 标准化处理（仅使用训练数据拟合scaler）
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_data_normalized = scaler.fit_transform(train_np.reshape(-1, 1))
    test_data_normalized = scaler.transform(test_np.reshape(-1, 1))

    # 转换为Tensor
    train_data = torch.FloatTensor(train_data_normalized).view(-1)
    test_data = torch.FloatTensor(test_data_normalized).view(-1)

    pre_len = 24
    train_window = 24
    epochs = 60

    # 生成训练序列
    train_inout = create_inout_sequences(train_data, train_window, pre_len)
    lstm_model = LSTM(input_dim=1, output_dim=pre_len, hidden_dim=train_window)

    #训练
    print("训练列名:", set_col)
    train(train_inout,lstm_model,name,y_name)
    #预测
    predict(test_data,scaler,name,y_name)

