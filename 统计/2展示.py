import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
# 设置字体以支持中文和负号显示

matplotlib.rc("font", family='Microsoft YaHei')

def loadData(startDate, endDate):
    """
    加载指定日期范围的数据
    :param startDate: 开始日期
    :param endDate: 结束日期
    :return: 日期范围内的数据
    """
    df=pd.read_excel("../data/AirQualityUCI.xlsx",engine='openpyxl',)
    # 将日期列转换为datetime类型（确保列名正确，假设日期列名为'Date'）
    df['Date'] = pd.to_datetime(df['Date'], format="%Y/%m/%d")
    # 转换输入日期为datetime
    start = pd.to_datetime(startDate, format="%Y/%m/%d")
    end = pd.to_datetime(endDate, format="%Y/%m/%d")
    # 筛选数据
    filtered_df = df[(df['Date'] >= start) & (df['Date'] <= end)]
    return filtered_df

def process_data(df):
    # 读取文件
    # 去除缺失值
    # 提取需要预测的列
    col_list = df.columns.drop(['Date', 'Time'])
    # 处理缺失值
    # 将 -200 替换为 NaN
    df[col_list] = df[col_list].replace(-200, np.nan)
    # print(data_df.info())
    # 使用线性插值填充缺失值
    df[col_list] = df[col_list].interpolate(method='linear')
    return df

def get_data(startDate, endDate):
    data_df=loadData(startDate, endDate)
    print(data_df.info())
    df1=process_data(data_df)
    return df1


def draw(data_df):
    plot_columns = data_df.columns.drop(['Date', 'Time'])
    x_data = [i for i in range(0, 72)]
    y_labels = ['CO的浓度（微克/立方米）', 'CO浓度的传感器读数', 'NMHC的浓度（微克/立方米）', 'C6H6的浓度（微克/立方米）',
                'NMHC浓度的传感器读数', 'NOx的浓度（微克/立方米）', 'NOx浓度的传感器读数',
                'NO2的浓度（微克/立方米）', 'NO2浓度的传感器读数', 'O3浓度的传感器读数',
                '温度（°C）', '相对湿度（%）','绝对湿度（克/立方米）']
    color_list = [
        '#FF6B6B',  # 柔和亮红
        '#FF9F43',  # 暖橙色
        '#FFCD5C',  # 明亮黄
        '#4CD964',  # 清新绿
        '#5AC8FA',  # 淡蓝紫调
        '#AB63FA',  # 优雅紫
        '#FF8AC7',  # 温柔粉
        '#00CEC9',  # 青绿色
        '#FFA651',  # 暖橘红
        '#50E3C2',  # 蓝绿色
        '#B45DE0',  # 淡紫
        '#FFD343',  # 暖黄橙
        '#FF7675'  # 珊瑚红
    ]
    for i in range(len(plot_columns)):
        name = plot_columns[i]
        plt.figure(figsize=(12, 6))
        plt.plot(x_data, data_df[plot_columns[i]].values, color=color_list[i], label=plot_columns[i], linewidth=1.5,
                 marker='o')

        # 添加日期分界虚线
        plt.axvline(x=23.5, color='black', linestyle='--', alpha=0.7, linewidth=0.8)
        plt.axvline(x=47.5, color='black', linestyle='--', alpha=0.7, linewidth=0.8)

        # 设置横轴日期标注
        plt.xticks([11, 35, 59], ['3月27日', '3月28日', '3月29日'], fontsize=12)
        plt.yticks(fontsize=12)

        #plt.xlabel('时刻', fontsize=14)
        plt.ylabel(y_labels[i], fontsize=14)
        plt.legend()
        plt.grid(True)
        plt.savefig(f'pictures/展示/2_{name}.svg')
        #plt.show()

if __name__ == '__main__':
    start_date = "2004/3/27"
    end_date = "2004/3/29"
    data_df=get_data(start_date,end_date)
    print(data_df.info())
    draw(data_df)

