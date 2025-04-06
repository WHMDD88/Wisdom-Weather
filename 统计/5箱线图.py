import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib

# 设置字体以支持中文和负号显示
matplotlib.rc("font", family='Microsoft YaHei')


def get_data():
    data_df = pd.read_excel('../data/AirQualityUCI.xlsx', engine='openpyxl')
    return data_df


def draw(df1, selected_col, color, y_label):
    name=selected_col
    df2 = df1[['Date', 'Time', selected_col]]
    df2 = df2[df2[selected_col] != -200]
    df2['Date'] = pd.to_datetime(df2['Date'])
    df2['Year'] = df2['Date'].dt.year
    df2['Month'] = df2['Date'].dt.month

    # 按年份和月份排序
    df2 = df2.sort_values(by=['Year', 'Month'])

    # 组合年份和月份为新的分类变量
    df2['Year_Month'] = df2['Year'].astype(str) + '-' + df2['Month'].astype(str)

    plt.figure(figsize=(14, 6))
    # 绘制箱线图
    sns.boxplot(data=df2, x='Year_Month', y=selected_col, color=color)
    #plt.title(f'{selected_col} 月季变化箱线图', fontsize=16)
    plt.xlabel('', fontsize=12)
    plt.ylabel(y_label, fontsize=12)
    plt.xticks(rotation=25)
    plt.grid()
    plt.savefig(f'pictures/箱线图/{name}.svg')
    #plt.show()

if __name__ == '__main__':
    df1 = get_data()
    col_list = df1.columns.drop(['Date', 'Time'])
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
    y_labels = ['CO的浓度（微克/立方米）', 'CO浓度的传感器读数', 'NMHC的浓度（微克/立方米）', 'C6H6的浓度（微克/立方米）',
                'NMHC浓度的传感器读数', 'NOx的浓度（微克/立方米）', 'NOx浓度的传感器读数',
                'NO2的浓度（微克/立方米）', 'NO2浓度的传感器读数', 'O3浓度的传感器读数',
                '温度（°C）', '相对湿度（%）','绝对湿度（克/立方米）']
    for i in range(len(col_list)):
        draw(df1, col_list[i], color_list[i], y_labels[i])
