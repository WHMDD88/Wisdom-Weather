import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib

# 设置字体以支持中文和负号显示
matplotlib.rc("font", family='Microsoft YaHei')
bar_color = [
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

# 生成颜色更深的列表
line_color = []
for color in bar_color:
    # 将十六进制颜色拆分为 RGB 分量
    r = int(color[1:3], 16)
    g = int(color[3:5], 16)
    b = int(color[5:7], 16)
    # 降低亮度，这里简单地将每个分量乘以 0.7 来加深颜色
    r = max(0, int(r * 0.7))
    g = max(0, int(g * 0.7))
    b = max(0, int(b * 0.7))
    # 转换回十六进制字符串
    dark_color = '#{:02x}{:02x}{:02x}'.format(r, g, b)
    line_color.append(dark_color)

def get_data():
    data_df = pd.read_excel('../data/AirQualityUCI.xlsx', engine='openpyxl')
    data_df['Date'] = pd.to_datetime(data_df['Date'])
    data_df['Date'] = data_df['Date'].dt.strftime('%Y/%m/%d')
    df1 = data_df.replace(-200, pd.NA)
    return df1

def draw(df):
    # 选择要分析的列，这里排除日期和时间列
    plot_columns = df.columns.drop(['Date', 'Time'])
    for i, col in enumerate(plot_columns):
        name=col
        # 绘制直方图和核密度估计图，并设置颜色
        plt.figure(figsize=(10, 6))
        ax = sns.histplot(df[col], bins=28, stat='density', color=bar_color[i])
        # 单独绘制核密度曲线
        sns.kdeplot(df[col], ax=ax, color=line_color[i])
        # 添加标题和坐标轴标签
        #plt.title(f'{name}分布情况', fontsize=16)
        plt.xlabel(col, fontsize=14)
        plt.ylabel('密度', fontsize=14)
        plt.savefig(f'pictures/分布/{name}.svg')
        #plt.show()

if __name__ == '__main__':
    data_df = get_data()
    draw(data_df)
