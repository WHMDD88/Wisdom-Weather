import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

# 设置字体以支持中文和负号显示
matplotlib.rc("font", family='Microsoft YaHei')

def get_data():
    data_df=pd.read_excel(r'../output_data/without_nan.xlsx',engine='openpyxl')
    # 1. 先将 Date 列转换为 datetime 类型
    data_df['Date'] = pd.to_datetime(data_df['Date'], format='%Y/%m/%d')  # 匹配 '2004/3/10' 格式
    return data_df
def corr(data_df,col_list):
    # 提取数据并计算相关性
    X_data = data_df[col_list]
    corr_matrix = X_data.corr()
    # 绘制热力图
    plt.figure(figsize=(12, 8))  # 设置画布尺寸
    cmap = sns.diverging_palette(220, 20, as_cmap=True)  # 发散型配色，区分正负相关

    # 绘制热力图并添加详细参数
    sns.heatmap(
        corr_matrix,
        cmap=cmap,
        annot=True,  # 显示数值
        fmt=".2f",  # 数值格式
        linewidths=0.5,  # 单元格边框宽度
        center=0,  # 颜色中心值
        annot_kws={"fontsize": 10},  # 数值字体大小
        cbar=True,  # 显示颜色条
        cbar_kws={"shrink": 0.8, "label": "相关系数"}  # 颜色条参数
    )

    # 优化坐标轴标签
    plt.xticks(rotation=35, ha='right')  # X轴标签旋转45度并右对齐
    plt.yticks(rotation=0)  # Y轴标签不旋转
    # plt.title("特征相关性热力图", fontsize=14, pad=20)  # 添加标题
    plt.savefig('pictures/相关性/热力图.svg')
    plt.show()
if __name__ == '__main__':
    data_df=get_data()
    col_list = data_df.columns.drop(['Date', 'Time'])
    corr(data_df,col_list)

