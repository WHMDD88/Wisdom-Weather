import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from matplotlib.colors import to_rgba
import matplotlib

matplotlib.rc("font", family='Microsoft YaHei')

def draw(set_col1, set_col2, base_color):
    X, y = get_train_data(set_col1, set_col2)
    # 创建并训练线性回归模型
    lin_reg = LinearRegression()
    lin_reg.fit(X, y)

    X_new = np.array([[X.min()], [X.max()]])
    y_predict = lin_reg.predict(X_new)

    scatter_color = to_rgba(base_color, alpha=0.3)
    line_color = base_color

    plt.figure(figsize=(6, 6))
    plt.scatter(X, y, color=scatter_color, s=50, label='数据点')
    plt.plot(X_new, y_predict, color=line_color, linewidth=5, label='回归直线')
    plt.xlabel(set_col1, fontsize=12, fontdict={'fontname': 'Times New Roman'})
    plt.ylabel(set_col2, fontsize=12, fontdict={'fontname': 'Times New Roman'})
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(fontsize=10, loc='upper left')

    # 获取斜率和截距
    a = lin_reg.coef_[0][0]
    b = lin_reg.intercept_[0]
    # 标注斜率和截距
    plt.text(0.68, 0.12, f'斜率 a: {a:.2f}', transform=plt.gca().transAxes, fontsize=10)
    plt.text(0.68, 0.07, f'截距 b: {b:.2f}', transform=plt.gca().transAxes, fontsize=10)

    plt.tight_layout()
    plt.savefig(f'pictures\\{set_col1}-{set_col2}.svg')
    #plt.show()


def get_train_data(set_col1, set_col2):
    df1 = pd.read_excel('../output_data/without_nan.xlsx', engine='openpyxl')
    # 将一维数组转换为二维数组
    X = df1[set_col1].values.reshape(-1, 1)
    y = df1[set_col2].values.reshape(-1, 1)
    return X, y


if __name__ == '__main__':
    col_list1 = ['CO(GT)', 'PT08.S1(CO)', 'NMHC(GT)', 'C6H6(GT)', 'PT08.S2(NMHC)',
                 'NOx(GT)', 'PT08.S3(NOx)', 'NO2(GT)', 'PT08.S4(NO2)', 'PT08.S5(O3)',
                 'T', 'RH', 'AH']

    draw('CO(GT)', 'PT08.S1(CO)', '#0066cc')

    draw('NMHC(GT)', 'PT08.S2(NMHC)', '#00cc00')

    draw('NOx(GT)', 'PT08.S3(NOx)', '#cc0000')

    draw('NO2(GT)', 'PT08.S4(NO2)', '#6600cc')

    draw('CO(GT)', 'NMHC(GT)', '#FF7F50')

    draw('CO(GT)', 'C6H6(GT)', '#40E0D0')

    draw('CO(GT)', 'NO2(GT)', '#FF69B4')

    draw('CO(GT)', 'NOx(GT)', '#FFBF00')

    draw('NMHC(GT)', 'C6H6(GT)', '#4169E1')

    draw('NMHC(GT)', 'NOx(GT)', '#708090')

    draw('NMHC(GT)', 'NO2(GT)', '#D2691E')

    draw('C6H6(GT)', 'NOx(GT)', '#191970')

    draw('C6H6(GT)', 'NO2(GT)', '#B22222')

    draw('NOx(GT)', 'NO2(GT)', '#F4A460')
