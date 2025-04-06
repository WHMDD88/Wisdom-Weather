import pandas as pd
import matplotlib.pyplot as plt
from networkx.algorithms.bipartite import color
from pmdarima.arima import auto_arima
from math import sqrt
from sklearn.metrics import mean_squared_error
import statsmodels.tsa.api as smt
import statsmodels.api as sm
import numpy as np

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def loadData(df,startDate, endDate):
    """
    加载指定日期范围的数据
    :param startDate: 开始日期
    :param endDate: 结束日期
    :return: 日期范围内的数据
    """
    # 将日期列转换为datetime类型（确保列名正确，假设日期列名为'Date'）
    df['Date'] = pd.to_datetime(df['Date'], format="%Y/%m/%d")
    # 转换输入日期为datetime
    start = pd.to_datetime(startDate, format="%Y/%m/%d")
    end = pd.to_datetime(endDate, format="%Y/%m/%d")
    # 筛选数据
    filtered_df = df[(df['Date'] >= start) & (df['Date'] <= end)]
    return filtered_df


def plotData(series,sel_col,y_name):
    """
    对于输入数据进行折线图显示
    :param series:
    :return:
    """
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.figure(figsize=(12, 6))
    plt.style.use('seaborn')  # 使用更美观的绘图风格
    plt.plot(series,color='#3498DB',linewidth=1.5)
    #plt.title('原始数据')
    plt.xlabel('时间', fontsize=14, fontdict={'fontname': 'SimSun'})
    plt.ylabel(y_name, fontsize=14, fontdict={'fontname':'SimSun'})
    plt.grid(True)
    plt.savefig(f'pictures/ARIMA/{sel_col}_原始数据.svg')
    #plt.show()


def plotAllData(df,startDate,endDate,sel_col,y_name):
    """
    对指定日期之后的数据进行展示，初步处理是将数据中的空值替换为0
    :param filterDate:
    :return:
    """
    df1 = loadData(df,startDate, endDate)
    data_df = df1[['Date','Time', sel_col]]
    # 处理缺失值
    # 将 -200 替换为 0
    data_df[sel_col].replace(-200, 0, inplace=True)
    plotData(data_df[sel_col],sel_col,y_name)


def preProcessData(df,startDate, endDate,sel_col):
    """
    对指定日期之后的数据进行预处理，预处理方式将标记的数据使用周围数据的平均值替换
    :param filterDate:
    :return: hp 处理后的数据，其中标记的数据使用平滑值来代替；
    hp_rolling 窗口为17的平滑值，平滑方法是平均值；
    hp_raw 处理后的数据，标记的数据使用0来替换
    """
    hp_all = loadData(df,startDate, endDate)
    # 使用np.where高效生成标记列
    hp_all.loc[:, 'IsNoted'] = np.where(
        hp_all[sel_col] == -200, 1, 0)
    hp = hp_all[sel_col].copy(deep=True)

    hp_all.loc[hp_all['IsNoted'] == 1, sel_col] = np.nan
    hp_raw = hp_all[[sel_col]].fillna(0, inplace=False)
    # 修改滚动窗口大小为 24
    hp_rolling = hp_all[sel_col].rolling(window=24, min_periods=1, center=True).mean()
    hp[hp_all['IsNoted'] == 1] = hp_rolling[hp_all['IsNoted'] == 1]
    return hp, hp_rolling, hp_raw


def plotProcessedData(df,startDate, endDate,sel_col,y_name):
    """
    对原始数据，平滑数据和处理后的数据进行展示
    :param filterDate:
    :return:
    """
    hp, hp_rolling, hp_raw = preProcessData(df,startDate, endDate,sel_col)
    plt.figure(figsize=(12, 6))
    plt.plot(hp_raw, label='Raw Data',color='#0D47A6')
    #plt.plot(hp_rolling, label='Rolling Data')
    # plt.scatter(hp.index, hp, label='Processed Data', c='red', marker='v', s=5)
    plt.plot(hp, label='Processed Data',color='orange')
    #plt.title('数据处理：原始数据，处理后的数据')
    plt.grid(True)
    plt.legend(loc='best')
    plt.xlabel('时间', fontsize=14, fontdict={'fontname': 'SimSun'})
    plt.ylabel(y_name, fontsize=14, fontdict={'fontname':'SimSun'})
    plt.savefig(f'pictures/ARIMA/{sel_col}_处理后的数据.svg')
    #plt.show()


def train(df, train_startDate, train_endDate, test_startDate, test_endDate, sel_col, y_name):
    train_hp, _, _ = preProcessData(df, train_startDate, train_endDate, sel_col)
    test_hp, _, _ = preProcessData(df, test_startDate, test_endDate, sel_col)

    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    model = auto_arima(train_hp, trace=True, error_action='ignore', suppress_warnings=True, seasonal=True, m=24,
                       stationary=False)
    model.fit(train_hp)

    # 打印模型参数的代码
    print("ARIMA 模型基本参数 (p, d, q):", model.order)
    print("季节性 ARIMA 参数 (P, D, Q, m):", model.seasonal_order)

    # 增加置信区间的预测
    forecast, conf_int = model.predict(n_periods=len(test_hp), return_conf_int=True)
    forecast = pd.DataFrame(forecast, index=test_hp.index, columns=['Prediction'])
    conf_int = pd.DataFrame(conf_int, index=test_hp.index, columns=['lower', 'upper'])

    rms = sqrt(mean_squared_error(test_hp, forecast))
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.figure(figsize=(12, 6))
    title = f'评估均方根误差：{rms:.2f}，ARIMA(p, d, q): {model.order}，季节性 ARIMA(P, D, Q, m): {model.seasonal_order}'
    plt.title(title)
    plt.plot(train_hp, label='Train', color='#0D47A6')
    plt.plot(test_hp, label='Test', color='#009688')
    plt.plot(forecast, label='Prediction', color='#D4AF37')
    # 绘制置信区间
    plt.fill_between(conf_int.index, conf_int['lower'], conf_int['upper'], color='k', alpha=0.1)
    plt.xlabel('时间', fontsize=14, fontdict={'fontname': 'SimSun'})
    plt.ylabel(y_name, fontsize=14, fontdict={'fontname': 'SimSun'})
    plt.legend()
    plt.savefig(f'pictures/ARIMA/{sel_col}_预测效果.svg')
    # plt.show()

def tsplot(y, sel_col,lags=None, style='bmh'):
    """
        Plot time series, its ACF and PACF, calculate Dickey–Fuller test
        y - timeseries
        lags - how many lags to include in ACF, PACF calculation
    """
    if not isinstance(y, pd.Series):
        y = pd.Series(y)

    with plt.style.context(style):
        fig = plt.figure(figsize=(12, 8))
        layout = (2, 2)
        ts_ax = plt.subplot2grid(layout, (0, 0), colspan=2)
        acf_ax = plt.subplot2grid(layout, (1, 0))
        pacf_ax = plt.subplot2grid(layout, (1, 1))

        y.plot(ax=ts_ax,color='purple')
        p_value = sm.tsa.stattools.adfuller(y)[1]
        ts_ax.set_title('Dickey-Fuller: p={0:.5f}'.format(p_value))
        smt.graphics.plot_acf(y, lags=lags, ax=acf_ax)
        smt.graphics.plot_pacf(y, lags=lags, ax=pacf_ax)
        plt.tight_layout()
        plt.savefig(f'pictures/ARIMA/{sel_col}_差分.svg')


if __name__ == '__main__':
    train_start_date = "2004/3/11"
    train_end_date = "2004/3/16"
    test_start_date = "2004/3/17"
    test_end_date = "2004/3/17"

    df = pd.read_excel("../data/AirQualityUCI.xlsx", engine='openpyxl', )
    col_list=df.columns.drop(['Date', 'Time'])

    y_labels = ['CO的浓度（微克/立方米）', 'CO浓度的传感器读数', 'NMHC的浓度（微克/立方米）', 'C6H6的浓度（微克/立方米）',
                'NMHC浓度的传感器读数', 'NOx的浓度（微克/立方米）', 'NOx浓度的传感器读数',
                'NO2的浓度（微克/立方米）', 'NO2浓度的传感器读数', 'O3浓度的传感器读数',
                '温度（°C）', '相对湿度（%）','绝对湿度（克/立方米）']
    sel_col=col_list[2]
    y_name=y_labels[2]
    #data_df = loadData(train_start_date, train_end_date)
    plotAllData(df,train_start_date, train_end_date,sel_col,y_name)
    hp, hp_rolling, hp_raw = preProcessData(df,train_start_date, train_end_date,sel_col)
    plotProcessedData(df,train_start_date, train_end_date,sel_col,y_name)

    #  是否进行差分操作可视化
    tsplot_bool = True
    if tsplot_bool:
        hp, _, hp_raw = preProcessData(df,train_start_date, train_end_date,sel_col)
        ads = hp_raw
        # 进行 24 阶差分
        ads_diff = ads[sel_col] - ads[sel_col].shift(24)
        ads_diff = ads_diff - ads_diff.shift(1)
        tsplot(ads_diff[24 + 1:], sel_col,lags=50)
    #plt.show()

    print("test")
    # 模型拟合以及预测
    print("训练列名:", sel_col)
    train(df,train_start_date, train_end_date, test_start_date, test_end_date,sel_col,y_name)