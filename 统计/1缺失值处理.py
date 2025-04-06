import pandas as pd

if __name__ == '__main__':
    data_df=pd.read_excel('../data/AirQualityUCI.xlsx',engine='openpyxl')
    print(data_df.info())
    print(data_df.head(10))
    # 步骤 1：将日期列转换为 datetime 类型
    data_df['Date'] = pd.to_datetime(data_df['Date'])
    # 步骤 2：格式化为目标样式（2004/3/10）
    data_df['Date'] = data_df['Date'].dt.strftime('%Y/%m/%d')
    # 步骤 1：将 -200 替换为缺失值（NaN）
    df = data_df.replace(-200, pd.NA)
    # 步骤 2：删除含有缺失值的行（样本）
    cleaned_df = df.dropna()
    print(cleaned_df.info())
    cleaned_df.to_excel(r'../output_data/without_nan.xlsx',index=False)

    data_df2 = pd.read_excel(r'../output_data/without_nan.xlsx', engine='openpyxl')
    print(data_df2.info())
