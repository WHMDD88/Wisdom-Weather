import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import metrics  # 新增：导入评价指标模块
from matplotlib.ticker import PercentFormatter
from sklearn.preprocessing import MinMaxScaler

def get_data():
    data_df = pd.read_excel('../data/process.xlsx', engine='openpyxl')
    # 步骤 1：将日期列转换为 datetime 类型
    data_df['Date'] = pd.to_datetime(data_df['Date'])
    # 步骤 2：格式化为目标样式（2004/3/10）
    data_df['Date'] = data_df['Date'].dt.strftime('%Y/%m/%d')
    # 步骤 1：将 -200 替换为缺失值（NaN）
    df = data_df.replace(-200, pd.NA)
    # 步骤 2：删除含有缺失值的行（样本）
    cleaned_df = df.dropna()
    # cleaned_df.to_excel('output_data/process_nonan.xlsx', index=False)
    return cleaned_df

def perform_kmeans(X_data, n_clusters=14, max_iter=1000, random_state=42):
    # 显式设置 n_init 参数
    kmeans = KMeans(n_clusters=n_clusters, init='random', max_iter=max_iter, random_state=random_state, n_init=10)
    kmeans.fit(X_data)
    clusters = kmeans.labels_
    return clusters

def plot_2d_scatter(data, clusters):
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(
        data.values[:, 0],
        data.values[:, 1],
        c=clusters,
        cmap='viridis',
        s=80,
        alpha=0.8,
        edgecolor='k'
    )
    plt.xlabel('feature1', fontsize=14, fontdict={'fontname': 'Times New Roman'})
    plt.ylabel('feature2', fontsize=14, fontdict={'fontname': 'Times New Roman'})
    cbar = plt.colorbar(scatter, label='Cluster Label')
    cbar.ax.tick_params(labelsize=12)
    plt.grid(linestyle='--', alpha=0.4)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.savefig('pictures/聚类结果/Kmeans-2D.svg')

def plot_3d_scatter(data, clusters):
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(
        data.values[:, 0],
        data.values[:, 1],
        data.values[:, 2],
        c=clusters,
        cmap='viridis',
        s=100,
        alpha=0.8,
        edgecolor='k'
    )
    ax.set_xlabel('feature1', fontsize=16, fontdict={'fontname': 'Times New Roman'})
    ax.set_ylabel('feature2', fontsize=16, fontdict={'fontname': 'Times New Roman'})
    ax.set_zlabel('feature3', fontsize=16, fontdict={'fontname': 'Times New Roman'})
    ax.view_init(elev=65, azim=45)
    ax.grid(linestyle='--', alpha=0.4)
    cbar = plt.colorbar(scatter, label='Cluster Label', shrink=0.7)
    cbar.ax.tick_params(labelsize=14)
    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)
    ax.tick_params(axis='z', labelsize=14)
    plt.savefig('pictures/聚类结果/Kmeans-3D.svg')

def plot_elbow_method(X_data, max_clusters=20):
    distortions = []
    for i in range(1, max_clusters + 1):
        kmeans = KMeans(n_clusters=i, init='random', max_iter=1000, random_state=42, n_init=10)
        kmeans.fit(X_data)
        distortions.append(kmeans.inertia_)

    plt.figure(figsize=(10, 5))
    plt.plot(range(1, max_clusters + 1), distortions, marker='o')
    plt.xlabel('Number of clusters', fontsize=14, fontdict={'fontname': 'Times New Roman'})
    plt.ylabel('Distortion', fontsize=14, fontdict={'fontname': 'Times New Roman'})
    plt.title('The Elbow Method showing the optimal k', fontsize=16, fontdict={'fontname': 'Times New Roman'})
    plt.grid(linestyle='--')
    plt.xticks(range(1, max_clusters + 1),fontsize=12)
    plt.yticks(fontsize=12)
    plt.savefig('pictures/聚类结果/肘部法.svg')
    plt.show()

if __name__ == '__main__':
    data_df = get_data()
    print(data_df.info())
    print(data_df.head(10))
    col_list = data_df.columns.drop(['Date', 'Time'])
    # 归一化
    scaler = MinMaxScaler()
    data_df[col_list] = scaler.fit_transform(data_df[col_list])

    X_data = data_df[col_list]
    # 肘部法
    plot_elbow_method(X_data)
    # k-means
    clusters = perform_kmeans(X_data)
    #plot_2d_scatter(X_data, clusters)
    #plot_3d_scatter(X_data, clusters)
    data_df['cluster_label'] = clusters
