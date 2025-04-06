import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import metrics  # 新增：导入评价指标模块
from matplotlib.ticker import PercentFormatter
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
import numpy as np

# 解决中文显示和负号问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

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
    cleaned_df.to_excel('../output_data/process_nonan.xlsx', index=False)
    df1=pd.read_excel('../output_data/process_nonan.xlsx', engine='openpyxl')
    return df1

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
    ax.view_init(elev=30, azim=45)
    ax.grid(linestyle='--', alpha=0.4)
    cbar = plt.colorbar(scatter, label='Cluster Label', shrink=0.7)
    cbar.ax.tick_params(labelsize=14)
    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)
    ax.tick_params(axis='z', labelsize=14)


def plot_elbow_method(X_data, max_clusters=20):
    distortions = []
    for i in range(1, max_clusters + 1):
        kmeans = KMeans(n_clusters=i, init='random', max_iter=1000, random_state=42, n_init=10)
        kmeans.fit(X_data)
        distortions.append(kmeans.inertia_)

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, max_clusters + 1), distortions, marker='o')
    plt.xlabel('Number of clusters', fontsize=14, fontdict={'fontname': 'Times New Roman'})
    plt.ylabel('Distortion', fontsize=14, fontdict={'fontname': 'Times New Roman'})
    plt.title('The Elbow Method showing the optimal k', fontsize=16, fontdict={'fontname': 'Times New Roman'})
    plt.grid(linestyle='--')
    plt.xticks(range(1, max_clusters + 1),fontsize=12)
    plt.yticks(fontsize=12)

def perform_pca(X_data, n_components=3):
    pca = PCA(n_components=n_components)
    data1 = pca.fit_transform(X_data)
    data = pd.DataFrame(data1)
    return data

def pca_analysis(X_data):
    cov_mat = np.cov(X_data.T)
    eig_vals, eig_vecs = np.linalg.eig(cov_mat)
    eig_pairs = list(zip(eig_vals, eig_vecs))
    eig_pairs.sort(key=lambda x: x[0], reverse=True)
    tot = sum(eig_vals)
    var_exp = [(i / tot) * 100 for i in sorted(eig_vals, reverse=True)]
    cum_var_exp = np.cumsum(var_exp)
    num_components = np.argmax(cum_var_exp >= 90) + 1
    print(f"使方差达到 80% 的特征数量为: {num_components}")
    plt.figure(figsize=(10, 5))
    bar_color = '#FFB6C1'
    plt.bar(range(len(var_exp)), var_exp, alpha=0.7, align="center", label='方差贡献率', color=bar_color)
    step_color = '#FF69B4'
    plt.step(range(len(cum_var_exp)), cum_var_exp, where='mid', label='累计方差贡献率', color=step_color, linewidth=2)
    plt.axhline(y=90, color='r', linestyle='--', label='90% 方差阈值', linewidth=1.5)
    plt.ylabel("方差贡献率百分比",fontdict={'fontname':'SimSun'},fontsize=14)
    plt.xlabel("主成分数量",fontdict={'fontname':'SimSun'},fontsize=14)
    #plt.title("方差贡献率与累计方差贡献率",fontdict={'fontname':'SimSun'})
    # 设置坐标轴刻度为百分比形式
    plt.gca().yaxis.set_major_formatter(PercentFormatter())
    # 设置图例字体大小
    plt.legend(loc="best", fontsize=10)
    # 调整网格线
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()


if __name__ == '__main__':
    data_df = get_data()
    print(data_df.info())
    print(data_df.head(10))
    col_list = data_df.columns.drop(['Date', 'Time'])
    # 归一化
    scaler = MinMaxScaler()
    data_df[col_list] = scaler.fit_transform(data_df[col_list])

    #PCA
    X_data = data_df[col_list]
    data = perform_pca(X_data)
    pca_analysis(X_data)

    # k-means

    plot_elbow_method(data)
    clusters = perform_kmeans(data,n_clusters=3)
    plot_2d_scatter(data, clusters)
    plot_3d_scatter(data, clusters)
    data_df['cluster_label'] = clusters
    plt.show()
