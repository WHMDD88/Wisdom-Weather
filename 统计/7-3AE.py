import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import metrics  # 新增：导入评价指标模块
from matplotlib.ticker import PercentFormatter
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

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

# 定义自编码器模型
class Autoencoder(nn.Module):
    def __init__(self, input_dim, encoding_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, encoding_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


def get_encoded_features(data,encoding_dim =5 ):
    """
    :param data: dataframe的values
    :return:
    """
    # 转换为PyTorch张量
    data_tensor = torch.tensor(data, dtype=torch.float32)
    input_dim = data.shape[1]
    autoencoder = Autoencoder(input_dim, encoding_dim)
    criterion = nn.MSELoss()  # 均方误差
    optimizer = optim.Adam(autoencoder.parameters(), lr=0.01)
    num_epochs = 10
    for epoch in range(num_epochs):
        outputs = autoencoder(data_tensor)
        loss = criterion(outputs, data_tensor)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, loss.item()))
    encoder = nn.Sequential(*list(autoencoder.children())[:-1])
    with torch.no_grad():
        encoded_features = encoder(data_tensor).numpy()
    return encoded_features



if __name__ == '__main__':
    data_df = get_data()
    print(data_df.info())
    print(data_df.head(10))
    col_list = data_df.columns.drop(['Date', 'Time'])
    # 归一化
    scaler = MinMaxScaler()
    data_df[col_list] = scaler.fit_transform(data_df[col_list])

    #AE
    X_data = data_df[col_list]
    data = get_encoded_features(X_data.values)
    X_encoder = pd.DataFrame(data)

    # k-means
    plot_elbow_method(X_encoder)
    clusters = perform_kmeans(X_encoder,n_clusters=3)
    plot_2d_scatter(X_encoder, clusters)
    plot_3d_scatter(X_encoder, clusters)
    data_df['cluster_label'] = clusters
    plt.show()
