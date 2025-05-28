import numpy as np
import matplotlib.pyplot as plt

def generate_heatmap(x_shared, y_shared, z_shared, epoch):
    # 从计算图中分离出来，防止梯度计算
    x_shared = x_shared.detach()
    y_shared = y_shared.detach()
    z_shared = z_shared.detach()

    # 将张量转换为 NumPy 数组
    data = np.concatenate([x_shared.cpu().numpy(), y_shared.cpu().numpy(), z_shared.cpu().numpy()], axis=0)

    # 生成热力图
    plt.figure(figsize=(8, 8))  # 设置图像大小

    # 使用 imshow 显示热力图，调整插值和颜色映射
    plt.imshow(data, cmap='viridis', interpolation='nearest', aspect='auto')

    # 添加颜色条
    plt.colorbar()

    # 添加标题
    plt.title(f"Heatmap for Epoch {epoch}")

    # 可选：调整热力图的轴标签，默认会显示
    # plt.xlabel('Feature Index')
    # plt.ylabel('Sample Index')

    # 保存图像
    plt.savefig(f'moseiheatmap_epoch_{epoch}.png')  # 保存图像
    plt.close()

