import numpy as np
import matplotlib.pyplot as plt

def save_depth_visualization(npy_path, save_path, cmap='jet'):
    # 1. 加载数据
    depth_data = np.load(npy_path)
    
    # 2. 处理无效值 (如果有 NaN 或极值，先进行裁剪或填充)
    # 比如：只保留 0.1m 到 10m 之间的有效深度
    # depth_data = np.clip(depth_data, a_min=0.1, a_max=10.0)

    # 3. 创建无边框的可视化图
    fig = plt.figure(frameon=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)

    # 4. 绘制并应用颜色映射
    # vmin/vmax 可以手动指定，确保多张图之间的颜色标尺一致
    ax.imshow(depth_data, cmap=cmap, aspect='auto')

    # 5. 保存为高分辨率图片
    fig.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    print(f"已保存可视化图至: {save_path}")

# 调用示例
i=5
npy_path =f"processed_data_with_depth_npy/Sim_5_UAVs/Samples/UAV{i}/3-40m-1623936157944367872-UAV{i}-depth.npy"
save_path=f"UAV{i}-depth.png"
save_depth_visualization(npy_path, save_path)