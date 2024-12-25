import numpy as np
from skimage import color, io
from collections import Counter
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 加载图像
image1 = io.imread('/Users/fiko/MacAir_Documents/TGRS_revision_1/Figure/CIE_LAB/0_418_hr.png')
# image1 = io.imread(r'G:\TGRS\TGRS_revision_1\Figure\Comparative_experiment\0_418_sr.png')  # 替换为实际的图像路径

# 将RGB图像转换为CIELAB色彩空间
lab_image1 = color.rgb2lab(image1)

# 定义颜色区间的数量
num_bins = 7
l_bins = np.linspace(lab_image1[..., 0].min(), lab_image1[..., 0].max(), num_bins)
a_bins = np.linspace(lab_image1[..., 1].min(), lab_image1[..., 1].max(), num_bins)
b_bins = np.linspace(lab_image1[..., 2].min(), lab_image1[..., 2].max(), num_bins)

# 将像素分配到这些区间中
l_indices = np.digitize(lab_image1[..., 0], l_bins) - 1
a_indices = np.digitize(lab_image1[..., 1], a_bins) - 1
b_indices = np.digitize(lab_image1[..., 2], b_bins) - 1

# 确保索引在合法范围内
l_indices = np.clip(l_indices, 0, len(l_bins) - 2)
a_indices = np.clip(a_indices, 0, len(a_bins) - 2)
b_indices = np.clip(b_indices, 0, len(b_bins) - 2)

# 合并为一个唯一的索引
combined_indices = np.stack([l_indices, a_indices, b_indices], axis=-1).reshape(-1, 3)

# 统计每个区间内的像素数量
index_counts = Counter([tuple(x) for x in combined_indices])

# 获取颜色区间中心作为颜色值
l_centers = (l_bins[:-1] + l_bins[1:]) / 2
a_centers = (a_bins[:-1] + a_bins[1:]) / 2
b_centers = (b_bins[:-1] + b_bins[1:]) / 2

# 创建3D散点图
fig = plt.figure(figsize=(14, 8))
ax = fig.add_subplot(111, projection='3d')

# 计算欧式距离
def euclidean_distance(l_val, a_val, b_val):
    return np.sqrt(l_val**2 + a_val**2 + b_val**2)

# 设置颜色和大小
for (l_idx, a_idx, b_idx), count in index_counts.items():
    l_val = l_centers[l_idx]
    a_val = a_centers[a_idx]
    b_val = b_centers[b_idx]

    # 根据CIELAB转换回RGB
    rgb_color = color.lab2rgb([[[l_val, a_val, b_val]]])[0][0]

    # 计算欧式距离
    distance = euclidean_distance(l_val, a_val, b_val)

    # 绘制球体
    ax.scatter(l_val, a_val, b_val, s=distance * 25, color=rgb_color, alpha=0.8)

# 设置坐标轴标签
ax.set_xlabel('L*')
ax.set_ylabel('a*')
ax.set_zlabel('b*')

# # 设置标题
# ax.set_title('CIELAB Color Space Representation')
# plt.axhline(y=ax.get_proj()[2, 3], color='black', linewidth=1)  # 添加横线

# 获取最大和最小欧式距离
distances = [euclidean_distance(l_centers[l_idx], a_centers[a_idx], b_centers[b_idx]) 
             for (l_idx, a_idx, b_idx) in index_counts.keys()]
max_distance = max(distances)
min_distance = min(distances)

# 绘制球体大小比例尺
def plot_size_legend(fig, max_count, min_count):
    # 设置比例球体的大小，使用最大、最小、和中间大小
    sizes = [min_count, (max_count + min_count) / 2, max_count]
    
    # 创建新的子图来绘制比例尺，调整位置使其与主图有一定距离
    # ax_legend = fig.add_axes([0.75, 0.2, 0.15, 0.6], aspect='equal')  # 减小 left 值
    # ax_legend.set_xlim(0, 2)
    # ax_legend.set_ylim(0, 6)

    ax_legend = fig.add_axes([0.73, 0.2, 0.15, 0.6], aspect='equal')
    ax_legend.set_xlim(+0.5, 2.3)  # 调整 xlim 向左偏移
    ax_legend.set_ylim(0, 6)

    # 绘制散点和文本
    for i, size in enumerate(sizes):
        ax_legend.scatter(1, 5 - 2 * i, s=size * 25, color='gray', alpha=0.8)
        ax_legend.text(1.4, 5 - 2 * i, f' {size:.1f}', va='center', fontsize=16)  # 增加字体距离

    # 去除坐标轴和刻度线
    ax_legend.set_xticks([])
    ax_legend.set_yticks([])

    # 添加标题
    ax_legend.set_title('Euclidean Distance', loc='center', pad=10, fontsize=22)

    # 为整个 ax_legend 添加边框
    # 使用 patch 来设置边框，这将包含标题、散点图和文本
    rect = ax_legend.patch
    rect.set_linewidth(1)   # 设置边框线宽
    rect.set_edgecolor('black')  # 设置边框颜色
    rect.set_facecolor('none')   # 使背景透明，保留边框

    # # 移除子图的边框
    # for spine in ax_legend.spines.values():
    #     spine.set_visible(False)

    # # 设置背景为透明
    # ax_legend.set_facecolor('none')

# 调用函数绘制比例尺
plot_size_legend(fig, max_distance, min_distance)

# 保存并显示图像
plt.savefig('CIE_LAB_HR_new.png', dpi=300, bbox_inches='tight')
plt.show()
