import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

def plot_stability_heatmap(prediction_percent, status_label):
    """
    绘制一个表示稳定性的线性热力图。
    左侧红色(0, 不稳定), 右侧绿色(1, 稳定)。
    """
    
    # ---------- 1. 设置参数和颜色 ----------
    # 将百分比转换为 0-1 之间的小数
    prediction_value = prediction_percent / 100.0
    
    # 定义柔和的颜色 (使用 Hex 代码)
    # 柔和的红色 (例如: Salmon / Light Coral)
    soft_red = '#E57373'  
    # 柔和的绿色 (例如: Light Green / Medium Sea Green tone)
    soft_green = '#81C784' 
    
    # 创建自定义的线性颜色映射 (Linear Segmented Colormap)
    # 只包含两个端点，让 matplotlib 自动完成中间的平滑过渡
    cmap_name = 'soft_red_green'
    colors = [soft_red, soft_green] # 从红到绿的列表
    n_bins = 100 # 过渡的细腻程度
    custom_cmap = mcolors.LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)

    # ---------- 2. 创建绘图骨架 ----------
    # 创建一个宽而矮的画布
    fig, ax = plt.subplots(figsize=(10, 1.5))
    # 调整子图布局，留出底部空间给文字
    plt.subplots_adjust(bottom=0.35)

    # 创建用于显示梯度的背景数据
    # 生成一个从0到1的横向梯度数组 (1行, 500列)
    gradient = np.linspace(0, 1, 500).reshape(1, -1)

    # ---------- 3. 绘制热力图 ----------
    # 使用 imshow 绘制梯度底图
    # extent=[0, 1, 0, 1] 设定了 x 轴和 y 轴的数据范围，方便我们后续按坐标标记
    # aspect='auto' 确保图像填满指定的长宽比
    ax.imshow(gradient, cmap=custom_cmap, extent=[0, 1, 0, 1], aspect='auto')

    # ---------- 4. 标记预测结果 ----------
    # 在预测值的位置画一条明显的垂直线
    # 颜色使用深灰色，避免太突兀，线宽加粗
    ax.axvline(x=prediction_value, color='#333333', linewidth=3, linestyle='-')

    # 在标记线上方添加具体的文字说明
    # 根据位置稍微调整对齐方式，防止文字超出边界 (0.002非常靠左，所以左对齐)
    ha_align = 'left' if prediction_value < 0.5 else 'right'
    text_offset = 0.005 if prediction_value < 0.5 else -0.005
    
    label_text = f"Prediction: {prediction_percent}% ({status_label})"
    
    ax.text(prediction_value + text_offset, 1.05, # 坐标 (x 稍微偏移, y 在图上方)
            label_text,
            color='#333333',
            fontsize=12,
            fontweight='bold',
            ha=ha_align, va='bottom')

    # ---------- 5. 美化坐标轴和标签 ----------
    # 隐藏 Y 轴刻度
    ax.set_yticks([])
    
    # 自定义 X 轴刻度和标签
    # 我们只需要显示两端和中间的刻度
    ax.set_xticks([0.0, 0.5, 1.0])
    # 设置刻度显示的文字，并在两端加上描述
    ax.set_xticklabels(['0\n(Unstable)', '0.5', '1\n(Stable)'], fontsize=10)
    
    # 稍微美化一下边框，只保留底部的轴线
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_color('#888888') # 底部轴线设为灰色
    ax.tick_params(axis='x', colors='#888888') # 刻度文字设为灰色

    # 设置总标题 (可选)
    # ax.set_title("Stability Prediction Heatmap", fontsize=14, pad=25, color='#555555')

    # plt.show()
    plt.savefig("heatmap.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

# =========================================
# 运行代码生成图像
# =========================================
# 输入你的预测参数
prediction_percentage = 0.2  # 0.2%
prediction_status = "Unstable"

plot_stability_heatmap(prediction_percentage, prediction_status)