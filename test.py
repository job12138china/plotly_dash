import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

# ==========================================
# 1. 视觉资产重构 (Refined Visual Assets)
# ==========================================
# 保持你的复古配色，但重新分配逻辑
COLOR_BG = '#FAEDDA'  # 你的米色背景
COLOR_MAIN_LINE = '#3C2F80'  # 深靛蓝 - 用于主线条
COLOR_ACCENT = '#26B6C6'  # 青色 - 用于强调点


# 辅助函数：将Hex颜色转为RGBA，用于精准控制透明度
def adjust_alpha(hex_color, alpha):
    rgb = mcolors.hex2color(hex_color)
    return (*rgb, alpha)


def setup_fonts():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "mathtext.fontset": "stix",
        "axes.unicode_minus": False,  # 解决负号显示问题
    })


# ==========================================
# 2. 迭代后的绘图引擎 (The Refined Engine)
# ==========================================
def plot_refined_elegant_line(x, y, title, xlabel, ylabel, output_filename=None):
    setup_fonts()

    # A. 画布：更加轻盈
    # 使用较高的透明度，让背景色不那么像“一块实心砖头”
    fig, ax = plt.subplots(figsize=(12, 5), facecolor=COLOR_BG)
    ax.set_facecolor(COLOR_BG)

    # B. 绘制数据：同源色彩，极细线条
    # 1. 线条：从2.5降到1.2。这才是精确。
    ax.plot(x, y, color=COLOR_MAIN_LINE, linewidth=1.2, zorder=3)

    # 2. 填充：关键修改。使用和线条完全一样的颜色，但Alpha设为0.15
    # 这样看起来像是线条投下的淡淡阴影，而不是两张皮。
    ax.fill_between(x, y, color=COLOR_MAIN_LINE, alpha=0.15, zorder=2)

    # C. 坐标轴与网格：退居幕后
    # 1. 隐藏上、右边框
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # 2. 左、下边框：变细，不要加粗。颜色稍微淡一点点，不要纯黑。
    ax.spines['left'].set_linewidth(0.8)
    ax.spines['bottom'].set_linewidth(0.8)
    ax.spines['left'].set_color('#505050')
    ax.spines['bottom'].set_color('#505050')

    # 3. 网格：参考图中是垂直虚线。我们模仿它。
    # 只要x轴的网格，不要y轴的，这样更干净，强调时间/点的推进。
    ax.grid(axis='x', color='gray', linestyle='--', linewidth=0.8, alpha=0.3, zorder=1)
    ax.grid(axis='y', visible=False)  # 关闭横向网格，减少噪音

    # D. 字体排版：层级分明
    # 标题：大，加粗，位于左侧
    ax.set_title(title, fontsize=18, fontweight='bold', color=COLOR_MAIN_LINE, pad=15, loc='left')

    # 轴标签：加粗，字号适中
    ax.set_xlabel(xlabel, fontsize=14, fontweight='bold', color=COLOR_MAIN_LINE)
    ax.set_ylabel(ylabel, fontsize=14, fontweight='bold', color=COLOR_MAIN_LINE)

    # 刻度标签：**不要加粗**，让它们显得纤细
    ax.tick_params(axis='both', which='major', labelsize=12, color='#505050')
    # 确保刻度文字颜色和轴线一致
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_color('#333333')
        label.set_fontweight('normal')  # 强制不加粗

    # E. 关键点标记 (仿照参考图的圆点)
    # 找出所有的波峰
    from scipy.signal import find_peaks
    peaks, _ = find_peaks(y, distance=10)  # distance防止点太密

    # 只标出显著的峰值（例如大于某个阈值）
    significant_peaks = peaks[y[peaks] > 0.5]

    # 绘制红点（这里用你的配色里的Accent色）
    ax.scatter(x[significant_peaks], y[significant_peaks],
               color=COLOR_ACCENT, s=40, zorder=4,
               edgecolor='white', linewidth=1)  # 加个白边，更精致

    # F. 布局与导出
    plt.tight_layout()
    if output_filename:
        # 导出时 DPI 设为 300，背景色保持一致
        plt.savefig(f"{output_filename}.png", dpi=300, facecolor=COLOR_BG)

    plt.show()


# ==========================================
# 3. 运行测试 (Simulation)
# ==========================================
# 模拟更像动量图的数据
x_sim = np.linspace(0, 300, 300)
# 生成一些随机波动，模拟比赛走势
np.random.seed(42)
noise = np.random.normal(0, 0.1, 300)
trend = np.convolve(noise, np.ones(10) / 10, mode='same')  # 平滑处理
y_sim = trend * 3

plot_refined_elegant_line(
    x_sim, y_sim,
    title="Match Momentum Analysis",
    xlabel="Points",
    ylabel="Momentum Index",
    output_filename="refined_chart"
)