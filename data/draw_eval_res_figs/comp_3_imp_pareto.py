import pandas as pd
from tools.base_param import BaseParam
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
params = BaseParam()
plt.rcParams['font.size'] = 19 # 25
from draw_param import DrawParam
d_params = DrawParam()
def processImp(df):
    df_ = d_params.pre_process(df)
    # 原地修改列名
    df_.rename(columns={'lr_avg': 'LR', 'acc_n': 'ACC', 'crc_avg': 'CRC'}, inplace=True)
    return df_

f_eval_res = params.f_eval_res_scores
df = pd.read_csv(f_eval_res)
# 取出LC\ACC\CRC
df_new = df[['backbone', 'method', 'lr_avg', 'acc_n', 'crc_avg']]  # lc,acc,crc
df_new = processImp(df_new)
# color_list = plt.cm.tab20
methods = d_params.methods_
backbones = d_params.backbones_
# 设置backbone颜色映射
backbone_color_map = d_params.backbone_color_map
methods_color_map = d_params.methods_color_map
indicators = ['LR', 'ACC', 'CRC']
show_types = [1, 2]

if 1 in show_types:
    # 柱状图6个子图，每个子图：左边是CRC，右边是ACC，范围40-100，x轴是method，不同颜色代表不同的backbone，同一个method有两个柱子代表CRC和ACC，柱子的颜色一个深色一个浅色
    fig, axs = plt.subplots(2, 3, figsize=(14, 8)) # 27, 14
    axs = axs.flatten()  # 展平数组方便迭代

    # 每个子图对应一个backbone
    for i, backbone in enumerate(backbones):
        ax = axs[i]

        df_backbone = df_new[df_new['backbone'] == backbone]
        width = 0.3  # 每个柱子的宽度
        x = np.arange(len(methods))  # X轴位置

        # 绘制左侧Y轴的CRC和右侧Y轴的ACC
        for j, method in enumerate(methods):
            crc_values = df_backbone[df_backbone['method'] == method]['CRC'].values
            acc_values = df_backbone[df_backbone['method'] == method]['ACC'].values
            lc_values = df_backbone[df_backbone['method'] == method]['LR'].values

            # 如果没有数据，用0替代
            crc = crc_values[0] if len(crc_values) > 0 else 0
            acc = acc_values[0] if len(acc_values) > 0 else 0
            lc = lc_values[0] if len(lc_values) > 0 else 0

            # # 每个柱子标上数值，灰色显示，小一点
            # fontsize = 8
            # ax.text(x[j] - width, crc, f'{crc:.2f}', ha='center', va='bottom', color='grey', fontsize=fontsize)
            # ax.text(x[j], acc, f'{acc:.2f}', ha='center', va='bottom', color='grey', fontsize=fontsize)
            # ax.text(x[j] + width, lc, f'{lc:.2f}', ha='center', va='bottom', color='grey', fontsize=fontsize)


            # ACC柱子，浅色
            ax.bar(x[j] - width, acc, width, label=f'{backbone} ACC' if j == 0 else "",
                   color=backbone_color_map[backbone])

            # CRC柱子，深色
            ax.bar(x[j], crc, width, label=f'{backbone} CRC' if j == 0 else "",
                   color=backbone_color_map[backbone], alpha=0.6)

            # LC柱子，浅色
            ax.bar(x[j] + width, lc, width, label=f'{backbone} LR' if j == 0 else "",
                   color=backbone_color_map[backbone], alpha=0.3)
            # ax.axhline(y=80, color='red', linestyle='--',alpha=0.3)  # 红色虚线

        # 设置 X 和 Y 轴
        ax.set_xticks(x)
        methods_new = ['B', 'C', 'T', 'Fs', 'Q', 'LD', 'NG']
        # ax.set_xticklabels(methods_new, rotation=45, ha='right')
        ax.set_xticklabels(methods_new,ha='center')
        if backbone in ['DeepSeek']:
            ax.set_ylim(50, 100)
        else:
            ax.set_ylim(30, 100)
        # 添加网格
        ax.grid(axis='y', linestyle='--', alpha=0.6)

        # 添加标题和标签
        ax.set_title(f'{backbone}', pad=8, fontweight='bold')
        if i % 3 == 0:
            ax.set_ylabel("Score", fontsize=22)
            ax.yaxis.set_label_coords(-0.13, 0.5)  # 更靠近 y 轴中心线
        # ax.legend(loc='upper left', fontsize=20)  # 每个子图单独添加图例

    # 在所有子图之后统一添加图例
    from matplotlib.patches import Patch
    # 添加统一图例
    legend_elements = [
        Patch(facecolor='gray', alpha=1.0, label='ACC'),
        Patch(facecolor='gray', alpha=0.6, label='CRC'),
        Patch(facecolor='gray', alpha=0.3, label='LR')
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=3, fontsize=18, frameon=False,
               bbox_to_anchor=(0.52, 0.01))

    # 调整布局
    fig.tight_layout(rect=[0, 0.05, 0.97, 1]) # rect=[left, bottom, right, top]
    plt.subplots_adjust(wspace=0.18, hspace=0.29)  # 调整子图间距

    # # 调整布局
    # fig.tight_layout()

    plt.show()
    # 保存pdf
    pdf_pages = PdfPages('pdf/fig_c3_imp_bar.pdf')
    pdf_pages.savefig(fig)
    pdf_pages.close()

if 2 in show_types:
    plt.rcParams['font.size'] = 15 # 25
    # 散点图：横坐标ACC, 纵坐标CRC, 每个点代表某个backbone的某个method的数据，不同颜色代表不同的backbone，不同形状代表不同的method
    # 为不同的method设置形状
    method_marker_map = {
        methods[0]: 'o',  # 圆形
        methods[1]: 's',  # 方形
        methods[2]: 'D',  # 菱形
        methods[3]: '^',  # 三角形
        methods[4]: 'v',  # 倒三角
        methods[5]: 'P',  # 五角星
        methods[6]: 'X'  # X
    }

    # 创建图形和子图
    fig, axs = plt.subplots(1, 3, figsize=(12, 5)) # 27,8

    # 为每个子图添加数据
    for ax, (x_col, y_col, title) in zip(axs, [('CRC', 'ACC', 'CRC vs. ACC'),
                                               ('LR',  'ACC', 'LR vs. ACC'),
                                               ('LR',  'CRC', 'LR vs. CRC')]):

        # 保存已经绘制过的标签，以避免重复
        labels = []
        Xs = []
        Ys = []
        # 遍历每个backbone和method，绘制散点图
        for backbone in backbones:
            df_backbone = df_new[df_new['backbone'] == backbone]
            for method in methods:
                df_method = df_backbone[df_backbone['method'] == method]

                # 获取 x 和 y 的值
                x_values = df_method[x_col].values
                y_values = df_method[y_col].values
                Xs.extend(x_values)
                Ys.extend(y_values)

                # 如果有数据，则绘制散点
                if len(x_values) > 0 and len(y_values) > 0:
                    label = f'{backbone} - {method}'
                    ax.scatter(
                        x_values, y_values,
                        label=label if label not in labels else "",  # 确保每个label只添加一次
                        color=backbone_color_map[backbone],
                        marker=method_marker_map[method],
                        s=230,  # 点的大小
                        edgecolor='black',  # 边框颜色
                        alpha=0.7
                    )
                    labels.append(label)  # 添加已使用的标签

        # 假设X和Y都是最大化目标
        maxX = True
        maxY = True
        sorted_list = sorted([[Xs[i], Ys[i]] for i in range(len(Xs))], reverse=maxX)
        pareto_front = [sorted_list[0]]
        for pair in sorted_list[1:]:
            if maxY:
                if pair[1] >= pareto_front[-1][1]:
                    pareto_front.append(pair)
            else:
                if pair[1] <= pareto_front[-1][1]:
                    pareto_front.append(pair)

        pareto_front = np.array(pareto_front)

        # 绘制帕累托前沿
        if pareto_front.size > 0:  # 只有在帕累托前沿存在时才绘制
            ax.plot(pareto_front[:, 0], pareto_front[:, 1], color='b', linestyle='--', label='Pareto Front',
                    linewidth=5)

        # 设置轴标签和标题
        ax.set_xlabel(x_col, fontsize=16)
        ax.xaxis.set_label_coords(0.5, -0.11)  # 更靠近 x 轴中心线
        ax.set_ylabel(y_col, fontsize=16)
        ax.yaxis.set_label_coords(-0.11, 0.5)  # 更靠近 y 轴中心线
        ax.set_ylim(40, 100)
        # ax.set_xlim(65,100)
        ax.set_title(title, pad=8, fontweight='bold', fontsize=17)

        ax.axhline(y=80, color='red', linestyle='--', label='y = 80', linewidth=4, alpha=0.7)
        ax.axvline(x=80, color='red', linestyle='--', label='x = 80', linewidth=4, alpha=0.7)  # 红色虚线

    # 创建颜色图例（显示backbone颜色）
    color_handles = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=backbone_color_map[backbone],
                   markersize=20, label=backbone) for backbone in backbones
    ]

    # 创建空心形状图例（显示method的形状，黑色边框）
    shape_handles = [
        plt.scatter([], [], marker=method_marker_map[method], color='black', label=method,
                    edgecolor='black', facecolor='none', s=350) for method in methods
    ]

    # # 合并两个图例的句柄
    # handles = color_handles + shape_handles

    # # 调整子图布局，留右侧和顶部空间
    fig.subplots_adjust(left=0.06, right=0.98, top=0.9, bottom=0.31,wspace=0.25)

    # === 💡 核心：按列优先重新组合 handles ===
    handles = []
    for c, s in zip(shape_handles, color_handles):
        handles.extend([c, s])  # 列顺序拼接
    # shape_handles 比 color_handles 多一个，所以最后一个 shape_handles 不会被添加
    handles.append(shape_handles[-1])  # 添加最后一个 shape_handles

    fig.legend(handles=handles,
               loc='lower center',
               bbox_to_anchor=(0.5, 0),
               fontsize=16,
               frameon=False,
               handleheight=1.5,  # 👈 控制两行之间的间距
               handletextpad=0.1, # 图标与字的间距
               ncol=7,
                columnspacing=1.5)

    plt.show()

    # 保存pdf
    pdf_pages = PdfPages('pdf/fig_c3_imp_pareto.pdf')
    pdf_pages.savefig(fig)
    pdf_pages.close()
