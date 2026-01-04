'''
cmoc的w和avg的雷达图
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from math import pi
from tools.base_param import BaseParam

params = BaseParam()
import sys

sys.path.append(params.test_os + 'draw_figs/')
from draw_param import DrawParam

d_params = DrawParam()
# 设定颜色
methods_color_map = d_params.methods_color_map
backbone_color_map = d_params.backbone_color_map


# 预处理数据
def processCCmoc(df):
    # 处理CMOC
    df_new = d_params.pre_process(df)
    # 原地修改列名'mocr_avg','mocr_w'
    df_new.rename(columns={'emoc_std': 'EMOC(std)', 'imp_std': 'IMOC(std)', 'cmoc_std': 'CMOC(std)'}, inplace=True)
    return df_new


f_res = params.f_eval_res_scores
df = pd.read_csv(f_res)
df_new = df[['backbone', 'method', 'emoc_std', 'imp_std', 'cmoc_std']]
df_new = processCCmoc(df_new)
backbones = d_params.backbones_
methods = d_params.methods_
metrics = ['EMOC(std)', 'IMOC(std)', 'CMOC(std)']
# 1: 柱状图
# 2: 分别的雷达图
# 3: 合并的雷达图
show_types = [3]  # [1,2]

if 1 in show_types:
    # 可视化标准差，每个backbone画一个柱状图
    N = len(methods)
    # 2行3列的画布
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    # 遍历每个backbone
    for i, bk in enumerate(backbones):
        # 获取对应 backbone 的数据
        backbone_data = df_new[df_new['backbone'] == bk]
        # 排序
        backbone_data.sort_values(by="CMOC(std)", ascending=False, inplace=True)
        # 获取对应方法的指标数据
        values = backbone_data["CMOC(std)"].values.tolist()
        methods = backbone_data["method"].values.tolist()
        print(methods)
        # 获取当前子图的坐标
        ax = axs[i // 3, i % 3]
        # 绘制柱状图
        bars = ax.bar(range(len(methods)), values, color=backbone_color_map[bk], alpha=0.8)
        # CMOC Standard Deviation of
        ax.set_title(bk, fontsize=20)
        ax.set_xticks(range(len(methods)))  # 设置位置
        ax.set_xticklabels(methods, rotation=45, ha='right')  # 设置标签和格式
        ax.set_ylabel("CMOC(std)", fontsize=15)
        ax.grid(axis='y', linestyle='--', alpha=0.6)
        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, height + 0.5, f'{height:.2f}', ha='center', va='bottom',
                    fontsize=8)

    plt.tight_layout()
    plt.show()

if 2 in show_types:
    # o: 表示性能较好， solid：中文，dashdot：英文模型
    linestyle_map = {
        'DeepSeek': ('solid', 'o'),
        'GLM-4': ('solid', 'x'),
        'Qwen': ('solid', 'x'),
        'GPT4o': ('dashdot', 'o'),
        'Gemini': ('dashdot', 'x'),
        'Mistral': ('dashdot', 'x'),
    }
    N = len(methods)  # 方法的数量
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]  # 闭合图形

    for metric in metrics:
        # 创建单独的雷达图
        fig, ax = plt.subplots(figsize=(7, 5), subplot_kw=dict(polar=True))

        # ax.set_title(metric, fontsize=20, color='black', pad=20)

        ax.set_theta_offset(np.pi / 2)  # 将雷达图的起始点设置在顶部
        ax.set_theta_direction(-1)  # 逆时针方向

        # 设置每个方法的角度标签
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(methods, fontsize=24)
        # 调整刻度标签与轴的距离
        ax.tick_params(axis='x', which='major', pad=8)

        # # 设置 Y 轴范围
        # ax.set_ylim(50, 100)

        # 为每个 backbone 绘制一条线
        for bk in backbones:
            # 获取对应 backbone 和指标的数据
            backbone_data = df_new[df_new['backbone'] == bk]
            values = backbone_data[metric].values.tolist()
            values += values[:1]  # 闭合图形

            # 获取线型和标记
            linestyle, marker = linestyle_map[bk]
            # 绘制雷达图的线
            ax.plot(
                angles,
                values,
                color=backbone_color_map[bk],
                linewidth=2,
                linestyle=linestyle,
                marker=marker,
                label=bk,
            )
            ax.fill(angles, values, color=backbone_color_map[bk], alpha=0.25)  # 填充区域

        # 调整 Y 轴标值的颜色
        ax.tick_params(axis='y', colors='grey', labelsize=15)

        if metric=='IMOC(std)':
            # 显示图例
            ax.legend(loc='upper right', bbox_to_anchor=(1.63, 1.2), fontsize=18)

        # 调整布局并显示
        plt.tight_layout()
        plt.show()

        # 保存pdf
        m_tmp = metric.replace('(', '_').replace(')', '')
        pdf_pages = PdfPages(params.f_draw_pdf_os + f'fig_c1_{m_tmp}_radar.pdf')
        pdf_pages.savefig(fig)
        pdf_pages.close()

if 3 in show_types:
    # o: 表示性能较好， solid：中文，dashdot：英文模型
    linestyle_map = {
        'DeepSeek': ('solid', 'o'),
        'GLM-4': ('solid', 'x'),
        'Qwen': ('solid', 'x'),
        'GPT4o': ('dashdot', 'o'),
        'Gemini': ('dashdot', 'x'),
        'Mistral': ('dashdot', 'x'),
    }
    N = len(methods)  # 方法的数量
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]  # 闭合图形

    # 创建一行三列的图
    fig, axes = plt.subplots(1, 3, figsize=(10, 5), subplot_kw=dict(polar=True))
    # fig, axes = plt.subplots(1, 2, figsize=(14, 9), subplot_kw=dict(polar=True)) # 设置图形大小

    metrics_set = metrics[:3]

    ms = ['EC_std', 'IC_std', 'CC_std']

    for idx, metric in enumerate(metrics_set):  # 设置指标个数
        ax = axes[idx]
        # 设置标题
        ax.set_title(ms[idx], fontsize=17, color='black', pad=10, fontweight='bold')

        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)

        ax.set_xticks(angles[:-1])
        methods_new =['B', 'C', 'T', 'Fs', 'Q', 'LD', 'NG']
        ax.set_xticklabels(methods_new, fontsize=16)
        ax.tick_params(axis='x', which='major', pad=3)
        for bk in backbones:
            backbone_data = df_new[df_new['backbone'] == bk]
            values = backbone_data[metric].values.tolist()
            values += values[:1]

            linestyle, marker = linestyle_map[bk]
            if idx == 0:
                ax.plot(
                    angles,
                    values,
                    color=backbone_color_map[bk],
                    linewidth=2,
                    linestyle=linestyle,
                    marker=marker,
                    label=bk,
                )
            else:
                ax.plot(
                    angles,
                    values,
                    color=backbone_color_map[bk],
                    linewidth=2,
                    linestyle=linestyle,
                    marker=marker,
                )
            ax.fill(angles, values, color=backbone_color_map[bk], alpha=0.25)

        ax.tick_params(axis='y', colors='grey', labelsize=12)

    # 设置图例在顶部横排
    handles, labels = axes[0].get_legend_handles_labels()
    # 你想要的列优先顺序（根据 matplotlib 的排列方式）
    labels_col_order = ['Qwen', 'Mistral', 'GLM-4', 'Gemini', 'DeepSeek', 'GPT4o']

    # 创建标签到句柄的映射
    label_to_handle = dict(zip(labels, handles))

    # 根据 labels_col_order 重新排列句柄和标签
    handles_sorted = [label_to_handle[label] for label in labels_col_order]
    labels_sorted = labels_col_order
    fig.legend(
        handles=handles_sorted,
        labels=labels_sorted,
        loc='lower center',
        bbox_to_anchor=(0.5, 0.05),
        fontsize=15,
        frameon=False,
        ncol=3
    )

    fig.subplots_adjust(wspace=0.3)  # 调整子图间距

    # 调整布局并显示
    # plt.tight_layout()
    plt.show()

    # 保存到PDF
    pdf_pages = PdfPages(params.f_draw_pdf_os + 'fig_c1_combined_radar.pdf')
    pdf_pages.savefig(fig)
    pdf_pages.close()

