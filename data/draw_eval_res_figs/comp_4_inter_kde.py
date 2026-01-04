import pandas as pd
from tools.base_param import BaseParam
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
import sys
import os
sys.path.append(r'D:\A_MyStudy\AI_Naming\dataset\test_dataset\draw_figs')
from draw_param import DrawParam

d_params = DrawParam()

params = BaseParam()
magic_os = params.test_magic_os
color_list = plt.cm.tab20
# 全局设置字体
# plt.rcParams['font.family'] = 'DejaVu Sans'  # 更换为你需要的字体
plt.rcParams['font.size'] = 23  # 20
# # 标题字体设置
plt.rcParams['axes.titlesize'] = 23  # 20
# plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['axes.labelsize'] = 20  # 20
# 图例字体设置
plt.rcParams['legend.fontsize'] = 18  # 15
from draw_param import DrawParam

d_params = DrawParam()


def processInter(df):
    df_ = d_params.pre_process_Upper(df)
    # 原地修改列名；'f_rk_rounds', 'f_gen_rounds', 'f_imp_rounds', 'f_exp_rounds'
    df_.rename(columns={'f_rk_rounds': 'Retrieval', 'f_gen_rounds': 'Generation',
                        'f_imp_rounds': 'Implicit Evaluation', 'f_exp_rounds': 'Explicit Evaluation'}, inplace=True)
    return df_


bks = ['qwen', 'glm4', 'deepseek', 'mistral', 'gemini', 'gpt4o']
df_all = pd.DataFrame()
for bk in bks:
    f = magic_os + f'magicMOO_{bk}.csv'
    df = pd.read_csv(f)
    df = df[df['method'].isin(['magic_moo'])]
    # # 检查df的数据是否存在重复值
    # print(df.duplicated().sum())
    df = df[['backbone', 'method', 'f_rk_rounds', 'f_gen_rounds', 'f_imp_rounds', 'f_exp_rounds']]
    df_all = pd.concat([df_all, df])
    df_all.reset_index(drop=True, inplace=True)

df_all = processInter(df_all)
df_all = df_all[['backbone', 'Retrieval', 'Generation', 'Implicit Evaluation', 'Explicit Evaluation']]
# 四个指标
metrics = ['Retrieval', 'Generation', 'Implicit Evaluation', 'Explicit Evaluation']

'''
使用核密度估计展示数据分布
在核密度估计（KDE）图中，密度（Density）是一个统计量，
用于估计数据在某个特定值附近的概率密度。密度值越高，
表示在该值附近数据点的集中程度越高。
KDE图通过平滑处理，提供了数据分布的连续估计，帮助我们理解数据的分布特征，
如集中趋势、离散程度和偏斜情况。
'''

# 一张图
plt.rcParams['font.size'] = 22
plt.rcParams['legend.fontsize'] = 20
plt.rcParams['axes.labelsize'] = 28
plt.rcParams['axes.titlesize'] = 30
df_all['Total requests'] = (df_all['Retrieval'] + df_all['Generation']
                            + df_all['Implicit Evaluation']+ df_all['Explicit Evaluation'])
fig = plt.figure(figsize=(16, 8.5))
ax = sns.kdeplot(data=df_all,
                 x='Total requests',
                 hue="backbone",
                 palette=d_params.backbone_color_map,
                 linewidth=2.5,
                 fill=True,
                 common_norm=False,
                 alpha=0.2)
plt.grid(axis='x', linestyle='--', alpha=0.6)
plt.grid(axis='y', linestyle='--', alpha=0.6)
legend = ax.get_legend()
if legend:  # 检查图例是否存在
    legend.set_title(None)  # 移除图例标题
# x轴数值是整数
plt.xticks(np.arange(0, 36, 2))
plt.title("Total API",  pad=8, fontweight='bold') # Interaction Distribution of Overall API
plt.ylabel("Density", labelpad=8)
plt.xlabel("Requests", labelpad=8)
plt.tight_layout()
fig.subplots_adjust(right=0.92)
plt.show()
# 保存pdf
pdf_pages = PdfPages(params.f_draw_pdf_os + 'fig_c4_inter_kde_overall.pdf')
pdf_pages.savefig(fig)
pdf_pages.close()

# 分图
plt.rcParams['font.size'] = 22  # 20
plt.rcParams['axes.titlesize'] = 30  # 20
plt.rcParams['axes.labelsize'] = 28  # 20
# plt.rcParams['legend.fontsize'] = 20  # 15
fig, axs = plt.subplots(2, 2, figsize=(18, 10))
for ax, metric in zip(axs.flat, metrics):
    sns.kdeplot(data=df_all,
                x=metric,
                hue="backbone",
                palette=d_params.backbone_color_map,
                ax=ax,
                linewidth=2.5,
                fill=True,
                common_norm=False,
                alpha=0.2)

    ax.grid(axis='x', linestyle='--', alpha=0.6)
    ax.grid(axis='y', linestyle='--', alpha=0.6)
    ax.set_title(f"{metric}", pad=10, fontweight='bold') # Iteration Distribution of {metric}
    ax.set_xlabel("Rounds")
    if metric == 'Generation':
        ax.set_xticks(np.arange(0, 16, 2))
    ax.set_ylabel("Density")
    legend = ax.get_legend()
    if legend:  # 检查图例是否存在
        legend.set_title(None)  # 移除图例标题
        # legend.set_visible(False)

plt.tight_layout()
plt.show()

# 保存pdf
pdf_pages = PdfPages(params.f_draw_pdf_os + 'fig_c4_inter_kde_detail.pdf')
pdf_pages.savefig(fig)
pdf_pages.close()
