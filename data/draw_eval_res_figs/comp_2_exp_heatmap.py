'''
显式多目标热力图
保留up_w，
测试每个目标关注度都有的情况下模型的表现如何
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
from tools.base_param import BaseParam

params = BaseParam()
from draw_param import DrawParam

d_params = DrawParam()
backbone_color_map = d_params.backbone_color_map
plt.rcParams['font.size'] = 30


def processExp(df):
    df_new = d_params.pre_process(df)
    # 原地修改列名
    df_ = df_new.rename(columns={'emoc_n': 'EMOC'})
    return df_


bks = ['qwen', 'glm4', 'deepseek', 'mistral', 'gemini', 'gpt4o']
# 合并所有的csv文件：eval_results_{}_final.csv
df_all = pd.DataFrame()
for bk in bks:
    f = params.f_eval_final_res.format(bk)
    df = pd.read_csv(f)
    df = df[['backbone', 'method', 'emoc_n', 'up_w']]
    df_all = pd.concat([df_all, df])
    # 索引重置
    df_all.reset_index(drop=True, inplace=True)

df_all = df_all[['backbone', 'method', 'emoc_n']]
df_all = processExp(df_all)

# df_new = pd.DataFrame(columns=['backbone', 'method', 'Cultural', 'Expectations', 'B&W', 'Personal', 'Others'])
df_new = pd.DataFrame(columns=['backbone', 'method', 'O1', 'O2', 'O3', 'O4', 'O5'])
for i in range(len(df_all)):
    try:
        tmp = eval(df_all.iloc[i]['EMOC'])
        df_new.loc[i] = [df_all.iloc[i]['backbone'], df_all.iloc[i]['method']] + tmp
    except:
        pass

# df_new 按照backbone进行分组计算每个列的均值
df_n = df_new.groupby(['backbone', 'method']).mean().reset_index()
# method_order = d_params.methods_
trans_method = {
    'Base': 'B',
    'CoT': 'C',
    'TDB': 'T',
    'Few-shot': 'Fs',
    'Q2Kw': 'Q',
    'LLM-D': 'LD',
    'NAMeGEn': 'NG'
}
# 将原df_n的method名字依据trans_method进行转换
df_n['method'] = df_n['method'].replace(trans_method)
method_order = ['B', 'C', 'T', 'Fs', 'Q', 'LD', 'NG']
df_n['method'] = pd.Categorical(df_n['method'], categories=method_order, ordered=True)  # 重新排序
df_n = df_n.sort_values(by=['backbone', 'method'])  # 按照backbone和method排序
# cultural significance, parental expectation, bazi wuxing, personal characteristics, other requirements

'''
每个backbone为一个子图，
每个子图：为热力图，x轴是method，y轴是指标，颜色代表指标值
'''
# 提取所有的backbone
backbones = d_params.backbones_

# 设置画布和子图（2行3列）
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(25, 16))
# 扁平化axes数组，便于遍历
axes = axes.flatten()

# 统一设置颜色范围（根据数据自适应或手动设定）
vmin = df_n[['O1', 'O2', 'O3', 'O4', 'O5']].min().min()
vmax = df_n[['O1', 'O2', 'O3', 'O4', 'O5']].max().max()
cmap = 'RdYlBu'  # YlGnBu,PiYG,RdYlBu

# 对每个backbone创建热力图
for i, backbone in enumerate(backbones):
    # 筛选出当前backbone的数据
    data = df_n[df_n['backbone'] == backbone].drop('backbone', axis=1)
    # 创建指标列表
    # indicators = ['Cultural', 'Expectations', 'B&W', 'Personal', 'Others']
    indicators = ['O1', 'O2', 'O3', 'O4', 'O5']
    # 转换为适合热力图的数据格式
    heatmap_data = data[indicators].values.T  # 转置后的数据用于热力图
    sns.heatmap(heatmap_data,
                annot=True,
                fmt='.0f',  # 设置数字格式 .2f
                cmap=cmap,
                xticklabels=data['method'],
                yticklabels=indicators,
                cbar=False,  # 不要每张图都加 colorbar
                # cbar_kws={'label': 'Score'},
                ax=axes[i],
                linewidths=0.5,
                linecolor='gray',
                vmin=vmin, vmax=vmax)
    # 设置格子里面的数字的大小
    for t in axes[i].texts:
        t.set_fontsize(35)
    # 设置标题和背景颜色
    axes[i].set_title(f'{backbone}',pad=15,fontsize=40, fontweight='bold')
    axes[i].title.set_position([0.5, 1.5])
    # axes[i].set_xticklabels(axes[i].get_xticklabels(), rotation=45, ha='right')
    axes[i].set_yticklabels(axes[i].get_yticklabels(), rotation=360, fontsize=35)
    axes[i].set_xticklabels(axes[i].get_xticklabels(), fontsize=35)
    axes[i].set_facecolor(backbone_color_map.get(backbone, 'white'))
    # # 设置热力图lenend区间
    # cbar = axes[i].collections[0].colorbar

from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

# 添加一个统一的 colorbar
# cbar_ax = fig.add_axes([0.92, 0.3, 0.015, 0.4])  # [left, bottom, width, height] 右侧
# cbar_ax = fig.add_axes([0.3, 0.95, 0.4, 0.03])  # [left, bottom, width, height] 顶部
cbar_ax = fig.add_axes([0.3, 0.05, 0.4, 0.03])  # [left, bottom, width, height] 底部
sm = ScalarMappable(cmap=cmap, norm=Normalize(vmin=vmin, vmax=vmax))
sm.set_array([])
# fig.colorbar(sm, cax=cbar_ax, orientation='horizontal', label='Score') # orientation='horizontal' 设置 colorbar 为横向；
cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
cbar.set_label('Score', labelpad=8, fontsize=35)  # 设置 colorbar 的标签
cbar.ax.xaxis.set_label_position('top')  # 标签在上面（紧贴色条上方）

# 布局调整
# plt.tight_layout()
# plt.subplots_adjust(top=0.88,bottom=0.05,left=0.03,right=0.97,wspace=0.12) # 顶部
plt.subplots_adjust(top=0.95,bottom=0.16,left=0.05,right=0.95,wspace=0.15, hspace=0.27) # 底部
plt.show()

# 保存pdf
pdf_pages = PdfPages(params.f_draw_pdf_os + 'fig_c2_exp_hmap.pdf')
pdf_pages.savefig(fig)
pdf_pages.close()
