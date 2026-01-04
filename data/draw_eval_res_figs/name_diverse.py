'''
尝试使用：同一个backbone，同一个query下，不同的method所生成name的差异性来衡量名字的多样性
我们的方法在所有backbone上都是最高的
'''
import pandas as pd
import os
from tqdm import tqdm
from tools.base_param import BaseParam
params = BaseParam()
f_eval = params.f_eval_final_res
backbones = ['qwen','glm4', 'deepseek', 'mistral', 'gemini', 'gpt4o']
methods = ['base','fewshot','CoT','TDB','query2keyword','llm_discussion','magic_moo']
df_plt = pd.DataFrame(columns=['backbone','method','nov_mean'])


for bk in backbones:
    f_m = f_eval.format(bk)
    df_m = pd.read_csv(f_m)
    if bk=='qwen':
        df_m = df_m[df_m['method'].isin(methods)] # qwen 找出对比实验的数据
        df_m.reset_index(drop=True, inplace=True)
    for row in tqdm(range(df_m.shape[0])):
        q = df_m.loc[row, 'query']
        name = df_m.loc[row, 'name']
        name_tmp_list = df_m[df_m['query']==q]['name'].tolist()
        name_tmp_list.remove(name)
        nov = 1 if name not in name_tmp_list else 0
        df_m.loc[row, 'nov'] = nov

    print(bk)
    # 按照method group by 计算 nov_mean
    nov_mean = df_m.groupby('method')['nov'].mean()
    print(nov_mean)
    dic_d = dict(nov_mean)
    for k,v in dic_d.items():
        if k not in methods:
            continue
        df_tmp = pd.DataFrame({'backbone':[bk],'method':[k],'div_mean':[v]})
        df_plt = pd.concat([df_plt, df_tmp], axis=0)
    print('-----------------')

import sys
sys.path.append(os.path.abspath(params.test_er_os+'/draw_figs'))
from draw_param import DrawParam
d_params = DrawParam()
df_plt = d_params.pre_process(df_plt)
# 乘上100，保存结果
df_plt['nov_mean'] = df_plt['nov_mean']*100
# 保存结果
df_plt.to_csv(params.test_er_os+'eval_res_nov.csv', index=False)

# 画出柱状图
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="whitegrid")
plt.figure(figsize=(15, 10))
# plt.ylim(50, 1.0)
sns.barplot(x='backbone', y='nov_mean', hue='method', data=df_plt)
plt.show()

