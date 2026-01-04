import pandas as pd
from tools.base_param import BaseParam
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
import json
from collections import Counter
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import gaussian_kde
from draw_param import DrawParam
step1 = True  # 先处理并将结果保存到json文件中

d_params = DrawParam()
params = BaseParam()
magic_os = params.test_magic_os
backbone_color_map = d_params.backbone_color_map
color_list = plt.cm.tab20
backbones = ['Qwen', 'GLM-4', 'DeepSeek', 'Mistral', 'Gemini', 'GPT4o']
show_types = [1]
# 1:  画核密度估计图

class GetDict:
    def __init__(self, backbone):
        self.df = pd.read_csv(magic_os + f'magicMOO_{backbone}.csv')
        self.dic_avg = {}
        self.dic_cnt = {}
        self.dic_avg_his = {}

    def run(self):
        # 处理历史结果列
        self.process_h('h_imp_rss')
        self.process_h('h_imp_css')
        # 计算最终结果平均值
        self.eval_avg_res('f_gen_rounds')
        self.eval_avg_res('f_rk_rounds')
        self.eval_avg_res('f_imp_rounds')
        self.eval_avg_res('f_exp_rounds')
        self.eval_avg_res('f_imp_t')
        self.eval_avg_res('f_exp_t')
        self.eval_avg_res('f_imp_s')
        self.eval_avg_res('f_exp_s')
        # 计算历史结果平均值
        self.eval_avg_his_res('h_imp_ss')
        self.eval_avg_his_res('h_imp_rss')
        self.eval_avg_his_res('h_imp_css')
        self.eval_avg_his_res('h_imp_ts')
        self.eval_avg_his_res('h_exp_ss')
        self.eval_avg_his_res('h_exp_ts')
        # print(self.dic_avg)
        # print(self.dic_cnt)
        # print(self.dic_avg_his)
        return self.dic_avg, self.dic_cnt, self.dic_avg_his

    def process_h(self, col_name):
        # 只将总分取出
        res = self.df[col_name].values.tolist()
        for i in range(len(res)):
            t_li = eval(res[i])
            new_res_i = []
            for j in range(len(t_li)):
                if t_li[j] is None:
                    new_res_i.append(None)
                    continue
                new_res_i.append(t_li[j][1])
            res[i] = str(new_res_i)
        self.df[col_name] = res

    def eval_avg_res(self, col_name):
        res = self.df[col_name].values.tolist()
        avg_res = sum(res) / len(res)
        # 四舍五入取整
        avg_res = round(avg_res, 1)
        self.dic_avg[col_name] = avg_res
        if 'rounds' in col_name:
            # 每个值的个数
            cnt = Counter(res)
            cnt = dict(cnt)
            cnt = {k: v for k, v in sorted(cnt.items(), key=lambda item: item[0], reverse=False)}
            self.dic_cnt[col_name] = cnt

    def eval_avg_his_res(self, col_name):
        '''
        1. 先找到某历史结果列的最长长度
        2. 按照df的行数，将每个历史结果列的值取出，然后按照最长长度进行填充，多余的用None填充
        3. 计算每列的平均值
        '''
        res = self.df[col_name].values.tolist()
        for i in range(len(res)):
            res[i] = eval(res[i])
        max_len = 0
        for i in res:
            if len(i) > max_len:
                max_len = len(i)
                # print(i)
        new_res = []
        for i in res:
            if len(i) < max_len:
                i += [np.nan] * (max_len - len(i))
            new_res.append(i)
        # 转为numpy数组，再计算每列的平均值
        new_res = np.array(new_res)
        new_res = np.where(new_res == None, np.nan, new_res)
        avg_res_list = []
        for i in range(max_len):
            avg_res = np.nanmean(new_res[:, i])
            avg_res = round(avg_res, 4)
            avg_res_list.append(avg_res)
        self.dic_avg_his[col_name] = avg_res_list


if step1:
    backbones = ['baichuan', 'qwen', 'glm4', 'deepseek', 'mistral', 'gemini', 'gpt4o']
    dic_all = {}
    for bk in backbones:
        gd = GetDict(bk)
        dic_avg, dic_cnt, dic_avg_his = gd.run()
        print(dic_avg)
        print(dic_cnt)
        print(dic_avg_his)
        print('=====================')
        if bk in ['baichuan', 'qwen', 'mistral', 'gemini']:
            # 首字母大写
            bk = bk.title()
        elif bk == 'glm4':
            bk = 'GLM-4'
        elif bk == 'gpt4o':
            bk = 'GPT4o'
        elif bk == 'deepseek':
            bk = 'DeepSeek'
        dic_all[bk] = [dic_avg, dic_cnt, dic_avg_his]
    print(dic_all)
    # 存入当前文件夹的json文件
    with open(params.test_os + 'draw_figs/res.json', 'w+') as f:
        json.dump(dic_all, f)
    print('json文件保存成功！')

if 1 in show_types:
    '''
    统计平均结果将平均结果绘制为密度曲线图
    子图，每个子图代表一个backbone，作用：证明我们模型反馈的有效性与泛化性
    每个子图形式：曲线图（最终评估得分图）
    每个子图：
    x轴：轮次
    y轴：得分（0-1）
    '''
    plt.rcParams['font.size'] = 25  # 20
    # 从json文件中读取数据
    with open(params.test_os + 'draw_figs/res.json', 'r+') as f:
        dic_all = json.load(f)
    # 创建子图
    plt.rcParams['font.size'] = 20
    fig, axs = plt.subplots(1, 3, figsize=(18, 6), sharey=False)
    num_rounds = np.arange(1, 10)  # 轮次从1开始
    backbones = ['Qwen', 'GLM-4', 'DeepSeek']
    for i, bk in enumerate(backbones):
        h_imp_ss = dic_all[bk][2]['h_imp_ss']
        h_imp_ts = dic_all[bk][2]['h_imp_ts']
        h_exp_ss = dic_all[bk][2]['h_exp_ss']
        h_exp_ts = dic_all[bk][2]['h_exp_ts']
        # 如果是DeepSeek，i从3开始
        if i > 2:
            i -= 3
        # 绘制曲线图
        l = 3
        m = 12
        if bk == 'DeepSeek':
            num_rounds=range(1,6)
        axs[i].plot(num_rounds, h_imp_ss[:len(num_rounds)], label='IScore', marker='o', color=color_list(6),
                    linewidth=l, markersize=m)
        axs[i].plot(num_rounds, h_imp_ts[:len(num_rounds)], label='IThreshold', marker='x', color=color_list(7),
                    linestyle='--', linewidth=l, markersize=m)
        axs[i].plot(num_rounds, h_exp_ss[:len(num_rounds)], label='EScore', marker='s', color=color_list(0),
                    linewidth=l, markersize=m)
        axs[i].plot(num_rounds, h_exp_ts[:len(num_rounds)], label='EThreshold', marker='^', color=color_list(1),
                    linestyle='--', linewidth=l, markersize=m)
        # 添加标签和标题
        axs[i].set_title(bk, pad=10, fontsize=25, fontweight='bold')
        axs[i].set_xlabel('Iteration Round', fontsize=25)
        axs[i].set_ylabel('Score Value', fontsize=25)
        axs[i].set_ylim(0.4, 1.05)  # 设置 y 轴范围
        handles, labels = axs[i].get_legend_handles_labels()
        axs[i].legend(handles, labels, loc='lower left', fontsize=18, frameon=True, framealpha=0.8)
        axs[i].grid(True)


    # # 获取统一图例句柄（从第一个子图获取）
    # handles, labels = axs[0].get_legend_handles_labels()
    # # 添加统一图例（放在图下方中间）
    # fig.legend(handles, labels, loc='lower center', ncol=4, fontsize=20, bbox_to_anchor=(0.5, -0.05))

    plt.tight_layout()
    # plt.suptitle('Effectiveness and Generalizability of MOO', fontsize=16, y=1.08)  # 添加总标题
    plt.show()
    # 保存pdf
    pdf_pages = PdfPages(params.f_draw_pdf_os + 'fig_c5_itera_line.pdf')
    pdf_pages.savefig(fig)
    pdf_pages.close()

if 2 in show_types:
    '''
    核密度估计图，x轴为轮次，y轴为分数区间，z轴为密度, 不同的backbone不同的子图，一共有3个子图
    分数区间：0-0.2,0.2-0.4,0.4-0.6,0.6-0.8,0.8-1
    收敛性
    '''

    # 从csv文件中读取数据
    def processItera(df):
        df_ = d_params.pre_process_Upper(df)
        # 列名修改
        df_.rename(columns={'h_imp_ss': 'IScore', 'h_exp_ss': 'EScore'}, inplace=True)
        return df_


    # bks = ['glm4', 'qwen', 'baichuan', 'mistral', 'gemini']
    bks = ['glm4', 'qwen', 'baichuan']
    df_all = pd.DataFrame()
    for bk in bks:
        f = magic_os + f'magicMOO_{bk}.csv'
        df = pd.read_csv(f)
        df = df[df['method'].isin(['magic_moo'])]
        df = df[['backbone', 'method','h_imp_ss', 'h_exp_ss']]
        df_all = pd.concat([df_all, df])
        # 索引重置
        df_all.reset_index(drop=True, inplace=True)

    df_all = processItera(df_all)
    df_all = df_all[['backbone', 'IScore', 'EScore']]


    def p_figs(sname):
        data = []
        for i in range(len(df_all)):
            tmp = eval(df_all.iloc[i][sname])
            if len(tmp) < 10:
                tmp += [None] * (10 - len(tmp))
            for j in range(10):
                data.append({'backbone': df_all.iloc[i]['backbone'], 'round': j + 1, sname: tmp[j]})
        # 使用 pd.DataFrame 构建新 DataFrame
        df_new = pd.DataFrame(data)
        # 分数区间
        bins = [0, 0.2, 0.4, 0.6, 0.8, 1]
        labels = ["0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0"]

        # 分箱
        df_new['score_bin'] = pd.cut(df_new[sname], bins=bins, labels=labels)

        # 删除缺失值
        df_plot = df_new.dropna(subset=[sname])

        # 创建三维图
        fig = plt.figure(figsize=(20, 6))
        unique_backbones = df_plot['backbone'].unique()
        num_backbones = len(unique_backbones)

        for idx, backbone in enumerate(unique_backbones):
            ax = fig.add_subplot(1, num_backbones, idx + 1, projection='3d')
            data = df_plot[df_plot['backbone'] == backbone]

            # 获取轮次和分数区间数据
            rounds = data['round']
            scores = data[sname]

            # 创建网格
            x = np.linspace(rounds.min(), rounds.max(), 50)
            y = np.linspace(scores.min(), scores.max(), 50)
            X, Y = np.meshgrid(x, y)
            positions = np.vstack([X.ravel(), Y.ravel()])

            # 计算核密度估计
            kde = gaussian_kde(np.vstack([rounds, scores]))
            Z = np.reshape(kde(positions), X.shape)

            # 绘制曲面
            ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.9)

            # 设置标题和标签
            ax.set_title(f"3D KDE for {backbone} {sname}", fontsize=20, pad=10)
            ax.set_xlabel("Rounds", labelpad=10, fontsize=15)
            ax.set_ylabel(f"{sname} Range", labelpad=10, fontsize=15)
            ax.set_zlabel("Density", labelpad=10, fontsize=15)

            ax.set_zlim(0, 2)
            # # 调整视角
            # ax.view_init(elev=30, azim=-45)  # 设置仰角和方位角，调整观察角度

        plt.tight_layout()
        plt.show()

    p_figs('IScore')
    p_figs('EScore')
