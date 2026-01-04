import pandas as pd
import numpy as np
from tqdm import tqdm
from utils.SearchHanzi import Hanzi
from utils.InfoProcess import GetMoreInfo
from utils.preprocess_ori_data import trans_dy, keep_chinese, keep_chinese_and_pipe
import argparse
from utils.base_param import *

params = BaseParam()
from utils.LLMs import *
from prompts.promptsMetrics import MetricsPromptDesign
from utils.ChineseNames import ChineseNames

mpd = MetricsPromptDesign()
llm = Kimi()
df_p = pd.read_csv(params.poems_all_v1_os + 'poems_All.csv', low_memory=False)
df_p = df_p.fillna('')
df_p['cont_text'] = df_p['content'].apply(lambda x: re.sub(r'[^\u4e00-\u9fa5]', '', x))
df_p.drop_duplicates(subset=['cont_text'], keep='first', inplace=True)
df_p['title_text'] = df_p['title'].apply(lambda x: re.sub(r'[^\u4e00-\u9fa5]', '', x))


class Evaluator:
    def __init__(self, user_query, gen_cont_all, gen_cont, exp):
        self.p_emoc = mpd.EMOC
        self.p_crc = mpd.CRC
        self.p_lc = mpd.LC
        self.p_acc = mpd.Acc
        self.user_query = user_query
        self.gen_cont_all = gen_cont_all
        self.gen_cont = gen_cont
        self.exp = exp

    def eval_EMOC(self):
        prompt_emoc = self.p_emoc.format(user_query=self.user_query, gen_cont=self.gen_cont, exp=self.exp)
        res, _ = getLlmRes(llm, prompt_emoc, '评估分数')
        return res['评估分数']

    def eval_CRC(self):
        prompt_comp = self.p_crc.format(user_query=self.user_query, gen_cont=self.gen_cont, exp=self.exp)
        res, _ = getLlmRes(llm, prompt_comp, '评估分数')
        return res['评估分数']

    def eval_LC(self):
        prompt_lc = self.p_lc.format(user_query=self.user_query, gen_cont=self.gen_cont, exp=self.exp)
        res, _ = getLlmRes(llm, prompt_lc, '评估分数')
        return res['评估分数']

    def eval_Acc(self):
        def check_poems(poem_info):
            for p in poem_info:
                try:
                    tmp = p['题目']
                except:
                    p['题目'] = '无'
                if p['题目'] == '无' and p['诗人'] == '无' and p['朝代'] == '无' and p['诗句'] == '无':
                    continue
                title = keep_chinese(p['题目']) if p['题目'] != '无' else ''
                author = p['诗人'] if p['诗人'] != '无' else ''
                dynasty = trans_dy(p['朝代']) if p['朝代'] != '无' else ''
                content = keep_chinese_and_pipe(p['诗句']) if p['诗句'] != '无' else ''
                if '|' not in content:
                    # 直接检查古诗的信息
                    df_ = df_p[(df_p['title_text'].str.contains(title))
                               & (df_p['author'].str.contains(author))
                               & (df_p['dynasty'].str.contains(dynasty))
                               & (df_p['cont_text'].str.contains(content))]
                else:
                    # 检查诗句是否出自同一首古诗
                    df_ = df_p[(df_p['title_text'].str.contains(title))
                               & (df_p['author'].str.contains(author))
                               & (df_p['dynasty'].str.contains(dynasty))]
                    con_li = content.split('|')
                    df_cont = pd.DataFrame()
                    for con in con_li:
                        if df_cont.shape[0] == 0:
                            df_cont = df_[(df_['cont_text'].str.contains(con))]
                        else:
                            df_tmp = df_[(df_['cont_text'].str.contains(con))]
                            tmp_ids = set(df_tmp.index.values.tolist())
                            cont_ids = set(df_cont.index.values.tolist())
                            if len(tmp_ids & cont_ids) > 0:
                                df_cont = df_cont.loc[list(tmp_ids & cont_ids)]
                    df_ = df_cont
                if df_.shape[0] == 0:
                    return False
            return True

        def check_wuxing(haw_info):
            if len(haw_info) == 0 or haw_info == '无':
                return True
            haw_dic = {}
            for i in haw_info:
                try:
                    haw_dic[i['字']] = i['五行']
                except:
                    pass
            h_li = haw_dic.keys()
            hz = Hanzi()
            h_wuxing = {h: hz.getWuxingByHanzi(h) for h in h_li}
            for h in h_li:
                if haw_dic[h] != h_wuxing[h]:
                    return False
            return True

        def extract_date_time(query):
            # 定义正则表达式匹配日期（如：2028年1月15日）
            date_pattern_year = r'(\d{4})年'
            date_pattern_month = r'(\d{1,2})月'
            date_pattern_day = r'(\d{1,2})日'
            date_pattern_minute = r'(\d{1,2})分'
            date_pattern_second = r'(\d{1,2})秒'
            # 分别定义匹配具体时间点（如：3点、6点）和时间段说明（如：午夜、傍晚）
            date_pattern_hour = r'(\d{1,2})点'
            period_pattern = r'\d{1,2}日?(上午|中午|下午|凌晨|晚|早|傍晚|晚上|午夜)'
            # 处理季节的简写和完整形式
            season_pattern = r'年(春|夏|秋|冬)'
            # 匹配日期
            match_y = re.search(date_pattern_year, query)
            if match_y:
                year = int(match_y.group(1))
            else:
                year = '00'
            match_m = re.search(date_pattern_month, query)
            if match_m:
                month = int(match_m.group(1))
            else:
                month = '00'
            match_d = re.search(date_pattern_day, query)
            if match_d:
                day = int(match_d.group(1))
            else:
                day = '00'
            match_h = re.search(date_pattern_hour, query)
            if match_h:
                hour_ = int(match_h.group(1))
            else:
                hour_ = 00
            # 匹配时间段说明
            hour_period_match = re.search(period_pattern, query)
            if hour_period_match:
                period = hour_period_match.group(1)
                # 如果匹配到"傍晚"，并且时间小于12，则时间加12小时
                if period in ['下午', '晚', '傍晚', '晚上'] and hour_ < 12 and hour_ > 0:
                    hour_ += 12
                    # print(query)
            if hour_ == 0:
                hour_ = '00'
            match_m = re.search(date_pattern_minute, query)
            if match_m:
                minute = int(match_m.group(1))
            else:
                minute = '00'
            match_s = re.search(date_pattern_second, query)
            if match_s:
                second = int(match_s.group(1))
            else:
                second = '00'
            birth_date = f'{year}-{month}-{day}-{hour_}-{minute}-{second}'
            # 匹配季节
            season_match = re.search(season_pattern, query)
            if season_match:
                season = season_match.group(1)
            else:
                season = None

            return birth_date, season

        def check_born(born_info):
            birth, season = extract_date_time(self.user_query)
            ef = GetMoreInfo(birth=birth, season=season)
            expand_info = ef.get_baby_info_new()
            acc_li = [1] * 6
            if born_info['生肖'] != '无' and expand_info['生肖'] != born_info['生肖']:
                acc_li[0] = 0
            if born_info['季节'] != '无' and expand_info['季节'][0] != born_info['季节'][0]:
                acc_li[1] = 0
            if born_info['节气'] != '无' and expand_info['节气'] != born_info['节气']:
                acc_li[2] = 0
            if born_info['节日'] != '无' and expand_info['节日'] != born_info['节日']:
                acc_li[3] = 0
            if born_info['八字'] != '无':
                gen_bz = keep_chinese(born_info['八字'])
                fact_bz = keep_chinese(expand_info['八字和五行'])
                if len(gen_bz) < len(fact_bz):
                    fact_bz = re.sub(r'\(.*?\)', '', expand_info['八字和五行']).strip()
                if gen_bz != fact_bz:
                    acc_li[4] = 0
            if born_info['五行缺失'] != '无':
                gen_wx = born_info['五行缺失'].split('|')
                gen_wx = [i for i in gen_wx if i != '']
                fact_wx = expand_info['五行缺失']
                if set(gen_wx) & set(fact_wx) != set(gen_wx):
                    acc_li[5] = 0
            return acc_li

        prompt_fa = self.p_acc.format(exp=self.exp)
        infos, _ = getLlmRes(llm, prompt_fa, '古诗信息')
        f_poems = check_poems(poem_info=infos['古诗信息'])
        f_wuxing = check_wuxing(haw_info=infos['字和五行'])
        f_born = check_born(born_info=infos['出生信息'])
        f_poems = 1 if f_poems else 0
        f_wuxing = 1 if f_wuxing else 0
        f_li = [f_poems, f_wuxing]
        f_li.extend(f_born)
        return f_li

    def run(self):
        emoc = self.eval_EMOC()
        crc = self.eval_CRC()
        lc = self.eval_LC()
        acc = self.eval_Acc()
        return emoc, crc, lc, acc


def eval_Nov(name_li):
    '''
    （√）NU👆: Name uniqueness, 名字独特性，1~6越高分越独特，NU = 2 和 3 分别表示 1/100 和 1/1000 的人在名字中使用了这个字符（在他们的出生年份）。
    （√）CCU👆: Character corpus uniqueness, 字符语料库独特性1~6(基于当代中文语料库中某个字符的使用频率来计算的独特性指标。与NU不同，CCU衡量的是日常语言使用中字符的流行度，而不是名字中的使用频率。)
    （√）NV👆: Name valence, 名字情感价值 基于16位中文评价者对2614个名字字符意义的积极程度的主观评分（1到5分）。（1 =非常负面，3 =中性，5 =非常正面）
    （√）NW👆: Name warmth, 名字温暖度/道德感 基于10位中文评价者对名字中包含的字符可能带来的温暖相关特质的主观评分（1到5分）。（1 =极不可能具有，3 =中等可能性，5 =极有可能具有）
    （√）NC👆: Name competence, 名字能力/自信 基于10位中文评价者对名字中包含的字符可能带来的能力相关特质的主观评分（1到5分）。（1 =极不可能具备，3 =可能性中等，5 =极有可能具备）。
    '''
    cn = ChineseNames()
    df_res = cn.compute_name_index(name=name_li, birth=[0] * len(name_li))
    df = df_res[['NU', 'CCU', 'NV', 'NW', 'NC']].copy()
    # 将df的每一列都进行最大最小归一化
    for col in ['NU', 'CCU']:
        df.loc[:, col] = (df[col] - 1) / (6 - 1)
    for col in ['NV', 'NW', 'NC']:
        df.loc[:, col] = (df[col] - 1) / (5 - 1)
    # 再求平均值
    df.loc[:, 'Nov'] = (df['NU'] + df['CCU'] + df['NV'] + df['NW'] + df['NC']) / 5
    # df['name'] = name_li
    # print(df)
    return df['Nov']


def weighted_average(weights, values):  # 定义一个函数来计算加权平均
    if values is np.nan:
        return np.nan
    try:
        # 确保weights和values长度相同
        weights = eval(weights)
        # 计算加权平均
        weights = np.array(weights)
        values = np.array(values)
        return np.round(np.sum(weights * values), 4)
    except Exception as e:
        print(e)
    return np.nan


def eval_res(df, num, f_eval):
    if os.path.exists(f_eval):
        df_res = pd.read_csv(f_eval)
    else:
        df_res = pd.DataFrame(columns=['query', 'name', 'exp', 'r_poem', 'backbone', 'method', 'up_w', 'output',
                                       'nov', 'emoc', 'crc', 'lc', 'acc'])
        df_res.to_csv(f_eval, index=False, encoding='utf-8')

    # 新颖度
    df_nov = eval_Nov(name_li=df['name'].values.tolist()[:num])
    df.loc[:num - 1, 'nov'] = df_nov.values.tolist()
    # print(df_nov.values)

    for i in tqdm(range(num)):
        # 若已经计算过，则跳过
        df_find = df_res[(df_res['query'] == df.loc[i, 'query'])
                         & (df_res['backbone'] == df.loc[i, 'backbone'])
                         & (df_res['method'] == df.loc[i, 'method'])]
        if df_find.shape[0] > 0:
            continue
        # 该方法不在评估列表中，则跳过
        if df.loc[i, 'method'] not in eval(args.method_li):
            continue
        try:
            user_query = df.loc[i, 'query']
            gen_cont = df.loc[i, 'name']
            exp = df.loc[i, 'exp']
            gen_cont_all = '{}。解释：{}'.format(gen_cont, exp)
            evaluator = Evaluator(user_query=user_query, gen_cont_all=gen_cont_all, gen_cont=gen_cont, exp=exp)
            emoc, crc, lc, acc = evaluator.run()
            df.loc[i, 'emoc'] = str(emoc)
            df.loc[i, 'crc'] = str(crc)
            df.loc[i, 'lc'] = str(lc)
            df.loc[i, 'acc'] = str(acc)
            df_tmp = pd.DataFrame({'query': [user_query],
                                   'name': [gen_cont],
                                   'exp': [exp],
                                   'r_poem': [df.loc[i, 'r_poem']],
                                   'backbone': [df.loc[i, 'backbone']],
                                   'method': [df.loc[i, 'method']],
                                   'up_w': [df.loc[i, 'up_w']],
                                   'output': [df.loc[i, 'output']],
                                   'nov': [df.loc[i, 'nov']],
                                   'emoc': [df.loc[i, 'emoc']],
                                   'crc': [df.loc[i, 'crc']],
                                   'lc': [df.loc[i, 'lc']],
                                   'acc': [df.loc[i, 'acc']]})
            df_tmp.to_csv(f_eval, index=False, encoding='utf-8', mode='a', header=False)
        except Exception as e:
            print(e)
            print('Error in line:', i)
            # time.sleep(30)
            continue


def calc_scores(f_in, f_new, f_summary):
    # 重新读取res，计算其他指标计算
    df = pd.read_csv(f_in)
    # 处理异常值
    # 将crc里面len小于5的去掉
    df = df[df['crc'].apply(lambda x: len(eval(x)) == 5)]
    df = df[df['lc'].apply(lambda x: len(eval(x)) == 5)]
    df = df[df['emoc'].apply(lambda x: len(eval(x)) == 5)]
    # 将crc和lc里面的nan值去掉
    df = df[~df['acc'].isna()]
    df = df[~df['crc'].isna()]
    df = df[~df['lc'].isna()]
    df = df[~df['emoc'].isna()]
    # 删除[]里面存在str的行
    df = df[~df['crc'].apply(lambda x: any([isinstance(i, str) for i in eval(x)]))]
    df = df[~df['lc'].apply(lambda x: any([isinstance(i, str) for i in eval(x)]))]
    df = df[~df['emoc'].apply(lambda x: any([isinstance(i, str) for i in eval(x)]))]

    # 预处理
    def normalize_to_100(x):
        if isinstance(x, str):
            x = eval(x)
        x = np.array(x, dtype=float)  # 转成 NumPy 数组以便向量化处理
        x = (x - 0) / (3 - 0) * 100  # 分数范围：0-3
        x = np.round(x, 4)  # 四舍五入保留两位小数
        return list(x)  # 转回 list

    def weighted_average(weights, values):  # 定义一个函数来计算加权平均
        if values is np.nan:
            return np.nan
        try:
            # 确保weights和values长度相同
            weights = eval(weights)
            # 计算加权平均
            weights = np.array(weights)
            values = np.array(values)
            return np.round(np.sum(weights * values) / 1, 4)  # 权重总和为1
        except Exception as e:
            print(e)
        return np.nan

    # EMOC, EMOC(std) [3,2,2,3,1] -> 标准差
    df.loc[:, 'emoc_n'] = df.loc[:, 'emoc'].apply(lambda x: normalize_to_100(x))  # 归一化（0-3）到（0-100）
    df.loc[:, 'emoc_w'] = df.loc[:,['up_w','emoc_n']].apply(lambda row: weighted_average(row['up_w'], row['emoc_n']), axis=1)  # 计算加权平均
    df.loc[:, 'emoc_std'] = df.loc[:, 'emoc_n'].apply(lambda x: np.round(np.std(x), 4))  # 标准差，多目标稳定性
    # CRC
    df.loc[:, 'crc_n'] = df.loc[:, 'crc'].apply(lambda x: normalize_to_100(x))
    df.loc[:, 'crc_avg'] = df.loc[:, 'crc_n'].apply(lambda x: np.round(np.mean(x), 4))
    # LR
    df.loc[:, 'lr_n'] = df.loc[:, 'lc'].apply(lambda x: normalize_to_100(x))
    df.loc[:, 'lr_avg'] = df.loc[:, 'lr_n'].apply(lambda x: np.round(np.mean(x), 4))
    # ACC
    df.loc[:, 'acc_n'] = df.loc[:, 'acc'].apply(
        lambda x: round(sum(eval(x)) / len(eval(x)) * 100, 2))  # 准确率=正确的数量/总数*100
    # 计算综合评分
    df[['emoc_w', 'acc_n', 'emoc_std', 'crc_avg', 'lr_avg']] = df[
        ['emoc_w', 'acc_n', 'emoc_std', 'crc_avg', 'lr_avg']].astype(float)
    # IMOC, IMOC(std)
    df.loc[:, 'imp'] = (1 / 3) * df.loc[:, 'acc_n'] + (1 / 3) * df.loc[:, 'crc_avg'] + (1 / 3) * df.loc[:, 'lr_avg']
    df.loc[:, 'imp_std'] = df.loc[:, ['acc_n', 'crc_avg', 'lr_avg']].std(axis=1)  # 计算标准差
    # CMOC
    df.loc[:, 'cmoc'] = 0.5 * df.loc[:, 'emoc_w'] + 0.5 * df.loc[:, 'imp']
    df.loc[:, 'cmoc_std'] = df[['emoc_w', 'imp']].std(axis=1)  # 计算标准差
    # 重新保存结果
    df_new = df.replace('', np.nan)
    df_new.to_csv(f_new, index=False, encoding='utf-8')

    # 计算每个方法的平均值
    df_need = df[['backbone', 'method',
                  'emoc_w', 'emoc_std',
                  'crc_avg', 'lr_avg', 'acc_n',
                  'imp', 'imp_std',
                  'cmoc', 'cmoc_std']]
    # 按照backbone和method分组，计算每列的平均值，存入新的df
    df_summary = df_need.groupby(['backbone', 'method']).mean().reset_index()

    if os.path.exists(f_summary):
        # 若已存在相同backbone和method的数据，则用新的数据替换
        df_ori = pd.read_csv(f_summary)
        for i in range(df_summary.shape[0]):
            df_tmp = df_summary.iloc[i, :]
            df_find = df_ori[(df_ori['backbone'] == df_tmp['backbone']) & (df_ori['method'] == df_tmp['method'])]
            if df_find.shape[0] > 0:
                # 如果数据存在，则把旧的数据删除
                df_ori = df_ori[~((df_ori['backbone'] == df_tmp['backbone']) & (df_ori['method'] == df_tmp['method']))]
            # 将新的数据添加到df_ori中
            df_add = pd.DataFrame(df_tmp).T
            df_ori = pd.concat([df_ori, df_add], axis=0)
            # 保存结果
            df_ori.to_csv(f_summary, index=False, encoding='utf-8')
            print('Update scores:', df_tmp['backbone'], df_tmp['method'])
    else:
        # 创建新文件并保存结果
        df_summary.to_csv(f_summary, index=False, encoding='utf-8')


def addScoresToCSV():
    # model_li = ['baichuan', 'gemini', 'gpt4o_mini', 'gpt4o', 'gpt4', 'glm4','glm-4-long','glm-4-flash', 'mistral', 'qwen']
    model_li = ['qwen', 'glm4', 'deepseek', 'gemini', 'mistral', 'gpt4o']
    # model_li = ['qwen']
    for m in tqdm(model_li):
        print(f'Evaluate {m}...')
        f_eval_res_emoc = params.f_eval_res_emoc.format(m)  # '0914/eval_results_{}_emoc.csv'
        f_eval_final_res = params.f_eval_final_res.format(m)  # 'final_res/eval_results_{}_final.csv'
        f_scores = params.f_eval_res_scores  # 'eval_results_scores.csv'
        calc_scores(f_in = f_eval_res_emoc, f_new=f_eval_final_res, f_summary = f_scores)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate the results of the methods.")
    # parser.add_argument("-b", "--backbone", required=True, help="Evaluate the results from the backbone.")
    # parser.add_argument("-m", "--method_li", default="['base','fewshot','CoT','TDB','llm_discussion','query2keyword','magic_moo']",
    #                     help="Evaluate the results from the method.")
    # parser.add_argument("-s1", "--step1", default=True, help="Evaluate the results.")
    # parser.add_argument("-s2", "--step2", default=True, help="Evaluate the final scores.")
    args = parser.parse_args()
    # ====测试====
    args.step1 = False
    args.step2 = True
    args.backbone = 'gemini' # qwen
    args.method_li = "['magic_moo']" # magic_moo_wo-evalExp
    # ============
    step1 = args.step1
    step2 = args.step2
    # 评估
    if step1:  # 分别评估每个模型的结果
        model = args.backbone  # 'baichuan','qwen', 'mistral', 'gemini', 'glm-4-long','gpt4o',
        print(f'Evaluate {model}...{args.method_li}')
        if 'magic_moo' in args.method_li:
            f = params.test_bl_os + f'magicMOO/magicMOO_{model}.csv'  # todo: 修改路径
            method = eval(args.method_li)[0]
            if 'wo' in method:
                ab = method.split('_')[-1]
                f = params.test_bl_os + f'magicMOO/magicMOO_{model}_{ab}.csv'  # 'wo-R', 'wo-evalR', 'wo-Imp', 'wo-Exp', 'wo-evalGen'
        else:
            f = params.test_bl_os + f'0818/baseline_{model}.csv'
        f_eval_res = params.f_eval_res.format(model)  # '0914/eval_results_{}.csv'
        f_eval_res_emoc = params.f_eval_res_emoc.format(model)  # '0914/eval_results_{}_emoc.csv'
        df_ = pd.read_csv(f)
        num = df_.shape[0]
        eval_res(df_, num, f_eval_res_emoc)  # 评估
        print('Done!')
    if step2:  # 计算最终得分
        addScoresToCSV()
        print('Done!')

    # # 单独测试
    # args.method_li = "['base','fewshot','CoT','TDB','llm_discussion','RAG']"
    # metod = 'RAG'  # base,fewshot,CoT,RAG
    # model = 'glm-4-flash' # 'baichuan','qwen', 'mistral', 'gemini', 'glm-4-flash','gpt4o',
    # f = params.test_bl_os + f'0818/baseline_{model}.csv'
    # f_eval_res = params.f_eval_res.format(model)  # '0818/eval_results_{}.csv'
    # df_ = pd.read_csv(f)
    # print(f'Evaluate {model} {metod}...')
    # num = 20
    # df_ = df_.loc[df_['method'] == metod].reset_index(drop=True)
    # eval_res(df_, num, f_eval_res)
    # addScoresToCSV()
