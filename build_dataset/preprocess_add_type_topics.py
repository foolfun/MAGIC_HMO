'''

chinses-poetry-master文件夹中的json文件合并为csv文件，得到古诗类型type列和topics列
poems-db-master得到topics列
原poems_all_v3.csv文件中的tags列取部分内容合并到topics列

'''
import os
import json
import pandas as pd
from opencc import OpenCC
from tqdm import tqdm
import numpy as np
from utils.base_param import BaseParam
from build_dataset.preprocess_ori_data import trans_con, trans_df, keep_chinese_and_pipe
from build_dataset.preprocess_llms_data import poemsLinkHanzi

param = BaseParam()
file_base = param.chinese_poetry_os
f_v3_os = param.poems_all_v3_os


def bulid_topics():
    df = pd.read_csv(f_v3_os + 'poems_all_v3.csv')
    # print(df.isnull().any())
    df_p = pd.read_csv(param.pdm_os + 'poems_db_fin.csv', low_memory=False)
    try:
        # 删除Unnamed: 0列
        df.drop(columns=['Unnamed: 0'], inplace=True)
        df.drop(columns=['tags_tmp'], inplace=True)
    except:
        pass

    def check_tags(s, t):
        if tags[t] != '未知' and tags[t] != '无' and tags[t] != '' and tags[t] != '无特定节日' and tags[t] != '不详':
            if type(tags[t]) == list:
                for item in tags[t]:
                    s += item.replace('、', '|').replace('，', '').replace(',', '') + '|'
            else:
                s += tags[t].replace('、', '|').replace('，', '').replace(',', '') + '|'
        return s

    for i in tqdm(df.index):
        tags = eval(df.loc[i, 'tags'])
        s = ''
        # s = check_tags(s, 'event')
        # s = check_tags(s, 'event_time')
        # s = check_tags(s, 'event_location')
        s = check_tags(s, 'event_holiday')
        s = check_tags(s, 'event_season')
        s = check_tags(s, 'event_weather')
        s = check_tags(s, 'sentiment')
        df.loc[i, 'tags_short'] = s[:-1]

    # df_p的content和tag里面的topics取出来
    df_p['topics'] = df_p['tags'].apply(lambda x: eval(x)['topics'])
    df_p = df_p[['content', 'topics']]
    # 把topics为空的行删除
    df_p = df_p[df_p['topics'] != '']
    # 去重
    df_p.drop_duplicates(inplace=True)
    # 如果content重复，则保留topics更长的结果
    df_p = df_p.sort_values(by='topics', ascending=False).drop_duplicates(subset='content', keep='first')
    # 遍历df_p的topics列，如果包含noise的词，则删除
    noise = '唐诗三百首学高初年级教师赋序'
    for i in tqdm(df_p.index):
        topics = df_p.loc[i, 'topics']
        t_li = topics.split(';')
        for t in t_li:
            if set(t) & set(noise):
                t_li.remove(t)
        df_p.loc[i, 'topics'] = '|'.join(t_li)
    # 检查是否有重复的content
    duplicate_content_in_df_p = df_p[df_p.duplicated(subset=['content'], keep=False)]
    print(duplicate_content_in_df_p)
    duplicate_content_in_df = df[df.duplicated(subset=['content'], keep=False)]
    print(duplicate_content_in_df)

    # 将df_p的topics列合并到df，以content为连接桥梁,仅保留df的content
    df_m = pd.merge(df, df_p[['content', 'topics']], how='left', on='content')

    # topics列为nan的改为空
    df_m['topics'] = df_m['topics'].fillna('')

    for i in tqdm(df_m.index):
        if df_m.loc[i, 'topics'] != '':
            df_m.loc[i, 'tags_short'] = df_m.loc[i, 'tags_short'] + '|' + df_m.loc[i, 'topics'].replace(';', '|')

    # 删除原topics列
    df_m.drop(columns=['topics'], inplace=True)
    df_m.rename(columns={'tags_short': 'topics'}, inplace=True)
    df_m['topics'] = df_m['topics'].apply(lambda x: '|'.join(part for part in set(x.split('|')) if part))
    # 保存
    df_m.to_csv(f_v3_os + 'poems_all_v3_02.csv', index=False)


def add_type(f_li_n, file_os, type_name):
    data = []
    for j in f_li_n:
        with open(file_os + '/' + j, 'r') as f:
            a = json.load(f)
            if type(a) == list:
                data.append(a)
            else:
                data.append([a])
    data = sum(data, [])
    df = pd.DataFrame(data)
    df['type'] = type_name
    try:
        try:
            df['content'] = df['paragraphs'].apply(lambda x: trans_con(str(x)))
        except:
            df['content'] = df['content'].apply(lambda x: trans_con(str(x)))
    except:
        df['content'] = df['para'].apply(lambda x: trans_con(str(x)))
    return df


def wudai01():  # ['花间集', '南唐二主词']
    file_os = file_base + '五代诗词/' + 'huajianji'
    # 花间集
    f_li_n = os.listdir(file_os)
    f_li_n.remove('huajianji-0-preface.json')
    f_li_n.remove('README.md')
    df1 = add_type(f_li_n, file_os, '花间集')
    # 南唐二主词
    file_os = file_base + '五代诗词/' + 'nantang'
    f_li_n = os.listdir(file_os)
    file_os = [i for i in f_li_n if 'poetrys' in i]
    df2 = add_type(file_os, file_base + '五代诗词/nantang', '南唐二主词')
    df = pd.concat([df1, df2], axis=0)
    df['type'] = df['type'] + '|' + df['rhythmic']
    return df


def yuanqu02():  # 元曲
    file_os = file_base + '元曲'
    f_li_n = os.listdir(file_os)
    df = add_type(f_li_n, file_os, '元曲')
    return df


def tangsong03():
    # 全唐诗
    file_os = file_base + '全唐诗'
    f_all = os.listdir(file_os)
    f_li_n = [i for i in f_all if 'poet.tang' in i]  # 找到poet.tang.x.json文件
    df1 = add_type(f_li_n, file_os, '全唐诗')
    # 全宋诗
    f_li_n = [i for i in f_all if 'poet.song' in i]  # 找到poet.song.x.json文件
    df2 = add_type(f_li_n, file_os, '全宋词')
    # 唐诗三百首
    f_li_n = [i for i in f_all if '唐诗三百首' in i]  # 找到 唐诗三百首.json 文件
    df3 = add_type(f_li_n, file_os, '唐诗三百首')
    df = pd.concat([df1, df2, df3], axis=0)
    return df


def sishuwujing04():  # 四书五经
    file_os = file_base + '四书五经'
    f_all = os.listdir(file_os)
    f_li_n = [i for i in f_all if 'daxue.json' in i]  # 找到poet.tang.x.json文件
    df1 = add_type(f_li_n, file_os, '四书五经|大学')
    f_li_n = [i for i in f_all if 'mengzi.json' in i]  # 找到poet.tang.x.json文件
    df2 = add_type(f_li_n, file_os, '四书五经|孟子')
    df2['chapter'] = '孟子·' + df2['chapter']
    f_li_n = [i for i in f_all if 'zhongyong.json' in i]  # 找到poet.tang.x.json文件
    df3 = add_type(f_li_n, file_os, '四书五经|中庸')
    df = pd.concat([df1, df2, df3], axis=0)
    df['title'] = df['chapter']
    # 删除chapter列
    df.drop('chapter', axis=1, inplace=True)
    return df


def songci05():  # 宋词
    file_os = file_base + '宋词'
    f_li_n = os.listdir(file_os)
    f_li_n = [i for i in f_li_n if 'ci.song.' in i]  # 找到poet.tang.x.json文件
    df = add_type(f_li_n, file_os, '宋词')
    return df


def youmeng06():  # 幽梦影
    file_os = file_base + '幽梦影'
    f_li_n = os.listdir(file_os)
    f_li_n = [i for i in f_li_n if '.json' in i]  # 找到poet.tang.x.json文件
    df = add_type(f_li_n, file_os, '幽梦影')
    return df


def yudingquantangshi07():  # 御定全唐詩
    file_os = file_base + '御定全唐詩/json'
    f_li_n = os.listdir(file_os)
    f_li_n = [i for i in f_li_n if '.json' in i]  # 找到poet.tang.x.json文件
    df = add_type(f_li_n, file_os, '御定全唐詩')
    return df


def caocao08():  # 曹操
    file_os = file_base + '曹操诗集'
    f_li_n = os.listdir(file_os)
    f_li_n = [i for i in f_li_n if '.json' in i]  # 找到poet.tang.x.json文件
    df = add_type(f_li_n, file_os, '曹操')
    return df


def chuci09():  # 楚辞
    file_os = file_base + '楚辞'
    f_li_n = os.listdir(file_os)
    f_li_n = [i for i in f_li_n if '.json' in i]  # 找到poet.tang.x.json文件
    df = add_type(f_li_n, file_os, '楚辞')
    return df


def shuimo10():  # 水墨唐诗
    file_os = file_base + '水墨唐诗'
    f_li_n = os.listdir(file_os)
    f_li_n = [i for i in f_li_n if '.json' in i]  # 找到poet.tang.x.json文件
    df = add_type(f_li_n, file_os, '水墨唐诗')
    return df


def nlxd11():  # 纳兰性德
    file_os = file_base + '纳兰性德'
    f_li_n = os.listdir(file_os)
    f_li_n = [i for i in f_li_n if '.json' in i]  # 找到poet.tang.x.json文件
    df = add_type(f_li_n, file_os, '纳兰性德')
    return df


def mengxue12():  # 蒙学
    file_os = file_base + '蒙学'
    f_li_n = os.listdir(file_os)
    f_li_n = [i for i in f_li_n if '.json' in i]  # 找到poet.tang.x.json文件
    df = add_type(f_li_n, file_os, '蒙学')
    return df


def lunyu13():  # 论语
    file_os = file_base + '论语'
    f_li_n = os.listdir(file_os)
    f_li_n = [i for i in f_li_n if '.json' in i]  # 找到poet.tang.x.json文件
    df = add_type(f_li_n, file_os, '论语')
    df['title'] = '论语·' + df['chapter']
    # 删除chapter列
    df.drop('chapter', axis=1, inplace=True)
    return df


def shijing14():  # 诗经
    file_os = file_base + '诗经'
    f_li_n = os.listdir(file_os)
    f_li_n = [i for i in f_li_n if '.json' in i]  # 找到poet.tang.x.json文件
    df = add_type(f_li_n, file_os, '诗经')
    df['title'] = df['title'] + '·' + df['chapter'] + '·' + df['section']
    # 删除chapter,section列
    df.drop(['chapter', 'section'], axis=1, inplace=True)
    return df


def merge_type():
    df_1 = wudai01()
    df_2 = yuanqu02()
    df_3 = tangsong03()
    df_4 = sishuwujing04()
    df_5 = songci05()
    df_6 = youmeng06()
    df_7 = yudingquantangshi07()
    df_8 = caocao08()
    df_9 = chuci09()
    df_10 = shuimo10()
    df_11 = nlxd11()
    df_12 = mengxue12()
    df_13 = lunyu13()
    df_14 = shijing14()
    df_mgT = pd.concat([df_1, df_2, df_3, df_4, df_5, df_6, df_7, df_8, df_9, df_10, df_11, df_12, df_13, df_14],
                       axis=0)

    # 处理content
    df_mgT['content'] = df_mgT['content'].apply(lambda x: '|'.join(part for part in x.split('|') if part))
    df_mgT = df_mgT[df_mgT['content'] != '']

    # 合并相同content的type
    df_g = df_mgT.groupby('content')['type'].agg(lambda x: '|'.join(x)).reset_index()
    df_g['type'] = df_g['type'].fillna('')
    df_g['type'] = df_g['type'].apply(
        lambda x: '|'.join(part for part in set(x.split('|')) if part))  # 全唐诗|全唐诗|御定全唐,给type列去重

    # 处理好的type合并到原df
    df_mgT.drop('type', axis=1, inplace=True)
    df_mgT = pd.merge(df_mgT, df_g, on='content', how='left')

    # 繁体转简体
    cc = OpenCC('t2s')  # t2s 表示繁体到简体，s2t 表示简体到繁体
    df_mgT = df_mgT.applymap(lambda x: cc.convert(str(x)))

    # 处理步骤（type合并）完成后才可去重，否则会丢失type信息
    df_mgT.drop_duplicates(subset='content', inplace=True)
    df_mgT.to_csv(file_base + 'merge_data.csv', index=False)


def add_golden_data():
    # 原数据
    df_p = pd.read_csv(f_v3_os + 'poems_all_v3_02.csv', low_memory=False)
    df_p['content_'] = df_p['content'].apply(lambda x: x.replace('|', ''))

    # 标准数据
    df_t = pd.read_csv(file_base + 'merge_data.csv', low_memory=False)  # True data
    df_t['content_'] = df_t['content'].apply(lambda x: x.replace('|', ''))
    df_t['tags'] = df_t['tags'].fillna('')
    for i in tqdm(df_t.index):
        try:
            if df_t.loc[i, 'tags'] != '':
                df_t.loc[i, 'topics'] = '|'.join(eval(df_t.loc[i, 'tags']))
        except:
            df_t.loc[i, 'topics'] = df_t.loc[i, 'tags']
    df_t = df_t[['title', 'dynasty', 'author', 'content', 'topics', 'type', 'content_']]

    # 合并
    df_m = df_p.merge(df_t, on='content_', how='left', suffixes=('_dfp', '_dft'))  # dfp可能有问题，dft正确数据
    # 处理title, author, dynasty的冲突，选择保留dft的信息
    df_m['title'] = df_m['title_dft'].where(df_m['title_dft'].notna(), df_m['title_dfp'])
    df_m['dynasty'] = df_m['dynasty_dft'].where(df_m['dynasty_dft'].notna(), df_m['dynasty_dfp'])
    df_m['author'] = df_m['author_dft'].where(df_m['author_dft'].notna(), df_m['author_dfp'])
    df_m['content'] = df_m['content_dft'].where(df_m['content_dft'].notna(), df_m['content_dfp'])
    df_m['topics_dft'] = df_m['topics_dft'].fillna('')
    df_m['topics_dfp'] = df_m['topics_dfp'].fillna('')

    for i in tqdm(df_m.index):
        if df_m.loc[i, 'topics_dft'] != '':
            df_m.loc[i, 'topics'] = df_m.loc[i, 'topics_dft'] + '|' + df_m.loc[i, 'topics_dfp']
        else:
            df_m.loc[i, 'topics'] = df_m.loc[i, 'topics_dfp']

    df_m.drop(columns=['title_dft', 'title_dfp', 'dynasty_dft', 'dynasty_dfp', 'author_dft', 'author_dfp',
                       'topics_dft', 'topics_dfp', 'content_dft', 'content_dfp', 'content_'], inplace=True)

    df_m['topics'] = df_m['topics'].fillna('')
    df_m['topics'] = df_m['topics'].apply(lambda x: '|'.join(part for part in set(x.split('|')) if part))

    # 核查
    df_m.replace('', np.nan, inplace=True)  # 将所有空字符串替换为NaN
    # title, author, dynasty,content,type,topics列放在前面
    df_m = df_m[['id', 'type', 'title', 'dynasty', 'author', 'content',
                 'topics', 'explain', 'implication', 'tags', 'key_words',
                 'metaphor', 'tags_all', 'retrieval_info', 'words', 'hwuxing',
                 '金_words', '木_words', '水_words', '火_words', '土_words', '无_words']]
    try:
        # 删除Unnamed: 0列
        df_m.drop(columns=['Unnamed: 0'], inplace=True)
    except:
        pass
    print(df_m.isnull().any())
    df_m['long_c'] = df_m['content'].apply(lambda x: x.replace('|', ''))  # 178176 原数据去重
    df_m.sort_values(by=['topics', 'type', 'explain'], ascending=False, inplace=True)
    df_m.drop_duplicates(subset=['long_c'], inplace=True)  # 177938
    df_m.drop(columns=['long_c'], inplace=True)
    df_m_new = trans_df(df_m)
    df_m_new.to_csv(f_v3_os + 'poems_all_v3_03.csv',index=False)


def add_golden_data_part():
    # 原数据
    df_p = pd.read_csv(f_v3_os + 'poems_all_v3_03.csv', low_memory=False)
    df_p['content_'] = df_p['content'].apply(lambda x: x.replace('|', ''))
    # 标准数据
    df_t = pd.read_csv(file_base + 'merge_data.csv', low_memory=False)  # True data
    df_t['content_'] = df_t['content'].apply(lambda x: x.replace('|', ''))
    df_t['tags'] = df_t['tags'].fillna('')
    for i in tqdm(df_t.index):
        try:
            if df_t.loc[i, 'tags'] != '':
                df_t.loc[i, 'topics'] = '|'.join(eval(df_t.loc[i, 'tags']))
        except:
            df_t.loc[i, 'topics'] = df_t.loc[i, 'tags']
    df_t = df_t[['title', 'dynasty', 'author', 'content', 'topics', 'type', 'content_']]

    # 合并
    '''
    前五个字符或后五个字符相同,且长度相同，则认为可能是一首诗.
    对候选可能的诗进行比较，如果相似度大于0.9(不同的概率为0.1)，则认为是同一首诗
    '''
    df_p = df_p.fillna('')
    df_t = df_t.fillna('')
    df_p.sort_values(by='type', ascending=False, inplace=True)
    for i in tqdm(df_p.index):
        if df_p.loc[i, 'type'] != '':
            continue
        start_p = df_p.loc[i, 'content_'][:5]
        end_p = df_p.loc[i, 'content_'][-5:]
        df_tmp = df_t[df_t['content_'].str.startswith(start_p) | df_t['content_'].str.endswith(end_p)]
        for k in df_tmp.index:
            str_p = df_p.loc[i, 'content_']
            str_t = df_t.loc[k, 'content_']
            if len(str_p) == len(str_t):
                diff_count = sum(1 for a, b in zip(str_p, str_t) if a != b)
                if diff_count / len(str_p) > 0.1:
                    continue
                else:
                    # ratio = SequenceMatcher(None, str_p, str_t).ratio()
                    # if ratio > 0.9:
                    type = df_tmp.loc[k, 'type']
                    title = df_tmp.loc[k, 'title'] if df_tmp.loc[k, 'title'] != '' else df_p.loc[i, 'title']
                    title = title.replace('：', '·')
                    dynasty = df_tmp.loc[k, 'dynasty'] if df_tmp.loc[k, 'dynasty'] != '' else df_p.loc[i, 'dynasty']
                    author = df_tmp.loc[k, 'author'] if df_tmp.loc[k, 'author'] != '' else df_p.loc[i, 'author']
                    topics = df_tmp.loc[k, 'topics'] + '|' + df_p.loc[i, 'topics']
                    content = df_tmp.loc[k, 'content']

                    noise = '唐诗三百首学高初年级教师赋序'
                    t_li = topics.split('|')
                    for t in t_li:
                        if set(t) & set(noise):
                            t_li.remove(t)
                    topics = '|'.join(t_li)

                    df_p.loc[i, 'type'] = type
                    df_p.loc[i, 'title'] = title
                    df_p.loc[i, 'dynasty'] = dynasty
                    df_p.loc[i, 'author'] = author
                    df_p.loc[i, 'topics'] = topics
                    df_p.loc[i, 'content'] = content
                    break

    df_p['topics'] = df_p['topics'].fillna('')
    df_p['topics'] = df_p['topics'].apply(lambda x: '|'.join(part for part in set(x.split('|')) if part))

    # 核查
    df_p.replace('', np.nan, inplace=True)  # 将所有空字符串替换为NaN
    # title, author, dynasty,content,type,topics列放在前面
    df_p_new = df_p[['id', 'type', 'title', 'dynasty', 'author', 'content',
                     'topics', 'explain', 'implication', 'tags', 'key_words',
                     'metaphor', 'tags_all', 'retrieval_info', 'words', 'hwuxing',
                     '金_words', '木_words', '水_words', '火_words', '土_words', '无_words']]
    print(df_p_new.isnull().any())  # 177938
    df_p_new['long_c'] = df_p_new['content'].apply(lambda x: x.replace('|', ''))
    print(df_p_new.shape)
    df_p_new.sort_values(by=['type', 'topics', 'explain'], ascending=False, inplace=True)  # 保留有type的
    df_p_new.drop_duplicates(subset=['long_c'], inplace=True, keep='first')
    print('去重后：', df_p_new.shape)  # 176721
    df_p_new.drop(columns=['long_c'], inplace=True)
    df_m_new_ = trans_df(df_p_new)
    df_m_new_.to_csv(f_v3_os + 'poems_all_v3_04.csv',index=False)


def manual_preprocess():
    df = pd.read_csv(f_v3_os + 'poems_all_v3_04.csv', low_memory=False)
    df = df.fillna('')
    try:
        # 删除Unnamed: 0列
        df.drop(columns=['Unnamed: 0'], inplace=True)
    except:
        pass
    # 删除content内容相同的重复行
    df['tmp_content'] = df['content'].apply(lambda x: x.replace('|', ''))
    df.sort_values(by=['title'], ascending=False, inplace=True)
    df.drop_duplicates(subset=['tmp_content'], inplace=True)
    df.drop(columns=['tmp_content'], inplace=True)
    # 增加四书五经的内容
    tmp = df.loc[(df['title'].str.contains('楚辞 ·'))]
    # type 四书五经|
    df.loc[(df['title'] == '大学') & (df['type'] == ''), 'type'] = '四书五经|四书|大学'
    df.loc[(df['title'].str.contains('孟子')) & (df['author'] == '孟子') & (df['type'] == ''), 'type'] = '四书五经|四书|孟子'
    df.loc[(df['title'].str.contains('中庸')) & (df['dynasty'] == '周') & (df['type'] == ''), 'type'] = '四书五经|四书|中庸'
    df.loc[(df['title'].str.contains('论语')) & (df['dynasty'] == '周') & (df['author'] == '孔子'), 'type'] = '四书五经|四书|论语'
    df.loc[(df['title'].str.contains('尚书')) & df['dynasty'].str.contains('周'), 'type'] = '四书五经|五经|尚书'
    df.loc[(df['title'].str.contains('礼记 ')), 'type'] = '四书五经|五经|礼记'
    df.loc[(df['title'].str.contains('周易')), 'type'] = '四书五经|五经|周易'
    df.loc[(df['title'].str.contains('左传 ')) & (df['dynasty'] == '周'), 'type'] = '四书五经|五经|春秋左传'
    df.loc[(df['title'].str.contains('公羊传 ')), 'type'] = '四书五经|五经|春秋公羊传'
    df.loc[(df['title'].str.contains('谷梁传 ')), 'type'] = '四书五经|五经|春秋谷梁传'
    df.loc[(df['title'].str.contains('谷梁传 ')), 'author'] = '谷梁赤'
    df.loc[(df['type'].str.contains('诗经')), 'type'] = '四书五经|五经|诗经'
    rows_to_update =df[((df['title'].str.contains('国风|大雅|小雅|周颂')) & (df['dynasty'].str.contains('周')) & (df['type']==''))]
    df.loc[rows_to_update.index, 'type'] = '四书五经|五经|诗经'
    # 庄子
    df.loc[(df['title'].str.contains('庄子 ·')), 'type'] = '庄子'
    # 山海经
    df.loc[(df['title'].str.contains('山海经 ·')), 'type'] = '山海经'
    # 孙子兵法
    df.loc[(df['title'].str.contains('孙子兵法 ·')), 'type'] = '孙子兵法'
    # 金刚经
    df.loc[(df['title'].str.contains('金刚经 ·')), 'type'] = '金刚经'
    # 佛说四十二章经
    df.loc[(df['title'].str.contains('佛说四十二章经 ·')), 'type'] = '佛说四十二章经'
    # 国语
    df.loc[(df['title'].str.contains('国语 ·')), 'type'] = '国语'
    # 纳兰性德
    df.loc[(df['author'] == '纳兰性德') & (df['dynasty'] == '纳兰性德') & (df['type'] == ''), 'type'] = '纳兰性德'
    # 《靖康小雅》作者
    df.loc[(df['author'] == '《靖康小雅》作者'), 'author'] = '靖康小雅'
    # 楚辞
    df.loc[(df['title'].str.contains('楚辞 ·')), 'type'] = '楚辞'
    for i in df.index:
        if (df.loc[i, 'dynasty'] == '周') and (df.loc[i, 'author'] == '屈原'):
            df.loc[i, 'dynasty'] = '先秦' # 把所有屈原的诗，周变为先秦
    df.loc[(df['title'].str.contains('楚辞|九歌|哀郢|离骚')) & (df['author'] == '屈原'), 'type'] = '楚辞'
    df.loc[(df['title'].str.contains('云中君|少司命|东君|国殇|大司命|湘夫人|山鬼|湘君|东皇太一|礼魂')) & (df['author'] == '屈原'), 'title'] = '楚辞·九歌·'+df.loc[(df['title'].str.contains('云中君|少司命|东君|国殇|大司命|湘夫人|山鬼|湘君|东皇太一|礼魂')) & (df['author'] == '屈原'), 'title']
    # 删除title = 九歌·九歌 东皇太一
    df.drop(df[(df['title'] == '楚辞·九歌·九歌 东皇太一')].index, inplace=True)
    df.drop(df[(df['title'] == '楚辞·九歌·楚辞 · 九歌 · 其四 · 湘夫人')].index, inplace=True)
    df.sort_values(by=['id'], inplace=True)

    # df 列的处理
    df['author'] = df['author'].apply(lambda x: keep_chinese_and_pipe(x))
    df['title'] = df['title'].apply(lambda x: x.replace('《', '').replace('》', ''))
    df['type'] = df['type'] + '|' + df['dynasty'] + '|' + df['author']  # 将朝代和作者加入type
    df['type'] = df['type'].apply(lambda x: '|'.join(part for part in set(x.split('|')) if part))
    df['topics'] = df['topics'].apply(lambda x: '|'.join(part for part in set(x.split('|')) if part))
    noise = '唐诗三百首学高初年级教师赋序'
    for i in tqdm(df.index):
        topics = df.loc[i, 'topics']
        t_li = topics.split('|')
        try:
            t_li.remove('七夕')
            t_li.append('七夕节')
        except:
            pass
        try:
            t_li.remove('七夕情人节')
            t_li.append('七夕节')
        except:
            pass
        for t in t_li:
            if set(t) & set(noise):
                t_li.remove(t)
        df.loc[i, 'topics'] = '|'.join(t_li)
    df['topics'] = df['topics'].apply(lambda x: '|'.join(part for part in set(x.split('|')) if part))  # 去重 176698

    df_new = df[df['topics'].str.len() > 0]
    df_new.replace('', np.nan, inplace=True)  # 将所有空字符串替换为NaN
    df_new.to_csv(f_v3_os + 'poems_all_v3_05.csv', index=False)  # 176459


def reset_id():
    # 汉字和诗词关联,设置id
    poemsLinkHanzi(hanzi_file=param.data_file_os + 'hanzi_all_simple.csv',
                   poems_file=param.poems_all_v3_os + 'poems_all_v3_05.csv',
                   new_poem_file=param.data_file_os + 'poems_all_v3_06.csv',
                   new_hp_file=param.hanzi_os + 'hanzi_poems_v3_06.csv')
    print('汉字和诗词关联Done!')
    df = pd.read_csv(param.data_file_os + 'poems_all_v3_06.csv')
    df.drop(columns=['Unnamed: 0'], inplace=True)
    df.to_csv(param.data_file_os + 'poems_all_v3_06.csv', index=False, encoding='utf-8')


if __name__ == '__main__':
    print('start')
    # # 基于原来的数据构建topics列，作为检索的values，存入poems_all_v3_02.csv
    # bulid_topics()
    #
    # # 将chinese-poetry-master文件夹中的json文件合并为csv文件，得到古诗类型type列
    # merge_type()
    #
    # # 初步处理golden_data，增加type列
    # add_golden_data()

    # # 进一步处理golden_data，增加type列
    # add_golden_data_part()

    # 手动处理数据
    manual_preprocess()

    # reset_id
    reset_id()

    # # check
    # df3 = pd.read_csv('/home/zsl/audrey_code/AI_Naming/dataset/poems_all_v3_03.csv', low_memory=False)  # 177938
    # df5 = pd.read_csv('/home/zsl/audrey_code/AI_Naming/dataset/poems_all_v3_05.csv', low_memory=False)  # 176721
    # df5.drop(columns=['Unnamed: 0'], inplace=True)
    # df3.drop(columns=['Unnamed: 0'], inplace=True)
    # df3_tmp = df3[['id', 'topics']]
    # df5.drop(columns=['topics'], inplace=True)
    # df5.drop(columns=['retrieval_info'], inplace=True)
    # df_new = pd.merge(df5, df3_tmp, on=['id'], how='left')  # 176721
    # df_new = df_new.fillna('')
    # df_new = df_new[df_new['topics'].str.len() > 0]
    # df_new = df_new.replace('', np.nan)  # 176384
    # df_new.to_csv('/home/zsl/audrey_code/AI_Naming/dataset/poems_all_v3_05.csv', index=False)
    # df = df.fillna('')
    # d = df[df['topics'].str.len() == 0]
    # print(df['type'].value_counts().sum())  # 176721
    # a = df['type'].value_counts()
    # df['type'] = df['type'].fillna('')
    # print(df[df['type'].str.contains('唐诗三百首')].shape[0])  # 243
    # df_1 = pd.read_csv(file_base + 'merge_data.csv', low_memory=False)
    # b1 = df[df['title'] == '大学']
    # b2 = df_1[df_1['title'] == '大学']

    # # 代码对比b1和b2具体内容，看是否有差异
    # diff_pos = find_differences(b1, b2)
    #
    # if diff_pos is not None:
    #     print(f"Difference found at position {diff_pos}: '{b1[diff_pos]}' vs '{b2[diff_pos]}'")
    # else:
    #     print("No differences found.")

    # def count_diff(str_li):  # 用于计算一个list里面所有字符串两两之间的不同字符数
    #     diff_ = []
    #     for i in range(len(str_li)):
    #         for j in range(i + 1, len(str_li)):
    #             diff_count = sum(1 for a, b in zip(str_li[i], str_li[j]) if a != b)
    #             diff_.append(diff_count)
    #     return diff_
    #     # 去除content基本一样的重复数据
    #
    #
    # df['title'] = df['title'].fillna('')
    # df['content'] = df['content'].fillna('')
    # df['c_part'] = df['content'].apply(lambda x: x.replace('|', '')[:5])
    # df['z_part'] = df['content'].apply(lambda x: x.replace('|', '')[-5:])
    # df['count'] = df['content'].apply(lambda x: len(x.replace('|', '')))
    # d_same = df[df.duplicated(subset=['c_part', 'z_part', 'count'], keep=False)]
    # d_same = d_same.sort_values(by='content')
    # grouped = d_same.groupby(['c_part', 'z_part', 'count'])
    # df_delete = pd.DataFrame(columns=d_same.columns)
    # for name, group in tqdm(grouped):
    #     c_li = group['content'].tolist()
    #     c_li = [x.replace('|', '') for x in c_li]
    #     diff_ = count_diff(c_li)
    #     tmp_li = [1 for i in range(len(diff_)) if diff_[i] > 3]
    #     if len(tmp_li) == 0:
    #         df_delete = pd.concat([df_delete, group], axis=0)
    # df_new = df[~df['id'].isin(df_delete['id'])]
