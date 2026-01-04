'''
=============================================================================================================
=============================文件和原数据解释，衍生的新数据=======================================================
=============================================================================================================
一、九歌数据集 CCPC（朝代、诗人、内容、题目、关键词；127682条古诗词，缺少名诗，如 将进酒-李白）
    1）ccpc_test_v1.0.json
    2）ccpc_train_v1.0.json
    3）ccpc_valid_v1.0.json
  --------------->
            Merge: ccpc_data.json
            Trans: ccpc_data.csv（古诗的标题、朝代、诗人、内容、关键词）

二、平行语料 parallel_corpus （包含部分解释）
    1） poetry_pairs.txt：诗句（单）+解释 11395
    2） wuyan_par.txt：诗句（单）+解释 6179
    3） transpoem_data.txt：题目+朝代+诗人+诗词（整）+解释+类别 ----------------> Trans: transpoem_data.csv（部分包含解释、topics）
                                                                                            +
三、名家名句 famous （包含有名诗句）
    1）【名诗】final_famous.txt 题目+诗人+朝代+诗词（整）+ 平行语料1，2数据-------> Trans: famous_poems.csv（得到部分含有exp、topics的【名诗】数据）
    ********************************************                            -------------------> Merge: merge_ft_poems.csv（合并含有exp、topics的古诗数据）
    *【下面的数据没有题目，暂时没用到】              *
    * 2）【北宋】bs_poems.txt  朝代+诗人+诗词（整） *
    * 3）【南宋】ns_poems.txt 朝代+诗人+诗词（整）  *
    * 4）【盛唐】st_poems.txt  朝代+诗人+诗词（整） *
    * 5）【晚唐】wt_poems.txt 朝代+诗人+诗词（整）  *
    ********************************************


四、GitHub开源数据 poems-db-master（https://github.com/yxcs/poems-db，包含部分解释和赏析）
    1）poems1.json
    2）poems2.json
    3）poems3.json
    4）poems4.json
 -------------->
         Merge: poems_db_fin.csv（部分包含 exp和 imp）


五、 爬取的数据 web_data（包含解释、赏析、关键词、关键词释义/意译）
    1）gushiju_poem.csv 古诗句网站，题目+诗人+朝代+诗词（整）+解释+赏析
    2）baidu_web_poems.csv 百度汉语网站，题目+诗人+朝代+诗词（整）+解释+赏析+链接

=============================================================================================================
=============================以上原数据均处理为“标准格式”（合并的数据注意：去重）=====================================
=============================================================================================================
标准格式：
"title": xx（题目）,
"dynasty": xx（朝代）,
"author": xx（诗人）,
"content": xx（内容）,
"explain": xx（翻译，暂定）,
"implication": xx（古诗的意译⭐）,
"tags":{
    "type": xx（类别标签,暂定），
    "event": xx（事件）,
    "event_time": xx（时间）,
    "event_location": xx（地点）,
    "event_holiday": xx（节日）,
    "event_season": xx（季节）,
    "event_weather":xx（天气）,
    "emotion": xx（情感）,
     "word_meanings": xx（隐喻⭐）
    }
=============================================================================================================
'''
import pandas as pd
from utils.base_param import BaseParam
import re
import json
import os
from copy import deepcopy
from tqdm import tqdm
from Levenshtein import distance
import numpy as np

params = BaseParam()
f_os_base = params.data_file_os
f_os_famous = params.famous_os
f_os_pdm = params.pdm_os
f_os_ccpc = params.ccpc_os
f_os_parallel = params.pc_os
f_os_web = params.web_data_os
f_os_pc = params.pc_os
f_os_v1 = params.poems_all_os


def get_tags(topics=None, event=None, event_time=None, event_location=None, event_holiday=None, event_season=None,
             event_weather=None, sentiment=None, metaphor=None):
    tags_ = {"topics": topics,
             "event": event,
             "event_time": event_time,
             "event_location": event_location,
             "event_holiday": event_holiday,
             "event_season": event_season,
             "event_weather": event_weather,
             "sentiment": sentiment,
             "metaphor": metaphor}
    return tags_


def duplicate_poems_df(df_n):  # 去重
    # 筛去content为nan的数据
    df_ = df_n[df_n['content'].notnull()]
    # 按照implication和explain排序
    df_s = df_.sort_values(by=['implication', 'explain', 'key_words'])
    # 对合并后的结果去除重复数据，并保留第一次出现的重复实例
    df_final1 = df_s.drop_duplicates(subset=['content'], keep='first')
    # df_final2 = df_final1.drop_duplicates(subset=['author', 'part_content'], keep='first')
    return df_final1


def trans_au(name):  # 对作者名做处理
    new_name = name
    if new_name in ['无名氏', '不详']:
        new_name = '佚名'
    if '(' in new_name:
        new_name = new_name[:new_name.index('(')]
    return new_name


# def trans_ti(title):  # 对标题做处理
#     new_t = title
#     if '/' in new_t:
#         new_t = new_t[:new_t.index('/')]
#     return new_t

def trans_dy(dy):  # 直接将str类型的朝代进行转换
    dy_new = dy
    if dy in ['唐', '唐朝', '初唐', '盛唐', '中唐', '晚唐', 'Tang', 'tang']:
        dy_new = '唐代'
    elif dy in ['宋', '宋朝', '北宋', '南宋', '宋末', 'Song', 'song']:
        dy_new = '宋代'
    elif dy in ['元', '元朝', '元末', 'Yuan', 'yuan']:
        dy_new = '元代'
    elif dy in ['明', '明朝', '明初', '明中叶', '明末', 'Ming', 'ming']:
        dy_new = '明代'
    elif dy in ['清', '清朝', '清初', '清中叶', '清末', 'Qing', 'qing']:
        dy_new = '清代'
    elif dy in ['金代', '金', 'Jin', 'jin']:
        dy_new = '金朝'
    elif dy in ['辽', 'Liao']:
        dy_new = '辽代'
    elif dy in ['隋', 'Sui']:
        dy_new = '隋代'
    elif dy == 'NanBei':
        dy_new = '南北朝'
    return dy_new


def trans_df_dynasty(df_):  # 给df，将df中的朝代进行转换
    for i in tqdm(df_.index):
        dynasty = df_.loc[i, 'dynasty']
        df_.loc[i, 'dynasty'] = trans_dy(dynasty)
    return df_


def trans_df(df_):
    for i in tqdm(df_.index):
        df_.loc[i, 'dynasty'] = trans_dy(df_.loc[i, 'dynasty'])
        df_.loc[i, 'author'] = trans_au(df_.loc[i, 'author'])
        df_.loc[i, 'content'] = trans_con(df_.loc[i, 'content'])
    return df_


def keep_chinese(text):
    # 使用正则表达式匹配中文字符和竖线 |
    pattern = re.compile(r'[^\u4e00-\u9fa5]')
    result = pattern.sub('', text)
    return result

def keep_chinese_and_pipe(text):
    # 使用正则表达式匹配中文字符和竖线 |
    pattern = re.compile(r'[^\u4e00-\u9fa5|]')
    result = pattern.sub('', text)
    return result


def trans_con(cont):
    clean_c1 = re.sub(r'[(（][^)）]*[)）]', '', cont)  # 去除括号及括号内的内容
    clean_c2 = re.sub(r'[，。？！；：、]', '|', clean_c1)  # 将标点符号替换为竖线
    clean_c2 = re.sub(r'<[^>]+>', '', clean_c2)  # 去除html标签
    new_cont = (clean_c2.replace('\n', '').replace('\r', '')
                .replace(' ', '').replace('||', '|'))
    new_cont = keep_chinese_and_pipe(new_cont)  # 仅保留中文和|字符
    return new_cont


class Preprocess_ccpc_data:
    def merge_ccpc_data(self, ori_file_li, new_file, write_flag=False):
        '''获取CCPC文件中ccpc_test_v1.0.json,ccpc_train_v1.0.json,ccpc_valid_v1.0.json中古诗词数据，整合并写入ccpc_data.json文件中'''
        ccpc_data = []
        for file in [f_os_ccpc + f for f in ori_file_li]:
            with open(file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        data = line.strip()
                        ccpc_data.append(data)
                    except json.JSONDecodeError as e:
                        print(f"Error decoding JSON: {e}")

        # 汇总后为ccpc_data，去重得到ccpc_data_new
        s = set(ccpc_data)
        ccpc_data_new = list(s)
        if write_flag:
            with open(f_os_ccpc + new_file, 'w+', encoding='utf-8') as f:
                for i in ccpc_data_new:
                    json.dump(i, f, ensure_ascii=False)
                    f.write('\n')
        return ccpc_data_new

    def trans_ccpc_data(self, ori_file, new_file):
        # 列
        title = None
        dynasty = None
        author = None
        content = None
        explain = None
        implication = None
        key_words = None
        key_words_imp = None
        tags = get_tags()
        # 读取文件内容
        with open(f_os_ccpc + ori_file, 'r', encoding='utf-8') as f:
            data = []
            for line in f:
                con = json.loads(eval(line))
                title = con['title']
                dynasty = trans_dy(con['dynasty'])
                author = trans_au(con['author'])
                content = trans_con(con['content'])
                key_words = con['keywords'].replace(' ', '|')
                data.append([title, dynasty, author, content, explain, implication, key_words, key_words_imp, tags])
            df = pd.DataFrame(data,
                              columns=['title', 'dynasty', 'author', 'content', 'explain', 'implication', 'key_words',
                                       'key_words_imp', 'tags'])
            df_ = df.drop_duplicates(subset=['content'])  # 去重
            df_.to_csv(f_os_ccpc + new_file, index=False, encoding='utf-8')


class Preprocess_poems_famous_parallel:
    def trans_transpoem_data(self, ori_file, new_file):
        # 读取文件内容
        with open(f_os_parallel + ori_file, "r", encoding="utf-8") as file:
            file_content = file.read()
        # 列
        title = None
        dynasty = None
        author = None
        content = None
        explain = None
        implication = None
        key_words = None
        key_words_imp = None
        tags = get_tags()
        # 将文件内容按行分割
        lines = file_content.strip().split("\n")
        data = []
        one_poem_li = []
        for i in tqdm(range(len(lines))):
            # 0: title,1:dynasty,2:author,3:content,4:explian,5:topics
            if i == 0:
                title = lines[i].strip()
                one_poem_li.append(title)
            elif i % 6 == 1:
                dynasty = trans_dy(lines[i].strip())
                one_poem_li.append(dynasty)
            elif i % 6 == 2:
                author = trans_au(lines[i].strip())
                one_poem_li.append(author)
            elif i % 6 == 3:
                content = trans_con(lines[i].strip())
                try:
                    content = content[:content.rindex('|')]  # 去掉一些有不完整内容的古诗
                except ValueError:
                    continue
                one_poem_li.append(content)
            elif i % 6 == 4:
                lin_con = lines[i].strip()
                one_poem_li.append(lin_con.replace('|', ''))
            elif i % 6 == 5:
                tags["topics"] = lines[i]
                one_poem_li.extend([implication, key_words, key_words_imp])
                one_poem_li.append(deepcopy(tags))
            elif i != 0 and i % 6 == 0:
                data.append(one_poem_li)
                one_poem_li = []  # 清空
                one_poem_li.append(lines[i])

        # 转换为DataFrame
        df = pd.DataFrame(data, columns=['title', 'dynasty', 'author',
                                         'content', 'explain', 'implication',
                                         'key_words', 'key_words_imp', 'tags'])

        # df = trans_df_dynasty(df)  # 整体朝代转换
        df_new = duplicate_poems_df(df)  # 去重，去除重复的数据，优先保留含有explain的行
        df_new.to_csv(f_os_parallel + new_file, index=False)  # 将DataFrame写入CSV文件

    def trans_fam_para_data(self, famous_file, p_li, new_file):
        def read_poetry_pairs(file_name):  # 读取parallel_corpus/poetry_pairs.txt，wuyan_par.txt
            db_dic = {}
            with open(f_os_pc + file_name, 'r', encoding='utf-8') as f:
                for line in f:
                    # print(line)
                    l = line.split('|||')
                    content = l[0].replace('，', '|').replace('。', '')
                    explain = l[1].strip()
                    db_dic[content] = explain
            return db_dic

        def search_db_dic(content, db_dic):  # 在db_dic中查找content
            num = int((content.count('|') + 1) / 2)
            c_li = content.split('|')
            explain_part = ''
            for j in range(num):
                content_part = '|'.join(c_li[j * 2:j * 2 + 2])
                if content_part in db_dic:
                    explain_part += db_dic[content_part]
                else:
                    explain_part = ''
            if explain_part:
                return explain_part
            else:
                return None

        d1 = read_poetry_pairs(p_li[0])
        d2 = read_poetry_pairs(p_li[1])
        db_dic = {}
        db_dic.update(d1)
        db_dic.update(d2)

        # files = ['bs_poems.txt', 'ns_poems.txt', 'st_poems.txt', 'wt_poems.txt']
        data = []

        with open(f_os_famous + famous_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        title = None
        dynasty = None
        author = None
        content = None
        explain = None
        implication = None
        key_words = None
        key_words_imp = None
        tags = get_tags()
        content_li = []
        cnt = 0
        for line in lines:
            if re.match('Title', line):
                if cnt > 0:
                    content = '|'.join(content_li)
                    content = trans_con(content)
                    explain = search_db_dic(content, db_dic)
                    data.append([title, dynasty, author, content, explain, implication, key_words, key_words_imp, tags])
                _, title, author, dynasty = line.strip().split(' ')
                author = trans_au(author)
                dynasty = trans_dy(dynasty)
                cnt += 1
                content_li = []
            else:
                content_li.append(line.strip())

        df = pd.DataFrame(data, columns=['title', 'dynasty', 'author',
                                         'content', 'explain', 'implication', 'key_words',
                                         'key_words_imp', 'tags'])

        # 去重
        df_new = duplicate_poems_df(df)

        df_new.to_csv(f_os_famous + new_file, index=False)

    def merge_fam_para_data(self, td_file, fp_file, new_file):
        df = pd.read_csv(f_os_pc + td_file)  # 将两个数据集合并，并去重
        # print('transpoem_data.csv的古诗数量：', df.shape[0])
        df_f = pd.read_csv(f_os_famous + fp_file)
        # print('famous_poems.csv的古诗数量为：', df_f.shape[0])
        df_mer = pd.concat([df, df_f], ignore_index=True)  # 合并
        # print('两者合并后古诗数量为；', df_mer.shape[0])
        df_mer_ = duplicate_poems_df(df_mer)  # 去重
        # print("去重后古诗数量为：", df_mer.shape[0])
        df_mer_.to_csv(f_os_base + new_file, index=False)


class Preprocess_poems_db:

    def merge_poems_db(self, ori_file_li, new_f):
        df = pd.DataFrame(
            columns=['title', 'dynasty', 'author',
                     'content', 'explain', 'implication', 'key_words', 'key_words_imp',
                     'tags'])
        data = []
        df.to_csv(f_os_pdm + new_f, index=False, encoding='utf-8')
        for file in [f_os_pdm + file for file in ori_file_li]:
            print(file)
            with open(os.path.join(file), 'r') as f:
                for line in f:
                    con = json.loads(line)
                    if con['content'] == [] or con['content'] == [''] or con['content'] == None:
                        continue
                    c = '|'.join(con['content']).strip()
                    if c == '' or c == None:
                        continue
                    content = trans_con(c)
                    try:
                        content = content[:content.rindex('|')]  # 去掉一些有不完整内容的古诗
                    except ValueError:
                        print(content)
                        continue
                    title = con['name']
                    dynasty = trans_dy(con['dynasty'])
                    author = trans_au(con['author'])
                    explain = ''.join(con['translate'])
                    implication = ''.join(con['appreciation']).replace('\u3000\u3000', '')
                    key_imp = ''.join(con['notes'])
                    topics = ';'.join(con['tags'])
                    tags = get_tags(topics=topics)
                    key_words = None
                    data.append([title, dynasty, author, content, explain, implication, key_words, key_imp, tags])
            df_ = pd.DataFrame(data,
                               columns=['title', 'dynasty', 'author', 'content', 'explain', 'implication', 'key_words',
                                        'key_words_imp', 'tags'])
            df_ = duplicate_poems_df(df_)  # 去重
            df_.to_csv(f_os_pdm + new_f, mode='a', index=False, header=False, encoding='utf-8')


class Preprocess_web_data:

    def trans_gushiju_data(self, f, f_new):
        print('trans_gushiju_data')
        # f = 'gushiju_poem.csv'
        df = pd.read_csv(f_os_web + f, encoding='utf-8', dtype=object).astype(str)
        df_new = duplicate_poems_df(df)  # 去掉content为nan的，去重,
        df_copy = deepcopy(df_new)
        for i in df_new.index:
            c = df_new.loc[i, 'content']
            if c == '' or c == None:
                continue
            content = trans_con(c)
            try:
                content = content[:content.rindex('|')]  # 去掉一些没有分割的古诗 比如：别来长忆西楼事结遍兰襟遗恨重寻弦断相如绿绮琴何时一枕逍遥夜细话初心若问如今也似当时著意深
            except ValueError:
                print(content)
                continue
            df_copy.loc[i, 'author'] = trans_au(df_new.loc[i, 'author'].strip())
            df_copy.loc[i, 'dynasty'] = trans_dy(df_new.loc[i, 'dynasty'].strip())
            df_copy.loc[i, 'content'] = content
            df_copy.loc[i, 'explain'] = df_new.loc[i, 'explain'].strip()
            df_copy.loc[i, 'implication'] = df_new.loc[i, 'implication'].strip()
        df_n = duplicate_poems_df(df_copy)  # 去重
        df_n = trans_df_dynasty(df_n)  # 整体朝代转换
        df_n.to_csv(f_os_web + f_new, index=False, encoding='utf-8')

    def merge_gushiju_db_data(self, f_gushiju, f_db, new_f):
        print('merge_gushiju_db_data')
        df1 = pd.read_csv(f_gushiju, low_memory=False, encoding='utf-8')
        df1 = df1[df1['implication'].notnull()]  # 取出implication列含有内容的行

        df2 = pd.read_csv(f_db, low_memory=False, encoding='utf-8')
        df2 = df2[df2['implication'].notnull()]  # 取出implication列含有内容的行

        df_new = pd.concat([df1, df2], ignore_index=True)  # 合并两个数据集
        df_new = duplicate_poems_df(df_new)  # 去重
        df_new.to_csv(f_os_base + new_f, index=False)  # 写入新文件

    def trans_to_json(self):
        # 读取数据
        data = pd.read_csv(f_os_base + 'poems_GG_only_imp.csv')
        with open(f_os_base + 'poems_GG_only_imp.json', 'w+', encoding='utf-8') as f:
            for i in data.index:
                con = {}
                con['title'] = data.loc[i, 'title']
                con['dynasty'] = data.loc[i, 'dynasty']
                con['author'] = data.loc[i, 'author']
                con['content'] = data.loc[i, 'content']
                con['explain'] = data.loc[i, 'explain']
                con['implication'] = data.loc[i, 'implication']
                json.dump(con, f, ensure_ascii=False)
                f.write('\n')

    def trans_xcz_data(self, f, f_new):
        data = []
        # cnt = 0 # 测试使用
        with open(f, 'r') as f:
            poems_li = json.load(f)
            for p in tqdm(poems_li):
                # 测试使用
                # cnt += 1
                # if cnt > 10:
                #     break
                title = None
                dynasty = None
                author = None
                content = None
                explain = None
                implication = None
                key_words = None
                key_words_imp = None
                tags = get_tags()

                c = p.get('Content')
                if c == '' or c == None:
                    continue
                content = trans_con(c)
                try:
                    content = content[:content.rindex('|')]
                except ValueError:
                    print(content)
                    continue

                title = p.get('Title')
                dynasty = trans_dy(p.get('Dynasty'))
                author = trans_au(p.get('Author'))
                explain = p.get('Translation')

                if p.get('Comment') and p.get('Intro'):
                    implication = p.get('Comment') + p.get('Intro')
                elif p.get('Comment') != None:
                    implication = p.get('Comment')
                elif p.get('Intro') != None:
                    implication = p.get('Intro')
                else:
                    implication = None

                try:
                    ki = p.get('Annotation').replace('\r', '').split('\n')
                except:
                    pass

                try:
                    key_words_imp = {}
                    for i in ki:
                        key, vla = i.split('：')
                        key_words_imp[key] = vla
                    key_words = '|'.join(list(key_words_imp.keys()))
                except:
                    pass

                data.append([title, dynasty, author, content, explain, implication, key_words, key_words_imp, tags])
        df = pd.DataFrame(data, columns=['title', 'dynasty', 'author', 'content', 'explain', 'implication', 'key_words',
                                         'key_words_imp', 'tags'])
        df_new = duplicate_poems_df(df)  # 去重
        df_new.to_csv(f_new, index=False)


# def duplicate_poems_df(df):  # 去重
#     # 筛选出含有赏析、解释、不含赏析、不含解释的行
#     df_with_imp = df[df['implication'].notnull()]
#     df_with_exp = df[df['explain'].notnull()]
#     df_without_imp = df[df['implication'].isnull()]
#     df_without_exp = df[df['explain'].isnull()]
#     # 合并四个结果
#     df_final = pd.concat([df_with_imp, df_with_exp, df_without_imp, df_without_exp])
#     # 对合并后的结果去除重复数据，并保留第一次出现的重复实例
#     df_final = df_final.drop_duplicates(subset=['content'], keep='first')
#     return df_final


def merge_all(f_li, new_f):
    # 7、Merge: ccpc_data.csv + merge_ft_poems.csv + gushiju_poem.csv + poems_db_fin.csv'
    #             '----> poems_Base.csv'
    # 9、Merge&Trans: poems_Base.csv + poems_base_baidu.csv -----> poems_base_baidu_new.csv'
    # 10、Merge: poems_base_baidu_new.csv + poem_xcz.csv  -----> poems_BaW.csv
    # 合并所有数据
    df_li = []
    for f in f_li:
        if f.split('/')[-1] != 'poems_base_baidu.csv':  # 只要不是百度的数据
            df_i = pd.read_csv(f, low_memory=False, encoding='utf-8')
        else:
            print('trans: poems_base_baidu.csv')
            df = pd.read_csv(f_os_web + f_poems_base_baidu, encoding='utf-8')
            df_un = duplicate_poems_df(df)  # 去重
            df_i = trans_df(df_un)
        df_li.append(df_i)
    df_all = pd.concat(df_li, ignore_index=True)
    df_all = duplicate_poems_df(df_all)  # 去重
    df_all.to_csv(new_f, index=False)


def merge_souyun(f_poems, f_souyun, f_new):
    df_poems = pd.read_csv(f_poems, low_memory=False, encoding='utf-8')
    df_souyun = pd.read_csv(f_souyun, encoding='utf-8')
    df_sy_new = df_souyun[df_souyun['word_meanings'].notna()]
    # 写入{keyword:meaning}的字典dic_wm
    dic_wm = {}
    for i in df_sy_new.index:
        if df_sy_new.loc[i, 'word'] in dic_wm.keys():
            continue
        else:
            dic_wm[df_sy_new.loc[i, 'word']] = df_sy_new.loc[i, 'word_meanings']
    # 遍历df_poems和dic_wm，将dic_wm中的内容合并到df_poems中
    for i in tqdm(df_poems.index):
        kw_li = []  # 新关键词列表
        kw_imp = {}  # 新关键词及其意义
        try:
            ex_kws_li = df_poems.loc[i, 'key_words'].split('|')  # 已有的关键词列表
        except:
            ex_kws_li = []
        for j in dic_wm.keys():
            if j in str(df_poems.loc[i, 'content']) and j not in ex_kws_li:  # 如果关键词在诗词中且不在已有关键词中
                kw_li.append(j)
                kw_imp.update({j: dic_wm[j]})

        kw = '|'.join(kw_li)
        if kw != '':
            if pd.isnull(df_poems.loc[i, 'key_words']):  # 如果key_words为空，则直接将kw赋值给key_words
                df_poems.loc[i, 'key_words'] = kw
                df_poems.loc[i, 'key_words_imp'] = str(kw_imp)
            else:
                # 如果key_words不为空，则将kw添加到key_words后面
                df_poems.loc[i, 'key_words'] = df_poems.loc[i, 'key_words'] + '|' + kw
                # 将kw_imp添加到key_words_imp中
                ext_imp = df_poems.loc[i, 'key_words_imp']
                if pd.isna(ext_imp) == False and ext_imp != 'nan':  # 如果ext_imp不为空
                    # print(ext_imp)
                    ext_imp = eval(ext_imp)
                    ext_imp.update(kw_imp)
                else:
                    ext_imp = kw_imp  # 如果ext_imp为空，则直接赋值
                    # print(ext_imp)
                df_poems.loc[i, 'key_words_imp'] = str(ext_imp)

    # 结果写入poems_all_new.csv
    df_poems.to_csv(f_new, index=False, encoding='utf-8')


def find_same_poems(texts, indexs):
    n = len(texts)
    same_poems = []
    # 计算字符串之间的 Levenshtein 距离
    for i in range(n):
        for j in range(i + 1, n):
            if distance(texts[i], texts[j]) <= 4:  # 如果两个字符串最多有4个字符不同
                # 获取被认为是同一首诗的索引
                index1 = indexs[i]
                index2 = indexs[j]
                # 将相同诗的索引合并为一个列表
                same_poems.append((index1, index2))
                # print(texts[i],'\n', texts[j])

    return same_poems


def drop_same_poems(f_Baw):
    df = pd.read_csv(f_Baw, encoding='utf-8', low_memory=False)
    print('去重前诗数量为：', df.shape[0])
    # step1：先将同一作者的所有诗筛选出来
    at_d = df.groupby('author')['title'].apply(list).to_dict()
    # step2：保留同一个作者存在大于2个标题且有重复的标题的诗
    fil_at_d = {author: titles for author, titles in at_d.items() if len(titles) > 2}
    # step3：遍历这些诗fil_at_d，去重标题 (注意：有可能存在不同标题但是是同一首诗，比如：《送别.其一》、《送别》）
    for a, ts in tqdm(fil_at_d.items()):
        fil_at_d[a] = list(set(ts))  # 去重
    sorted_keys = sorted(fil_at_d.keys())
    # step4：遍历fil_at_d，去df中寻找到对应行（可能存在多个），然后判断这些诗是否为同一首诗（同作者，同标题，诗的内容最多有4字不同，则认为是同一首诗）；如果是，则删除
    for a in tqdm(sorted_keys):
        ts = fil_at_d[a]
        for t in tqdm(ts):
            df_m = df[(df['author'] == a) & (df['title'] == t)]  # 根据作者和标题在 DataFrame 中查找匹配的行
            same_poems = find_same_poems(df_m['content'].values, df_m.index.values)  # 相同诗的索引列表
            # if len(same_poems) > 1:
            #     print(a, t)
            # 删除的时候优先删除不含explian、implication的行,把行号保存下来
            for i, j in same_poems:
                if df_m.loc[i, 'title'] == None or df_m.loc[j, 'title'] == None:  # 如果其中一个title为None，则跳过
                    continue
                elif df_m.loc[i, 'explain'] is None and df_m.loc[i, 'implication'] is None and df_m.loc[
                    i, 'key_words'] is None:  # 如果都不含explain和implication,则删除i
                    df.loc[i, 'title'] = None
                else:
                    df.loc[j, 'title'] = None
    df.dropna(subset=['title'], inplace=True)  # 删除title为None的行
    df.to_csv(f_Baw, index=False, encoding='utf-8')  # 结果写入文件
    print('去重后的诗数量为：', df.shape[0])


def drop_noise():
    df = pd.read_csv(f_os_v1 + 'poems_All.csv', encoding='utf-8', low_memory=False)
    df.fillna('', inplace=True)
    # 对explain做处理
    for i in tqdm(df.index):
        exp = df.loc[i, 'explain']
        exp = exp.replace(' ', '')
        if exp.count('中文译文：') > 1 or exp.count('译文：') > 1:
            df.loc[i, 'explain'] = ''
            continue
        elif exp.count('中文译文：') == 1:
            exp = exp[exp.index('中文译文：') + 5:]
        elif exp.count('译文:') == 1:
            exp = exp[exp.index('译文:') + 3:]

        exp = re.sub(r'[\n\u3000\r\t]', '', exp)
        pattern = re.compile(r'[^\u4e00-\u9fa5]')
        exp_text = pattern.sub('', exp)
        con_text = pattern.sub('', df.loc[i, 'content'])
        if (len(exp) < 10) or (con_text in exp_text):
            exp = ''
        df.loc[i, 'explain'] = exp

    # # 测试用
    # df.replace('', np.nan, inplace=True)
    # df_tt = df[df['explain'].str.len() < 10]

    # 对implication做处理
    prefixes = ('诗意：', '鉴赏：', '诗意和鉴赏：', '诗意和赏析：', '赏析：')
    # # 查看df的implication中是否有长度小于10的
    # df_imp = df[(df['implication'].str.len() > 0) & (df['implication'].str.len() < 100)]['implication']
    # 遍历，只要是以head开头的，就进行处理
    for i in tqdm(df.index):
        imp = df.loc[i, 'implication']
        if imp.startswith(prefixes):
            imp = imp[imp.index('：') + 1:]
        imp = re.sub(r'[\n\u3000\r\t]', '', imp)
        if len(imp) < 50:  # 长度要最后判断
            df.loc[i, 'implication'] = ''
            continue
        df.loc[i, 'implication'] = imp

    # 存储
    df.replace('', np.nan, inplace=True)
    df.isna().any()  # 检查是否有空值
    tags = get_tags()
    df['tags'] = df['tags'].fillna(str(tags))  # 将tags中的nan替换成tags
    df.to_csv(f_os_v1 + 'poems_All_clean.csv', index=False, encoding='utf-8')
    # 将df中explain不为nan的行筛选出来
    df_exp = df[df['explain'].notna()]  # 152813
    df_imp = df[df['implication'].notna()]  # 173152
    df_exp_imp = df_exp[df_exp['implication'].notna()]  # 82920
    # df_exp去除key_words_imp和key_words列
    df_exp.drop('key_words_imp', axis=1, inplace=True)
    df_exp.drop('key_words', axis=1, inplace=True)
    # df_exp去除key_words_imp和key_words列
    df_exp.drop('key_words_imp', axis=1, inplace=True)
    df_imp.drop('key_words', axis=1, inplace=True)
    # 存储
    df_exp.to_csv(f_os_v1 + 'poems_All_clean_exp.csv', index=False, encoding='utf-8')  # 原数据中explain不为空的
    df_imp.to_csv(f_os_v1 + 'poems_All_clean_imp.csv', index=False, encoding='utf-8')  # 原数据中implication不为空的
    df_exp_imp.to_csv(f_os_v1 + 'poems_All_clean_exp_imp.csv', index=False,
                      encoding='utf-8')  # 原数据中explain和implication都不为空的

    # # 测试用
    # df_tt = df_exp_imp[df_exp_imp['explain'].str.len() < 10]


if __name__ == '__main__':
    # 1、Trans  ccpc_data (base 九歌)
    f_ccpc_1 = 'ccpc_test_v1.0.json'
    f_ccpc_2 = 'ccpc_train_v1.0.json'
    f_ccpc_3 = 'ccpc_valid_v1.0.json'
    f_ccpc_data_j = 'ccpc_data.json'
    f_ccpc_data_c = 'ccpc_data.csv'

    # 2、Trans  transpoem_data (base 平行语料)
    f_parallel_pairs = 'poetry_pairs.txt'
    f_parallel_wuyan = 'wuyan_par.txt'
    f_parallel_transpoem_t = 'transpoem_data.txt'
    f_parallel_transpoem_c = 'transpoem_data.csv'

    # 3、Merge&Trans famous_poems (base 名人名句)
    f_famous_t = 'final_famous.txt'
    f_famous_c = 'famous_poems.csv'

    # 4、Merge poems_TF (合并2、3)
    f_merge_tf = 'poems_TF.csv'

    # 5、Merge&Trans poems_db_fin (GitHub，古诗文)
    f_poems_db_li = ['poems1.json', 'poems2.json', 'poems3.json', 'poems4.json']
    f_merge_poems_db = 'poems_db_fin.csv'

    # 6、Trans&Merge poems_GG_only_imp (合并5和古诗句网数据中含有鉴赏的数据)
    f_gushiju = 'gushiju_poem0427.csv'  # 数据
    f_gushiju_new = 'gushiju_poem_new0427.csv'  # 预处理格式和去重后的数据
    f_merge_gg_imp = 'poems_GG_only_imp.csv'

    # 7、Merge  poems_Base (合并1、4、5、6)
    f_poems_Base = 'poems_Base.csv'  # 合并的数据

    # 8、Trans xcz (GitHub，西窗烛)
    f_xcz = params.web_data_os + 'works.json'  # GitHub上由作者从西窗烛爬取得到的80w条数据
    f_xcz_new = params.f_xcz_new

    # 9、Trans poems_base_baidu_new (poems_base之后爬取百度的解释,即运行get_baidu_data.py，得到poems_base_baidu)
    f_poems_base_baidu = 'poems_base_baidu.csv'
    f_poems_base_baidu_new = 'poems_base_baidu_new.csv'

    # 10、Merge poems_BaW (合并8、9，所有base+web数据)
    f_poems_BaW = 'poems_BaW.csv'

    # 11、Merge poems_All (10增加搜韵tags）
    f_souyun_word = 'souyun_words.csv'
    f_poems_All = 'poems_All.csv'
    f_poems_All_kw = 'poems_All_kw.csv'

    flag_li = [0] * 13
    flag_trans_gush = 0

    # flag_li[0] = 1  # 1、Trans       ccpc_data (base 九歌)
    # flag_li[1] = 1  # 2、Trans       transpoem_data (base 平行语料)
    # flag_li[2] = 1  # 3、Merge&Trans famous_poems (base 名人名句)
    # flag_li[3] = 1  # 4、Merge       poems_TF (合并2、3)
    # flag_li[4] = 1  # 5、Merge&Trans poems_db_fin (GitHub，古诗文)
    #
    # flag_li[5] = 1  # 6、Trans&Merge poems_GG_only_imp (合并5和古诗句网数据中含有鉴赏的数据)
    # flag_trans_gush = 1  # 6、step1 Trans gushiju_poems_new.csv
    # flag_li[6] = 1  # 7、Merge       poems_Base (合并1、4、5、6)
    # flag_li[7] = 1  # 8、Trans       xcz (GitHub，西窗烛)
    # # flag_li[8] = 1  # 9、Trans&Merge poems_base_baidu_new (poems_base之后爬取百度的解释,即运行get_baidu_data.py，得到poems_base_baidu)
    # flag_li[9] = 1  # 10、Merge      poems_BaW (合并8、9，所有base+web数据)
    # flag_li[10] = 1  # 11、处理重复数据
    # flag_li[11] = 1  # 12、Merge     poems_All (10增加搜韵keywords和tags）
    # flag_li[12] = 1  # 13、去除噪声数据

    if flag_li[0]:
        print('1、Trans：ccpc_*_v1.0.json ------> ccpc_data.json------> ccpc_data.csv ')
        pcd = Preprocess_ccpc_data()
        pcd.merge_ccpc_data([f_ccpc_1, f_ccpc_2, f_ccpc_3], f_ccpc_data_j, True)
        pcd.trans_ccpc_data(f_ccpc_data_j, f_ccpc_data_c)
        print('==========1、ok=============')

    if flag_li[1]:
        print('2、Trans：transpoem_data.txt -----> transpoem_data.csv ')
        ppfp = Preprocess_poems_famous_parallel()
        ppfp.trans_transpoem_data(f_parallel_transpoem_t, f_parallel_transpoem_c)
        print('==========2、ok=============')

    if flag_li[2]:
        print('3、Merge & Trans：final_famous.txt+ poetry_pairs.txt、wuyan_par.txt----> famous_poems.csv')
        ppfp = Preprocess_poems_famous_parallel()
        ppfp.trans_fam_para_data(f_famous_t, [f_parallel_pairs, f_parallel_wuyan], f_famous_c)
        print('==========3、ok=============')

    if flag_li[3]:
        print('4、Merge：famous_poems.csv + transpoem_data.csv------> poems_TF.csv')
        ppfp = Preprocess_poems_famous_parallel()
        ppfp.merge_fam_para_data(f_parallel_transpoem_c, f_famous_c, f_merge_tf)
        print('==========4、ok=============')

    if flag_li[4]:
        print('5、Merge & Trans：poems-db-master/poems1.json、poems2.json、poems3.json、poems4.json（GitHub） '
              '------> poems_db_fin.csv')
        ppd = Preprocess_poems_db()
        ppd.merge_poems_db(f_poems_db_li, f_merge_poems_db)
        print('==========5、ok=============')

    if flag_li[5]:
        print('6、Trans & Merge：poems_db_fin.csv + gushiju_poem.csv -----> poems_GG_only_imp.csv ')
        pwd = Preprocess_web_data()
        if flag_trans_gush:
            df_raw = pd.read_csv(f_os_web + 'gushiju_poem_raw0427.csv', encoding='utf-8', low_memory=False)
            df_raw = df_raw.drop(['tags'], axis=1)  # 去除df_raw的tags列（原来爬取的时候tags的设计有问题）
            df_raw['tags'] = str(get_tags())  # 加上tags列
            df_raw.to_csv(f_os_web + f_gushiju, index=False, encoding='utf-8')  # 存入gushiju_poem.csv
            pwd.trans_gushiju_data(f_gushiju, f_gushiju_new)  # 去重，格式化author、dynasty、content
        # 合并数据
        pwd.merge_gushiju_db_data(f_os_web + f_gushiju_new, f_os_pdm + f_merge_poems_db, f_merge_gg_imp)
        pwd.trans_to_json()
        print('==========6、ok=============')

    if flag_li[6]:
        print(
            '7、Merge: ccpc_data.csv + merge_ft_poems.csv + gushiju_poem.csv + poems_db_fin.csv'
            '----> poems_Base.csv')
        file_os_li = [f_os_ccpc + f_ccpc_data_c, f_os_base + f_merge_tf, f_os_web + f_gushiju_new,
                      f_os_pdm + f_merge_poems_db]
        merge_all(file_os_li, f_os_base + f_poems_Base)
        print('==========7、ok=============')

    if flag_li[7]:
        print('8、Trans: works.json -----> poem_xcz.csv')
        pwd = Preprocess_web_data()
        pwd.trans_xcz_data(f_xcz, f_xcz_new)
        print('==========8、ok=============')

    # if flag_li[8]:
    #     print('9、Merge&Trans: poems_Base.csv + poems_base_baidu.csv -----> poems_base_baidu_new.csv')
    #     merge_all([f_os_base + f_poems_Base, f_os_web + f_poems_base_baidu], f_os_base + f_poems_base_baidu_new)
    #     print('==========9、ok=============')

    if flag_li[9]:
        print('10、Merge: poems_Base.csv + poem_xcz.csv  -----> poems_BaW.csv')
        merge_all([f_os_base + f_poems_Base, f_xcz_new],
                  f_os_base + f_poems_BaW)  # poems_BaW: poems_base_add_web
        df = pd.read_csv(f_os_base + f_poems_BaW, encoding='utf-8', low_memory=False)
        trans_df_dynasty(df).to_csv(f_os_base + f_poems_Base, encoding='utf-8', index=False)
        print('==========10、ok=============')

    if flag_li[10]:
        print('11、处理poems_BaW的重复数据，还是保存为BaW')
        drop_same_poems(f_os_base + f_poems_BaW)
        print('==========11、ok=============')

    if flag_li[11]:
        print('12、Merge: poems_BaW.csv + souyun_words.csv -----> poems_All.csv')
        step1 = 0
        step2 = 1
        if step1:
            print('merge keywords -> poems_ALL_kw')
            merge_souyun(f_os_base + f_poems_BaW, f_os_web + f_souyun_word, f_os_v1 + f_poems_All_kw)
        if step2:
            print('merge tags -> poems_ALL')
            df_tags = pd.read_csv(f_os_web + 'poems_BaW_tags.csv', encoding='utf-8',
                                  low_memory=False)  # 基于poems_BaW爬取的poems_All_tags
            df_all = pd.read_csv(f_os_v1 + f_poems_All_kw, encoding='utf-8', low_memory=False)
            df_all_without_tags = df_all.drop(['tags'], axis=1)  # 把原数据的tags列去掉
            merged_df = pd.merge(df_all_without_tags, df_tags,
                                 on=['title', 'dynasty', 'author', 'content', 'explain', 'implication', 'key_words',
                                     'key_words_imp'], how='left')  # 将新数据的tags补进去

            merged_df_new = duplicate_poems_df(merged_df)  # 去重
            print('替换tags中的topics为tags样式')
            for i in tqdm(merged_df_new.index):
                try:
                    topics = eval(merged_df_new.loc[i, 'tags'])['topics']
                except:
                    topics = None
                merged_df_new.loc[i, 'tags'] == str(get_tags(topics=topics))
            merged_df_new.to_csv(f_os_v1 + f_poems_All, index=False, encoding='utf-8')  # 重写入
            # print(merged_df) # 显示合并后的结果
        print('==========12、ok=============')

    if flag_li[12]:
        print('13、去除poems_All中的噪声数据，得到poems_All_clean.csv')
        drop_noise()
        print('==========13、ok=============')

    print('All Done!')
