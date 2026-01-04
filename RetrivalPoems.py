'''
检索相关古诗词
'''
from build_dataset.preprocess_ori_data import keep_chinese, trans_dy, keep_chinese_and_pipe
import pandas as pd
from utils.base_param import BaseParam
from utils.SearchHanzi import Hanzi
import re

params = BaseParam()
hz = Hanzi()

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    # model_name="lier007/xiaobu-embedding-v2",
    # model_name="/mnt/disk1/zsl/cache_huggingface_hub/hub/models--lier007--xiaobu-embedding-v2/snapshots/ee0b4ecdf5eb449e8240f2e3de2e10eeae877691",
    model_name="/nfs-data/user14/cache_huggingface_hub/hub/models--lier007--xiaobu-embedding-v2/snapshots/1912f2e59a5c2ef802a471d735a38702a5c9485e",
    show_progress=True)
topics_db = Chroma(
    persist_directory=params.data_file_os + 'db_id_metadata_topics_xiaobu_latest',
    embedding_function=embeddings)
imp_db = Chroma(
    persist_directory=params.data_file_os + 'db_id_metadata_implication_xiaobu_latest',
    embedding_function=embeddings)



class RetrivalPoems:
    def __init__(self, base_info, query_key):
        self.query_key = query_key
        self.base_info = base_info
        if  base_info != {}:
            # 个人信息
            self.lunar_zodiac = base_info["生肖"]
            self.meaning = base_info['寓意']
            if '|' not in self.meaning:
                self.meaning += '|'  # 保证寓意中有至少一个关键词
            self.lunar_season = base_info["季节"]
            self.lunar_solar_term = base_info["节气"]
            self.lunar_holidays = base_info["节日"]
            self.lack_wuxing = base_info["五行缺失"]
            # 重新整合孩子的基础信息 base_info_li，仅包括姓氏、农历、生肖、八字、季节、节气、节日、寓意、其他信息
            baby_data = [self.lunar_zodiac, self.lunar_season]
            if self.lunar_solar_term != '无':
                baby_data.append(self.lunar_solar_term)
            if self.lunar_holidays != '无':
                baby_data.append(self.lunar_holidays)
            if self.meaning != '无':
                baby_data.append(self.meaning)
            self.base_info_li = [' '.join(baby_data)]

    def get_recommend_words(self, df, lwx):
        def combine(row, cand_words, wx_words):  # 定义一个函数来解析字符串并相加
            if pd.isnull(row[wx_words]):
                row[wx_words] = ''
            wx_name = wx_words.split('_')[0]
            if row[wx_words] == '':
                return row[cand_words]
            tmp_words = '诗句中属性为‘' + wx_name + '’的汉字有' + str(eval(row[wx_words]))
            row[cand_words] += tmp_words
            return row[cand_words]

        cols = ['id', 'title', 'dynasty', 'author', 'content', 'implication', 'tags', 'metaphor',
                'retrieval_info']
        if len(lwx) > 0:
            for wx in lwx:
                cols.append(wx + '_words')
        df_new = df[cols]
        # 合并所有列中的词汇
        df_new = df_new.copy()
        df_new.loc[:, 'cand_words'] = ''
        # 临时启用未来行为
        pd.set_option('future.no_silent_downcasting', True)
        df_new.fillna('', inplace=True)
        # 恢复选项到默认值
        pd.reset_option('future.no_silent_downcasting')
        for wx in lwx:
            df_new['cand_words'] = df_new.apply(combine, axis=1, args=('cand_words', wx + '_words'))
        return df_new

    def filter_poems(self, df, info):
        df_new = pd.DataFrame()
        if info != '无' and info != '':
            df_new = df[df['topics'].str.contains(info)]
        return df_new

    def searchByEmb(self, query, vector_db, top_k=100):
        results = vector_db.similarity_search_with_score(query, k=top_k)
        inds = [result[0].metadata['document'] for result in results]
        return inds

    def searchByEmb_new(self, query, vector_db, filter_ids, top_k=100):
        # results = vector_db.similarity_search_with_score(query, k=top_k)
        results = vector_db.similarity_search_with_score(query, k=top_k, filter={'document': {'$in': filter_ids}})
        inds = [result[0].metadata['document'] for result in results]
        # for i in inds:
        #     if str(52083) not in filter_ids:
        #         print('Warning: 检索结果中有不在filter')
        return inds

    def get_poems(self, df_poems, poem_type='', ids=list, num=5):
        if len(self.base_info) > 0:
            return self.get_poems_name(df_poems, poem_type, ids, num)
        else:
            return self.get_poems_others(df_poems, poem_type, ids, num)

    def get_poems_others(self, df_poems, poem_type='', ids=list, num=5):
        if len(ids) > 0:
            df_type = df_poems[~df_poems['id'].isin(ids)]
        else:
            df_type = df_poems

        # step1 筛查type
        if poem_type not in ['无', '']:
            print(poem_type)
            poem_type = keep_chinese_and_pipe(poem_type)
            type_li = poem_type.split('|')
            df_type = df_type[df_type['type'].apply(lambda x: all(t in x for t in type_li))]
            if len(df_type) == 0:
                df_type = df_type[df_type['type'].str.contains(poem_type)]
            if len(df_type) == 0:
                df_type = df_poems

        # step3 query_key与topics匹配，取出相关古诗词
        topk = min(300, int(len(df_type) / 2))
        inds_filter = self.searchByEmb(self.query_key, topics_db, top_k=topk)  # 通过寓意进行筛选
        df_filter = df_type[df_type['id'].astype(str).isin(inds_filter)]
        filter_ids = df_filter['id'].astype(str).tolist()

        # step4 query_key与implication匹配，取出相关古诗词
        inds_final = self.searchByEmb_new(self.query_key, imp_db, filter_ids, top_k=num)  # 通过模型生成的关键词进行筛选
        df_final = df_filter[df_filter['id'].astype(str).isin(inds_final)]

        # step5 生成古诗词信息,返回
        # 临时启用未来行为
        pd.set_option('future.no_silent_downcasting', True)
        df_final = df_final.fillna('')
        # 恢复选项到默认值
        pd.reset_option('future.no_silent_downcasting')
        df_final.loc[:, 'merge_info'] = df_final.apply(lambda row: (str(row['dynasty']) +
                                                                    '·' + str(row['author'])
                                                                    + '《' + str(row['title']) + '》' +
                                                                    '\n诗句：' + str(row['content']) +
                                                                    '。\n赏析：' + str(row['implication'])
                                                                    ).replace('nan', ''), axis=1)
        poems_li = []
        ids = df_final['id'].tolist()
        for i in df_final.iloc[:]['merge_info']:
            poems_li.append(i)

        return poems_li, ids
    def get_poems_name(self, df_poems, poem_type='', ids=list, num=5):
        pre_flag = True
        if len(ids) > 0:
            df_type = df_poems[~df_poems['id'].isin(ids)]
            if int(len(ids)/5) > 3: # 超过3轮检索尚未找到合适古诗，不再进行初筛选
                pre_flag = False
        else:
            df_type = df_poems

        # step1 筛查type
        if poem_type not in ['无', '']:
            print(poem_type)
            poem_type = keep_chinese_and_pipe(poem_type)
            type_li = poem_type.split('|')
            df_type = df_type[df_type['type'].apply(lambda x: all(t in x for t in type_li))]
            if len(df_type) == 0:
                df_type = df_type[df_type['type'].str.contains(poem_type)]
            if len(df_type) == 0:
                df_type = df_poems

        # step2 初筛选，节日、节气、季节分别匹配topics，再concat合并
        if pre_flag:
            df_festival = self.filter_poems(df_type, self.lunar_holidays)
            df_term = self.filter_poems(df_type, self.lunar_solar_term)
            df_season = self.filter_poems(df_type, self.lunar_season)
            if poem_type not in ['无', '']:
                df_base = pd.concat([df_type, df_festival, df_term, df_season], axis=0).drop_duplicates()  # 合并初筛结果
            else:
                df_base = pd.concat([df_festival, df_term, df_season], axis=0).drop_duplicates()
        else:
            df_base = df_type

        if df_base.shape[0]==0:
            df_base = df_type

        filter_ids = df_base['id'].astype(str).tolist()

        # step3 寓意与topics匹配，取出相关古诗词
        topk = min(300, int(len(df_base) / 2))
        inds_filter = self.searchByEmb_new(self.meaning, topics_db, filter_ids, top_k=topk)  # 通过寓意进行筛选
        df_filter = df_base[df_base['id'].astype(str).isin(inds_filter)]
        filter_ids = df_filter['id'].astype(str).tolist()

        # step4检索词与implication匹配，取出相关古诗词
        inds_final = self.searchByEmb_new(self.query_key, imp_db, filter_ids, top_k=num)  # 通过模型生成的关键词进行筛选
        df_final = df_filter[df_filter['id'].astype(str).isin(inds_final)]

        # step5 筛查包含推荐五行的古诗
        lack_w = self.lack_wuxing
        df_wx = pd.DataFrame()
        if len(lack_w) > 0:
            df_wx = pd.DataFrame(columns=df_final.columns)
            for w in lack_w:
                df_tmp = hz.getPoemsByWuxing(w, df_final)
                df_wx = pd.concat([df_wx, df_tmp], axis=0)
            df_wx = df_wx.drop_duplicates()
            df_wx = self.get_recommend_words(df_wx, lack_w)
            if df_wx.shape[0] > 0:
                df_final = df_wx

        # step6 生成古诗词信息,返回
        # 临时启用未来行为
        pd.set_option('future.no_silent_downcasting', True)
        df_final = df_final.fillna('')
        # 恢复选项到默认值
        pd.reset_option('future.no_silent_downcasting')
        if df_wx.shape[0] > 0:
            # df_final.loc[:, 'merge_info'] = df_final.apply(lambda row: (str(row['dynasty']) +
            #                                                             '·' + str(row['author'])
            #                                                             + '《' + str(row['title']) + '》' +
            #                                                             '\n诗句：' + str(row['content']) +
            #                                                             '。\n赏析：' + str(row['implication']) +
            #                                                             '\n五行汉字：' + str(row['cand_words']) + '。'
            #                                                             ).replace('nan', ''), axis=1)
            df_final.loc[:, 'merge_info'] = df_final.apply(lambda row: (str(row['dynasty']) +
                                                                        '·' + str(row['author'])
                                                                        + '《' + str(row['title']) + '》' +
                                                                        '\n诗句：' + str(row['content']) +
                                                                        '\n赏析：' + str(row['implication']) +
                                                                        str(row['cand_words']) + '。'
                                                                        ).replace('nan', ''), axis=1)
        else:
            df_final.loc[:, 'merge_info'] = df_final.apply(lambda row: (str(row['dynasty']) +
                                                                        '·' + str(row['author'])
                                                                        + '《' + str(row['title']) + '》' +
                                                                        '\n诗句：' + str(row['content']) +
                                                                        '\n赏析：' + str(row['implication'])
                                                                        ).replace('nan', ''), axis=1)
            # df_final.loc[:, 'merge_info'] = df_final.apply(lambda row: (str(row['dynasty']) +
            #                                                             '·' + str(row['author'])
            #                                                             + '《' + str(row['title']) + '》' +
            #                                                             '\n诗句：' + str(row['content']) +
            #                                                             '。\n释义：' + str(row['explain'])+
            #                                                             '\n赏析：' + str(row['implication'])
            #                                                             ).replace('nan', ''), axis=1)
        poems_li = []
        ids = df_final['id'].tolist()
        for i in df_final.iloc[:]['merge_info']:
            poems_li.append(i)

        return poems_li, ids

    def search_poems(self, df, related_info):  # 依据用户提供的古诗信息，检索相关古诗
        # 将古诗内容与标题中的非中文字符去除
        df['cont_text'] = df['content'].apply(lambda x: re.sub(r'[^\u4e00-\u9fa5]', '', x))
        df.drop_duplicates(subset=['cont_text'], inplace=True)
        df['title_text'] = df['title'].apply(lambda x: re.sub(r'[^\u4e00-\u9fa5]', '', x))
        df_merge = pd.DataFrame(columns=df.columns)
        poems_li = []
        for info in related_info:
            title, author, dynasty, content = info
            # 按照标题、诗人、朝代、诗句进行检索，若某一项为空则按照其他项检索
            title = keep_chinese(title) if title != '无' else ''
            author = author if author != '无' else ''
            dynasty = trans_dy(dynasty) if dynasty != '无' else ''
            content = keep_chinese(content) if content != '无' else ''
            # 检索
            df_ = df[(df['title_text'].str.contains(title))
                     & (df['author'].str.contains(author))
                     & (df['dynasty'].str.contains(dynasty))
                     & (df['cont_text'].str.contains(content))]
            df_merge = pd.concat([df_merge, df_], axis=0)

        # 若检索结果大于5项，则进行细筛
        try:
            if df_merge.shape[0] > 5:
                filter_ids = df_merge['id'].astype(str).tolist()
                inds_final = self.searchByEmb_new(self.base_info_li[0], imp_db, filter_ids, top_k=5)
                # inds_final = self.searchByEmb_new(self.meaning, self.imp_db, filter_ids, top_k=5)
                df_merge = df_merge[df_merge['id'].astype(str).isin(inds_final)]
        except:
            df_merge = df_merge.head(5)

        # 生成古诗词信息
        df_merge['merge_info'] = df_merge.apply(
            lambda row: (str(row['dynasty']) + '·' + str(row['author']) + '《' + str(row['title']) + '》' +
                         '\n诗句：' + str(row['content']) +
                         '\n赏析：' + str(row['implication'])
                         ).replace('nan', ''), axis=1)

        # 转换为字符串
        for i in df_merge.iloc[:]['merge_info']:
            poems_li.append(i)
        # print('检索到的古诗词：', poems_li)
        return poems_li