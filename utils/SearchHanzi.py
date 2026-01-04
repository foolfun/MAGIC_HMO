from utils.base_param import BaseParam
import pandas as pd

params = BaseParam()
f_os_base = params.data_file_os
f_hanzi = f_os_base + 'hanzi/'


class Hanzi:
    def __init__(self):
        self.df_hanzi = pd.read_csv(params.f_hanzi_poems)
        # self.df_hanzi = ''

    def recommendHanzi(self, bushou, wuxing_lack, poem_content, flag=0):
        '''
        bushou: 部首
        wuxing_lack: 五行缺失
        poem_content: 诗句
        flag: 1表示诗句中的字一起推荐，0表示诗句中的字不推荐
        '''
        # 依据部首、五行缺失、诗句推荐汉字
        df_wx = self.df_hanzi[self.df_hanzi['汉字五行'] == wuxing_lack]
        df_ji = df_wx[df_wx['吉凶寓意'] == '吉   ']
        # bushou = ['氵', '金', '玉', '白', '赤', '月', '魚', '酉', '亻']
        df_good = pd.DataFrame()
        for i in bushou:
            df_f = df_ji[df_ji['基本解释'].str.contains('部首：' + i + '；',regex=False)]
            df_good = pd.concat([df_good, df_f])
        # 形成一个str，只留/字/和/拼音/
        good_str = ''
        good_zi = []
        good_py = []
        for index, row in df_good.iterrows():
            good_str += row['字'] + '(' + row['拼音'] + ')、'
            good_zi.append(row['字'])
            good_py.append(row['拼音'])

        poem = poem_content.replace('|', '')
        py = []
        for i in poem:
            df_ = self.df_hanzi[self.df_hanzi['字'] == i]
            py.append(df_.iloc[0]['拼音'])
        # 做交集
        g_py = set(good_py)
        py = set(py)
        a = g_py & py
        fin_zi = []
        if flag == 1:
            for ind in range(len(good_zi)):
                if good_py[ind] in a:
                    fin_zi.append(good_zi[ind] + '(' + good_py[ind] + ')')
        else:
            for ind in range(len(good_zi)):
                if (good_py[ind] in a) and (good_zi[ind] not in poem):
                    fin_zi.append(good_zi[ind] + '(' + good_py[ind] + ')')
        return fin_zi

    def wxHanzi(self, wuxing_lack, poem_content):
        '''
        wuxing_lack: 五行缺失
        poem_content: 诗句
        '''
        fin_zi = []
        # 依据五行缺失、诗句推荐吉利的汉字
        df_wx = self.df_hanzi[self.df_hanzi['汉字五行'] == wuxing_lack]
        df_wx['吉凶寓意'].fillna('无', inplace=True)
        df_ji = df_wx[df_wx['吉凶寓意'].str.contains('吉', regex=False)]
        poem = poem_content.replace('|', '')
        for i in poem:
            df_ = df_ji[df_ji['字'] == i]
            fin_zi = df_['字'].values.tolist()
        return fin_zi

    def getWuxingByHanzi(self, hanzi):  # 字搜五行
        df_hanzi = self.df_hanzi[self.df_hanzi['字'] == hanzi]
        if df_hanzi.shape[0] == 0:
            return '无'
        elif df_hanzi.shape[0] > 1:
            print('Warning: 重复字')
        elif pd.isnull(df_hanzi['汉字五行'].values[0]) or df_hanzi['汉字五行'].values[0] == '' or \
                df_hanzi['汉字五行'].values[0] == 'nan':
            return '无'
        return df_hanzi['汉字五行'].values[0]

    def getWuxingByHli(self, hanzi_li):  # 字搜五行 返回字典{'x':'金', 'x':'木'}
        wuxing_li = {}
        for hanzi in hanzi_li:
            wuxing_li[hanzi] = self.getWuxingByHanzi(hanzi)
        return wuxing_li

    def getWuxingCombByHli(self, hanzi_li):  # 字搜五行 返回字典{'金':[x,x], '木':[x,x]}
        wuxing_comb = {}
        for hanzi in hanzi_li:
            wuxing = self.getWuxingByHanzi(hanzi)
            if pd.isnull(wuxing):
                wuxing = '无'
            if wuxing in wuxing_comb:
                wuxing_comb[wuxing].append(hanzi)
            else:
                wuxing_comb[wuxing] = [hanzi]
        return wuxing_comb

    def getHanziByWuxing(self, wuxing):  # 五行搜字
        df_hanzi = self.df_hanzi[self.df_hanzi['汉字五行'] == wuxing]
        df_re = df_hanzi[['字', '相关诗词']]
        return df_re

    def getPoemsByWuxing(self, wuxing, df):  # 五行搜诗词
        # self.df_poems = pd.read_csv(f_os_base + 'poems_all_v3_05.csv')
        # 五行搜字和相关诗词,相关诗词为nan的去除
        df_hanzi = self.df_hanzi[self.df_hanzi['汉字五行'] == wuxing]
        df_hanzi = df_hanzi.dropna(subset=['相关诗词'])
        # hanzi_li = df_hanzi['字'].values.tolist()
        related_poems_li = df_hanzi['相关诗词'].values.tolist()
        pids = list(set(sum([eval(p) for p in related_poems_li], [])))
        # 依据pids获取相关诗词
        df_poems = df[df['id'].isin(pids)]
        # return hanzi_li, df_poems
        return df_poems

    def getCiyu(self, hanzi):
        df_hanzi = self.df_hanzi[self.df_hanzi['字'] == hanzi]
        return df_hanzi['相关词语'].values[0] + df_hanzi['相关成语'].values[0]


if __name__ == '__main__':
    hz = Hanzi()
    # hz.hanziLinkPoems() # 先处理
    poems = hz.getPoemsByWuxing('金')
    print(poems)
    wuxing = hz.getWuxingByHanzi('饰')
    print(wuxing)
    df = hz.getHanziByWuxing('金')
    print(df)
