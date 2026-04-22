'''
分析诗词数据
1、ccpc_data.csv（由原数据ccpc整合并转换为csv），该数据的古诗词仅有诗人、朝代、标题、内容
2、transpoem_data.csv（由原数据transpoem_data.txt转换得到），该数据的古诗词有部分包含解释
3、分析transpoem_data.csv的数据
4、分析famous和parallel合并结果
5、分析poems_fin.csv的数据
6、分析web数据
7、分析gushiju和db合并含有赏析的数据
8、分析所有数据合并后的数据,all_poems.csv
9、分析all_poems_baidu.csv的数据
10、分析西窗烛的数据
'''

from utils.base_param import BaseParam
import pandas as pd

params = BaseParam()


class AnaDataset:

    def print_info(self, df, file_name):
        print(file_name, '的数据，分析如下：')
        # 统计某一列中包含NaN的数量
        nan_count = pd.isnull(df['explain']).sum()
        row_count = df.shape[0]
        print('总数', row_count, '含有解释的古诗词数量为', row_count - nan_count)
        print('---------------------------------------------------' * 2)
        print('共有：', len(df['dynasty'].unique()), '个不同朝代')
        print('朝代类别：', df['dynasty'].unique())
        print('朝代分布：', df['dynasty'].value_counts(normalize=False, dropna=False))
        print('---------------------------------------------------' * 2)
        print('共有：', len(df['author'].unique()), '个不同诗人')
        print('诗人：', df['author'].unique())
        print('诗人分布：', df['author'].value_counts(normalize=False, dropna=False))
        print('---------------------------------------------------' * 2)

    def ana_data(self, file):
        # 对合并结果进行分析
        df = pd.read_csv(file, encoding='utf-8')
        self.print_info(df, file)

    def ana_data_imp(self, file):
        df = pd.read_csv(file, low_memory=False, encoding='utf-8')
        nan_count_exp = 0
        nan_count_imp = 0
        nan_count_key = 0
        nan_count_key_imp = 0
        count_key = 0
        for i in df.index:
            # print(df.loc[i, 'title'])
            if pd.isnull(df.loc[i,'explain']) or pd.isna(df.loc[i, 'explain']) or df.loc[i, 'explain'] == 'nan':
                nan_count_exp += 1
            if pd.isnull(df.loc[i,'implication']) or pd.isna(df.loc[i, 'implication']) or df.loc[i, 'implication'] == 'nan':
                nan_count_imp += 1
            if count_key:
                if pd.isnull(df.loc[i,'key_words']) or pd.isna(df.loc[i, 'key_words']) or df.loc[i, 'key_words'] == 'nan':
                    nan_count_key += 1
                if pd.isnull(df.loc[i,'key_words_imp']) or pd.isna(df.loc[i, 'key_words_imp']) or df.loc[i, 'key_words_imp'] == 'nan':
                    nan_count_key_imp += 1
        rows = df.shape[0]
        rows_exp = rows - nan_count_exp
        rows_imp = rows - nan_count_imp
        print('含有解释:', rows_exp)
        print('含有鉴赏:', rows_imp)
        if count_key:
            rows_key = rows - nan_count_key
            rows_key_imp = rows - nan_count_key_imp
            print('含有关键词:', rows_key)
            print('含有关键词解释:', rows_key_imp)
        self.print_info(df, file)


if __name__ == '__main__':
    base_os = params.data_file_os
    inter_os = params.inter_data_os
    web_os = params.web_data_os
    poem_os = params.poems_all_os

    ad = AnaDataset()
    # # 1、分析ccpc_data.csv的数据
    # f_ccpc_c = params.ccpc_os + params.f_ccpc_data_c  # ccpc_data.csv
    # ad.ana_data_imp(f_ccpc_c)
    # print('*' * 100)

    # # 2、分析transpoem_data.csv的数据
    # f_pt_c = params.pc_os + params.f_parallel_transpoem_c  # transpoem_data.csv
    # ad.ana_data_imp(f_pt_c)
    # print('*' * 100)

    # # 3、分析famous_poem.csv的数据
    # f_fam_c = params.famous_os + params.f_famous_c  # famous_poems.csv
    # ad.ana_data_imp(f_fam_c)
    # print('*' * 100)

    # # 4、分析famous和parallel合并结果
    # f_m_fp = inter_os + params.f_merge_tf
    # ad.ana_data_imp(f_m_fp)
    # print('*' * 100)

    # # 5、分析poems_fin.csv的数据
    # f_m_pdb = params.pdm_os + params.f_merge_poems_db
    # ad.ana_data_imp(f_m_pdb)
    # print('*' * 100)

    # # 6、分析gushiju数据
    # f_gushiju = web_os + params.f_gushiju_new
    # ad.ana_data_imp(f_gushiju)
    # print('*' * 100)

    # # 7、分析gushiju和db合并含有赏析的数据
    # f_m_gdb = inter_os + params.f_merge_gg_imp
    # ad.ana_data_imp(f_m_gdb)
    # print('*' * 100)


    # # 8、分析poems_Base的数据
    # f_all_poems = inter_os + params.f_poems_Base
    # ad.ana_data_imp(f_all_poems)
    # print('*' * 100)

    # # 10、分析西窗烛的数据
    # f_xcz = params.f_xcz_new
    # ad.ana_data_imp(f_xcz)
    # print('*' * 100)

    # # 11、poems_BaW.csv的数据
    # f_all_poems = inter_os + params.f_poems_BaW
    # ad.ana_data_imp(f_all_poems)
    # print('*' * 100)

    # # 12、分析poems_All.csv的数据
    # f_all_poems = poem_os + 'poems_All.csv'
    # ad.ana_data_imp(f_all_poems)
    # print('*' * 100)

    # test
    f_all_poems_clean = poem_os + 'poems_All_clean.csv'
    ad.ana_data_imp(poem_os)
    print('*' * 100)