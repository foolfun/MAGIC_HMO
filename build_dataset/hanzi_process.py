'''
读取hanzi文件夹中的所有.csv文件，合并成一个文件hanzi_all.csv
得到一个简化的汉字数据集hanzi_all_simple.csv
'''

import numpy as np
from utils.base_param import BaseParam
import os
import pandas as pd

params = BaseParam()
f_os_base = params.data_file_os
f_hanzi = f_os_base + 'hanzi/'

if __name__ == '__main__':
    # 从hanzi文件夹中读取所有的.csv文件
    hanzi_files = [f_hanzi + f for f in os.listdir(f_hanzi) if f.endswith('.csv')]
    # 读取所有的汉字
    columns = ['id', '字', '拼音', '部首', '部首笔画', '总笔画', '康熙字典笔画',
               '五笔86', '五笔98', 'Unicode', '汉字五行', '吉凶寓意', '是否为常用字', '姓名学',
               '笔顺读写', '基本解释', '新华字典详细解释', '汉语大字典解释', '康熙字典解释', '说文解字详解',
               '说文解字详鰹图片', '字源演李图片', '相关书法', '相关词语', '相关成语', '相关诗词', '康熙字典原图']
    df_hanzi = pd.DataFrame(columns=columns)

    for f in hanzi_files:
        # 读取文件时指定列名
        df_tmp = pd.read_csv(f, usecols=range(27), header=None, names=columns)
        df_hanzi = pd.concat([df_hanzi, df_tmp], axis=0, ignore_index=True)  # 使用ignore_index=True忽略索引

    print("Shape of concatenated DataFrame:", df_hanzi.shape)
    df_hanzi.to_csv(f_hanzi + 'hanzi_all.csv', index=False)
    df_hanzi_simple = df_hanzi[
        ['字', '拼音', '汉字五行', '吉凶寓意', '姓名学', '基本解释', '新华字典详细解释', '汉语大字典解释',
         '康熙字典解释',
         '说文解字详解',
         '相关词语', '相关成语', '相关诗词']]
    df_hanzi_simple.loc[df_hanzi_simple['字'] == '饰', '汉字五行'] = np.nan
    df_hanzi_simple.to_csv(f_os_base + 'hanzi_all_simple.csv', index=False)


    # # 读取和分析
    # df_hanzi_simple = pd.read_csv(f_os_base + 'hanzi_all_simple.csv')
    # df_related_poems = df_hanzi_simple[df_hanzi_simple['相关诗词'].notnull()]['相关诗词']
    # print(df_hanzi_simple.head())
    # # 怡,yí,土,吉,姓,怡 yí 和悦，愉快：怡色（容色和悦）。怡声（语声和悦）。怡和。怡乐（l?）。怡神。怡悦。怡目（快意于所见，悦目）。心旷神怡。  笔画数：8； 部首：忄； 笔顺编号：44254251,怡 yí 【形】 (形声。从心,台(yí)声。本义:和悦的样子) 同本义〖cheerful;happy〗 怡,和也。——《说文》 公乃为诗以怡王。——《书·金传》。郑注:“悦也。” 下气怡色。——《礼记·内则》。注:“悦也。” 有庆未尝不怡。——《国语·周语》 〖亲稚〗狗之事大矣,而主之色不怡,何也?——《国语》 怡然自乐。——晋·陶渊明《桃花源记》 心旷神怡。——宋·范仲淹《岳阳楼记》 又如:怡心(和悦心情);怡目(悦目);怡怡(怡悦神情);怡情(怡悦心情);怡魂(使精神愉快);怡养(和乐);怡声(犹柔声);怡颜(和悦的容颜);怡宁(安宁) 喜乐的,使人心神感官愉快的〖pleasant〗 怡,乐也。——《尔雅》 眄庭柯以怡颜。——晋·陶渊明《归去来兮辞》 如:怡色;怡愉(喜悦);怡裕(安乐,欢乐);怡畅(欢畅);怡荡(怡悦放荡);怡乐(安乐,快乐);怡穆(愉悦和睦);怡怿(愉悦;快乐) 〖名〗∶姓  怡和 yíhé 〖kindly〗∶愉快和悦 神情怡和 〖fine〗∶风日和美 怡乐 yílè 〖joyful〗安乐;快乐 怡怡 yíyí 〖happy〗形容喜悦欢乐的样子 融融怡怡。——唐·李朝威《柳毅传》 怡然 yírán 〖happy;joyful;cheerful〗喜悦的;安适自在的样子 怡悦 yíyuè 〖happy〗喜悦;高兴 心情怡悦,[①]［yí］［《廣韻》與之切，平之，以。］“怠2”的被通假字。(1)和悦。(2)喜悦，快乐。(3)姓。北周有怡峰。见《周书》本传。,【卯集上】【心字部】\u3000怡； 康熙笔画：9； 页码：页381第16(点击查看原图)【唐韻】與之切【集韻】【韻會】盈之切【正韻】延知切，?音飴。【爾雅·釋言】悅也。【說文】和也。【玉篇】樂也。【禮·內則】下氣怡色。【論語】兄弟怡怡。\u3000又姓。周怡峰，本姓默合，避難改焉。\u3000又通作台。【史記·序傳】諸呂不台，言不爲人所怡悅也。,【卷十】【心部】 编号：6709\u3000\u3000怡，[與之切 ]，和也。从心台聲。,·安怡·处之怡然·愕怡·和怡·旷心怡神·秦怡(19·清怡·融融怡怡·融怡·神怡·神怡心旷·陶怡·熙怡·嬉怡·下气怡色·下气怡声·心荡神怡·心旷神怡,·旷心怡神·神怡心旷·下气怡色·下气怡声·心荡神怡·心旷神怡·心怡神旷·心悦神怡·兄弟怡怡·怡情理性·怡情养性·怡情悦性·怡然自得·怡然自乐·怡然自若·怡声下气·怡堂燕雀·怡性养神,·怡红快绿·赠漳州张怡使君·蝶恋花 戏题疏齐怡·南乡子 赠歌者怡云·怡然以垂云新茶见饷·怡斋·仗锡平老自都城回见·伐木赠张先怡·怡云山房诗·雷怡真小隐送春·怡斋·赠雷怡真
    # df_yi = df_hanzi_simple[df_hanzi_simple['字'] == '怡']
    # print('字', df_yi['字'].values[0])
    # print('拼音', df_yi['拼音'].values[0])
    # print('汉字五行', df_yi['汉字五行'].values[0])
    # print('吉凶寓意', df_yi['吉凶寓意'].values[0])
    # print('姓名学', df_yi['姓名学'].values[0])
    # print('基本解释', df_yi['基本解释'].values[0])
    # print('新华字典详细解释', df_yi['新华字典详细解释'].values[0])
    # print('汉语大字典解释', df_yi['汉语大字典解释'].values[0])
    # print('康熙字典解释', df_yi['康熙字典解释'].values[0])
    # print('说文解字详解', df_yi['说文解字详解'].values[0])
    # print('相关词语', df_yi['相关词语'].values[0])
    # print('相关成语', df_yi['相关成语'].values[0])
    # print('相关诗词', df_yi['相关诗词'].values[0])
