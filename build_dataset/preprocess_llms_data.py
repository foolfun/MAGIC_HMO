'''
处理LLMs补充好的数据
'''
import pandas as pd
from tools.base_param import BaseParam
import numpy as np
import re
from tqdm import tqdm
import os
from tools.SearchHanzi import Hanzi
from collections import defaultdict
from build_dataset.preprocess_ori_data import trans_con

param = BaseParam()
hz = Hanzi()
f_os_base = param.data_file_os
f_v1_os = param.poems_all_v1_os
f_v2_os = param.poems_all_v2_os
f_v3_os = param.poems_all_v3_os
f_os_web = param.web_data_os
f_souyun_word = param.f_souyun_word


def merge_poems_all_files(f_os, start_):
    # 合并f_os路径下的所有以“start_”开头的csv文件，以第一个csv文件列名作为列名，并去重
    files = os.listdir(f_os)
    files = [f for f in files if f.startswith(start_)]
    df_p = pd.read_csv(f_os + files[0], low_memory=False)
    for f in files[1:]:
        df_ = pd.read_csv(f_os + f, low_memory=False)
        df_p = pd.concat([df_p, df_], axis=0, ignore_index=True)

    # 对implication做处理,去空值
    df_p.replace('', np.nan, inplace=True)
    df_p.dropna(subset=['implication'], inplace=True)
    df_p.fillna('', inplace=True)

    # df按照content列去重
    df_p.drop_duplicates(subset=['content'], keep='first', inplace=True)

    # 对tags做处理
    for i in tqdm(df_p.index):
        tags = eval(df_p.loc[i, 'tags'])
        # 如果tags中的隐喻为none，则删去df中的这行
        try:
            if tags['metaphor'] is None or tags['metaphor'] == '' or tags['metaphor'] == '{}':
                df_p.drop(i, inplace=True)
        except:
            df_p.drop(i, inplace=True)

    # 存储
    df_p.replace('', np.nan, inplace=True)
    print(df_p.isna().any())  # 检查是否有空值
    df_p.dropna(subset=['explain'], inplace=True)  # 去除解释为空的行
    df_p.to_csv(f_os + start_ + '.csv', index=False, encoding='utf-8')


def drop_exp_noise(df_):
    # 对explain做处理
    for i in tqdm(df_.index):
        exp = df_.loc[i, 'explain']
        exp = exp.replace(' ', '')
        if exp.count('中文译文：') > 1 or exp.count('译文：') > 1:
            df_.loc[i, 'explain'] = ''
            continue
        elif exp.count('中文译文：') == 1:
            exp = exp[exp.index('中文译文：') + 5:]
        elif exp.count('译文:') == 1:
            exp = exp[exp.index('译文:') + 3:]

        exp = re.sub(r'[\n\u3000\r\t]', '', exp)
        pattern = re.compile(r'[^\u4e00-\u9fa5]')
        exp_text = pattern.sub('', exp)
        con_text = pattern.sub('', df_.loc[i, 'content'])
        if (len(exp) < 10) or (con_text in exp_text):
            exp = ''
        df_.loc[i, 'explain'] = exp
    return df_


def add_key_words(df):
    # df对应到df_all中对应的行，然后将其key_words_imp列的内容补充到df中
    df_all = pd.read_csv(f_v1_os + 'poems_All_0504.csv', low_memory=False)
    df_merged = df.merge(df_all[['content', 'key_words_imp']], on='content', how='left')
    df['key_words_imp'] = df_merged['key_words_imp']
    print('key_words_imp已添加！')
    return df


def build_db(df_db, new_f_test):
    '''
      增加tag_all，删除key_words_imp列
    '''
    # 删除key_words列
    if 'key_words' in df_db.columns:
        df_db.drop('key_words', axis=1, inplace=True)
    # 对key_words_imp做处理,将原来的key_words_imp加回来
    if 'key_words_imp' not in df_db.columns:
        df_db = add_key_words(df_db)
    df_db['key_words_imp'] = df_db['key_words_imp'].fillna('')

    # 对tags做合并处理
    cnt = 0
    df_db['key_words'] = ''
    for i in tqdm(df_db.index):
        try:
            tags = eval(df_db.loc[i, 'tags'])
            df_db.loc[i, 'metaphor'] = str(tags['metaphor'])
            if df_db.loc[i, 'key_words_imp'] != '':
                met = eval(df_db.loc[i, 'metaphor'])
                k_imp = eval(df_db.loc[i, 'key_words_imp'])
                met.update(k_imp)  # 将key_words_imp中的值增加到metaphor中
                df_db.loc[i, 'metaphor'] = str(met)
                tags = eval(df_db.loc[i, 'tags'])
                tags['metaphor'] = met
                df_db.loc[i, 'tags'] = str(tags)  # 更新tags列

            key_li = list(eval(df_db.loc[i, 'metaphor']).keys())
            df_db.loc[i, 'key_words'] += '|'.join(key_li)

            df_db.loc[i, 'tags_all'] = ('该诗事件：{event}。事件时间：{event_time}。'
                                        '事件地点：{event_location}。事件节日：{event_holiday}。'
                                        '事件季节：{event_season}。事件天气：{event_weather}。'
                                        '情感：{sentiment}。隐喻：{metaphor}').format(**tags)
        except:
            # print(tags)
            cnt += 1
            df_db.drop(i, inplace=True)
            df_db.loc[i, 'key_words'] = ''
            df_db.loc[i, 'metaphor'] = ''
    print('异常个数:', cnt)
    # 删除df_poems中key_words和metaphor列中为空的行
    df_db = df_db[df_db['key_words'] != '']
    df_db = df_db[df_db['metaphor'] != '']
    print('metaphor列中为空的行：', df_db.shape)

    df_db['retrieval_info'] = df_db.apply(lambda row: (str(row['implication']) + str(row['tags_all'])
                                                       ).replace('nan', ''), axis=1)

    print(new_f_test, df_db.shape)

    # 删除不需要的列
    df_db.drop('key_words_imp', axis=1, inplace=True)
    # 保存
    df_db.to_csv(new_f_test, index=False, encoding='utf-8')
    print('保存成功！')


def poemsLinkHanzi(hanzi_file, poems_file, new_poem_file, new_hp_file):
    df_hanzi = pd.read_csv(hanzi_file)
    df_poems = pd.read_csv(poems_file, low_memory=False)
    # 给df_poems添加一个索引列
    df_poems['id'] = df_poems.index
    # 将df_hanzi中相关诗词列置空
    df_hanzi['相关诗词'] = ''
    # 给df_poems添加一个words列
    df_poems['words'] = ''
    # 通过df_poems中的key_words列，将df_hanzi中相关诗词列填充，并给df_poems添加一个words列
    for i in tqdm(df_hanzi.index):
        hanzi = df_hanzi.loc[i, '字']
        df_poems.loc[df_poems['key_words'].str.contains(hanzi,regex=False), 'words'] += hanzi + '|'  # df_poems添加一个words列
        poems_li = df_poems[df_poems['key_words'].str.contains(hanzi,regex=False)]['id'].values.tolist()
        if len(poems_li) > 0:
            df_hanzi.loc[i, '相关诗词'] = str(poems_li)
        else:
            df_hanzi.loc[i, '相关诗词'] = ''
    # 如果相关诗词列为空则置为nan
    df_hanzi['相关诗词'] = df_hanzi['相关诗词'].apply(lambda x: x if x != '' else None)
    # 将df_poems中的words列的最后一个|去掉，并去重
    df_poems['words'] = df_poems['words'].apply(lambda x: str(list(set(x.split('|')[:-1]))))
    # 删除df_poems中words列为空的行
    df_poems = df_poems[df_poems['words'] != '[]']

    # for i in tqdm(df_poems.index):
    #     tags = eval(df_poems.loc[i, 'tags'])
    #     df_poems.loc[i, 'tags_tmp'] = ('该诗事件：{event}。事件时间：{event_time}。'
    #                                    '事件地点：{event_location}。事件节日：{event_holiday}。'
    #                                    '事件季节：{event_season}。事件天气：{event_weather}。'
    #                                    '情感：{sentiment}。').format(**tags)

    # 保存df_poems和df_hanzi
    df_poems.to_csv(new_poem_file,index=False)
    df_hanzi.to_csv(new_hp_file, index=False)
    print('df_poems（poems_all_v3.csv）和df_hanzi（hanzi_poems_v3.csv）保存成功！')


def addWuxingcol(poems_file, new_poem_file=''):
    if new_poem_file == '':
        new_poem_file = poems_file
    df_v3 = pd.read_csv(poems_file, low_memory=False)
    # 遍历df_v3的words列，将诗词中的字对应到汉字表中，得到对应的五行，新增列hwuxing: {'字':'五行'}
    df_v3['hwuxing'] = ''
    for i in tqdm(df_v3.index):
        words = eval(df_v3.loc[i, 'words'])
        hw_dic = {}
        for j in words:
            hw_dic[j] = hz.getWuxingByHanzi(j)
            if pd.isnull(hw_dic[j]):
                hw_dic[j] = '无'
        df_v3.loc[i, 'hwuxing'] = str(hw_dic)
    # 保存
    df_v3.to_csv(new_poem_file, index=False, encoding='utf-8')

    # 构建五行words列，金_words,木_words,水_words,火_words,土_words 无_words
    df_v3['金_words'] = ''
    df_v3['木_words'] = ''
    df_v3['水_words'] = ''
    df_v3['火_words'] = ''
    df_v3['土_words'] = ''
    df_v3['无_words'] = ''
    for i in tqdm(df_v3.index):
        hw = eval(str(df_v3.loc[i, 'hwuxing']).replace('nan', "'无'"))
        revse_hw = defaultdict(list)
        for k, v in hw.items():
            if pd.isna(v):
                v = '无'
            revse_hw[v].append(k)
        for wux, w_li in revse_hw.items():
            df_v3.loc[i, wux + '_words'] = str(w_li)
    # 保存
    df_v3.to_csv(new_poem_file, index=False, encoding='utf-8')


def add_daxue():
    title = '大学'
    dynasty = '先秦'
    author = '曾子'
    content = '''大学之道，在明明德，在亲民，在止于至善。知止而后有定，定而后能静，静而后能安，安而后能虑，虑而后能得。物有本末，事有终始。知所先后，则近道矣。古之欲明明德于天下者，先治其国。欲治其国者，先齐其家。欲齐其家者，先修其身。欲修其身者，先正其心。欲正其心者，先诚其意。欲诚其意者，先致其知。致知在格物。物格而后知至，知至而后意诚，意诚而后心正，心正而后身修，身修而后家齐，家齐而后国治，国治而后天下平。自天子以至于庶人，壹是皆以修身为本。 其本乱而末治者否矣。其所厚者薄，而其所薄者厚，未之有也。此谓知本，此谓知之至也。所谓诚其意者，毋自欺也。如恶恶臭，如好好色，此之谓自谦。故君子必慎其独也。小人闲居为不善，无所不至，见君子而后厌然，掩其不善而著其善。 人之视己，如见其肺肝然，则何益矣。此谓诚于中，形于外，故君子必慎其独也。 曾子曰：“十目所视，十手所指，其严乎！”富润屋，德润身，心广体胖，故君子必诚其意。《诗》云：“瞻彼淇澳，菉竹猗猗。有斐君子，如切如磋，如琢如磨。 瑟兮僴兮，赫兮喧兮。有斐君子，终不可喧兮。”“如切如磋”者，道学也。 “如琢如磨”者，自修也。“瑟兮僴兮”者，恂傈也。“赫兮喧兮”者，威仪也。“有斐君子，终不可喧兮”者，道盛德至善，民之不能忘也。《诗》云：“於戏，前王不忘！”君子贤其贤而亲其亲，小人乐其乐而利其利，此以没世不忘也。《康诰》曰：“克明德。”《大甲》曰：“顾諟天之明命。”《帝典》曰： “克明峻德。”皆自明也。汤之《盘铭》曰：“茍日新，日日新，又日新。”《康诰》曰：“作新民。” 《诗》曰：“周虽旧邦，其命维新。”是故君子无所不用其极。《诗》云：“邦畿千里，维民所止。”《诗》云：“缗蛮黄鸟，止于丘隅。” 子曰：“于止，知其所止，可以人而不如鸟乎？”《诗》云：“穆穆文王，於缉熙敬止！”为人君，止于仁；为人臣，止于敬；为人子，止于孝；为人父，止于慈； 与国人交，止于信。子曰：“听讼，吾犹人也。必也使无讼乎！”无情者不得尽其辞，大畏民志。此谓知本。所谓修身在正其心者，身有所忿懥，则不得其正，有所恐惧，则不得其正， 有所好乐，则不得其正，有所忧患，则不得其正。心不在焉，视而不见，听而不闻，食而不知其味。此谓修身在正其心。所谓齐其家在修其身者，人之其所亲爱而辟焉，之其所贱恶而辟焉，之其所畏敬而辟焉，之其所哀矜而辟焉，之其所敖惰而辟焉。故好而知其恶，恶而知其美者，天下鲜矣。故谚有之曰：“人莫知其子之恶，莫知其苗之硕。”此谓身不修，不可以齐其家。所谓治国必先齐其家者，其家不可教而能教人者，无之。故君子不出家而成教于国。孝者，所以事君也；弟者，所以事长也；慈者，所以使众也。《康诰》 曰：“如保赤子。”心诚求之，虽不中，不远矣。未有学养子而后嫁者也。一家仁，一国兴仁；一家让，一国兴让；一人贪戾，一国作乱，其机如此。此谓一言偾事， 一人定国。尧、舜率天下以仁，而民从之。桀、纣率天下以暴，而民从之。其所令反其所好，而民不从。是故君子有诸己而后求诸人，无诸己而后非诸人。所藏乎身不恕，而能喻诸人者，未之有也。故治国在齐其家。《诗》云：“桃之夭夭， 其叶蓁蓁。之子于归，宜其家人。”宜其家人，而后可以教国人。《诗》云：“ 宜兄宜弟。”宜兄宜弟，而后可以教国人。《诗》云：“其仪不忒，正是四国。” 其为父子兄弟足法，而后民法之也。此谓治国在齐其家。所谓平天下在治其国者，上老老而民兴孝，上长长而民兴弟，上恤孤而民不倍，是以君子有絜矩之道也。所恶于上，毋以使下，所恶于下，毋以事上；所恶于前，毋以先后；所恶于后，毋以从前；所恶于右，毋以交于左；所恶于左，毋以交于右；此之谓絜矩之道。《诗》云：“乐只君子，民之父母。”民之所好好之，民之所恶恶之，此之谓民之父母。《诗》云：“节彼南山，维石岩岩。赫赫师尹，民具尔瞻。”有国者不可以不慎，辟，则为天下僇矣。《诗》云：“殷之未丧师，克配上帝。仪监于殷，峻命不易。”道得众则得国，失众则失国。是故君子先慎乎德。有德此有人，有人此有土，有土此有财，有财此有用。德者本也，财者末也。外本内末，争民施夺。是故财聚则民散，财散则民聚。是故言悖而出者，亦悖而入；货悖而入者，亦悖而出。《康诰》曰：“惟命不于常。”道善则得之，不善则失之矣。《楚书》曰：“楚国无以为宝，惟善以为宝。”舅犯曰：“亡人无以为宝，仁亲以为宝。”《秦誓》曰：“若有一介臣，断断兮无他技，其心休休焉，其如有容焉。人之有技，若己有之；人之彦圣，其心好之，不啻若自其口出。实能容之，以能保我子孙黎民，尚亦有利哉！人之有技，媢疾以恶之；人之彦圣，而违之俾不通：实不能容，以不能保我子孙黎民，亦曰殆哉！”唯仁人放流之，迸诸四夷，不与同中国。此谓唯仁人为能爱人，能恶人。见贤而不能举，举而不能先，命也；见不善而不能退，退而不能远，过也。好人之所恶，恶人之所好，是谓拂人之性，菑必逮夫身。是故君子有大道，必忠信以得之，骄泰以失之。生财有大道，生之者众，食之者寡，为之者疾，用之者舒，则财恒足矣。仁者以财发身，不仁者以身发财。未有上好仁而下不好义者也，未有好义其事不终者也，未有府库财非其财者也。孟献子曰：“畜马乘，不察于鸡豚；伐冰之家，不畜牛羊；百乘之家，不畜聚敛之臣。与其有聚敛之臣，宁有盗臣。”此谓国不以利为利，以义为利也。长国家而务财用者，必自小人矣。彼为善之，小人之使为国家， 灾害并至。虽有善者，亦无如之何矣！此谓国不以利为利，以义为利也。
    '''
    content = trans_con(content)

    # # 去除content的数字
    # import re
    # content = re.sub(r'\d+', '', content).replace('\n', '')
    explain = '''
    《大学》的宗旨，在于弘扬高尚的德行，在于关爱人民，在于达到最高境界的善。知道要达到“至善”的境界方能确定目标，确定目标后方能心地宁静，心地宁静方能安稳不乱，安稳不乱方能思虑周详，思虑周详方能达到“至善”。凡物都有根本有末节，凡事都有终端有始端，知道了它们的先后次序，就与《大学》的宗旨相差不远了。
    在古代，意欲将高尚的德行弘扬于天下的人，则先要治理好自己的国家；意欲治理好自己国家的人，则先要调整好自己的家庭；意欲调整好自己家庭的人，则先要修养好自身的品德；意欲修养好自身品德的人，则先要端正自己的心意；意欲端正自己心意的人，则先要使自己的意念真诚；意欲使自己意念真诚的人，则先要获取知识；获取知识的途径则在于探究事理。探究事理后才能获得正确认识，认识正确后才能意念真诚，意念真诚后才能端正心意，心意端正后才能修养好品德，品德修养好后才能调整好家族，家族调整好后才能治理好国家，国家治理好后才能使天下太平。
    从天子到普通百姓，都要把修养品德作为根本。人的根本败坏了，末节反倒能调理好，这是不可能的。正像我厚待他人，他人反而慢待我；我慢待他人，他人反而厚待我这样的事情，还未曾有过。这就叫知道了根本，这就是认知的最高境界。
    所谓意念真诚，就是说不要自己欺骗自己。就像厌恶难闻的气味，喜爱好看的女子，这就是求得自己的心满意足。所以君子在独处时一定要慎重。小人在家闲居时什么坏事都可以做出来。当他们看到君子后，才会遮掩躲闪，藏匿他们的不良行为，表面上装作善良恭顺。别人看到你，就像能见到你的五脏六腑那样透彻，装模作样会有什么好处呢？这就是所说的心里是什么样的，会显露在外表上。因此，君子在独处的时候一定要慎重。曾子说：“一个人被众人注视，被众人指责，这是很可怕的啊！”富能使房屋华丽，德能使人品德高尚，心胸宽广能体态安适，所以，君子一定要意念真诚。
    《诗经》上说：“看那弯弯的淇水岸边，绿竹苍郁。那文质彬彬的君子，像切磋骨器、琢磨玉器那样治学修身。他庄重威严，光明显耀。那文质彬彬的君子啊，令人难以忘记！”所谓“像切磋骨器”，是说治学之道；所谓“像琢磨玉器”，是说自身的品德修养；所谓“庄重威严”，是说君子谦逊谨慎，所谓“光明显耀”，是说君子仪表的威严；“那文质彬彬的君子啊，令人难以忘记”，是说君子的品德完美，达到了最高境界的善，百姓自然不会忘记他。《诗经》上说：“哎呀，先前的贤王不会被人忘记。”后世君子，尊前代贤王之所尊，亲前代贤王之所亲，后代百姓因先前贤王而享安乐，获收益。这样前代贤王虽过世而不会被人遗忘。《尚书·周书》中的《康诰》篇上说：“能够弘扬美德。”《尚书·商书》中的《太甲》篇中说：“思念上天的高尚品德。”《尚书·虞书》中《帝典》篇中说：“能够弘扬伟大的德行。”这些都是说要自己发扬美德。商汤的《盘铭》上说：“如果一日洗刷干净了，就应该天天洗净，不间断。”《康诰》篇上说：“劝勉人们自新。”《诗经》上说：“周朝虽是旧国，但文王承受天命是新的。”
    因此，君子处处都要追求至善的境界。《诗经》上说：“京城方圆千里，都为百姓居住。”《诗经》上说：“啁啾鸣叫的黄莺，栖息在多树的山丘上。”孔子说：“啊呀，黄莺都知道自己的栖息之处，难道人反而不如鸟吗？”《诗经》上说：“仪态端庄美好的文王啊，他德行高尚，使人无不仰慕。”身为国君，当努力施仁政；身为下臣，当尊敬君主；身为人之子，当孝顺父母；身为人之父，当慈爱为怀；与国人交往，应当诚实，有信用。孔子说：“审断争讼，我的能力与他人的一般无二，但我力争使争讼根本就不发生。”违背实情的人，不能尽狡辩之能事，使民心敬畏。这叫做知道什么是根本。
    如要修养好品德，则先要端正心意。心中愤愤不平，则得不到端正；心中恐惧不安，则得不到端正；心里有偏好，则得不到端正；心里有忧患，则得不到端正。一旦心不在焉，就是看了，却什么也看不到；听了，却什么也听不到；吃了，却辨别不出味道。所以说，修养品德关键在端正心意。
    如要调整好家族，则先要修养好品德，为什么呢？因为人往往对他所亲近喜爱的人有偏见，对他所轻视讨厌的人有偏见，对他所畏惧恭敬的人有偏见，对他所怜惜同情的人有偏见，对他所傲视怠慢的人有偏见。所以喜爱一个人但又认识到他的缺点，不喜欢一个人但又认识到他优点的人，也少见。因此有一则谚语说：“人看不到自己孩子的过错，人察觉不到自己的庄稼好。”这就是不修养好品德，就调整不好家族的道理。
    要治理好国家，必须先要调整好自己的家族，因为不能教育好自己家族的人反而能教育好一国之民，这是从来不会有的事情。所以，君子不出家门而能施教于国民。孝顺，是侍奉君主的原则，尊兄，是侍奉长官的原则，仁慈，是控制民众的原则。《康诰》中说：“像爱护婴儿那样。”诚心诚意去爱护，即便不合乎婴儿的心意，也相差不远。不曾有过先学养育孩子再出嫁的人呀！一家仁爱相亲，一国就会仁爱成风；一家谦让相敬，一国就会谦让成风；一人贪婪暴戾，一国就会大乱——它们的相互关系就是这样。这就叫做一句话可以败坏大事，一个人可以决定国家。尧、舜用仁政统治天下，百姓就跟从他们实施仁爱。桀、纣用暴政统治天下，百姓就跟从他们残暴不仁。他们命令大家做的，与他自己所喜爱的凶暴相反，因此百姓不服从。因此，君子要求自己具有品德后再要求他人，自己先不做坏事，然后再要求他人不做。自己藏有不合“己所不欲，勿施于人”这一恕道的行为，却能使他人明白恕道，这是不会有的事情。因此，国家的治理，在于先调整好家族。《诗经》上说：“桃花绚烂，枝繁叶茂。姑娘出嫁，合家欢快。”只有合家相亲和睦后，才能够调教一国之民。《诗经》上说：“尊兄爱弟。”兄弟相处和睦后，才可以调教一国的人民。《诗经》上说：“他的仪容没有差错，成为四方之国的准则。”能使父亲、儿子、兄长、弟弟各谋其位，百姓才能效法。这就叫做治理好国家首先要调整好家族。
    要平定天下，先要治理好自己的国家。因为居上位的人敬重老人，百姓就会敬重老人；居上位的人敬重兄长，百姓就会敬重兄长，居上位的人怜爱孤小，百姓就不会不讲信义。所以，君子的言行具有模范作用。厌恶上级的所作所为，就不要用同样的做法对待下级；厌恶下级的所作所为，就不要用同样的做法对待上级；厌恶在我之前的人的所作所为，就不要用同样的做法对待在我之后的人，厌恶在我之后的人的所作所为，就不要用同样的做法对待在我之前的人，厌恶在我右边的人的所作所为，就不要用同样的方法与我左侧的人交往；厌恶在我左边的人的所作所为，就不要用同样的方法与我右侧的人交往。这就是所说的模范作用。《诗经》上说：“快乐啊国君，你是百姓的父母。”百姓喜爱的他就喜爱，百姓厌恶的他就厌恶，这就是所说的百姓的父母。《诗经》上说：“高高的南山啊，重峦叠嶂。光耀显赫的尹太师啊，众人都把你仰望。”统治国家的人不能不谨慎，出了差错就会被天下百姓杀掉。《诗经》上说：“殷朝没有丧失民众时，能够与上天的意旨相配合。应以殷朝的覆亡为鉴，天命得来不易啊。”这就是说得到民众的拥护，就会得到国家；失去民众的拥护，就会失去国家。
    所以，君子应该谨慎地修养德行。具备了德行才能获得民众，有了民众才会有国土，有了国土才会有财富，有了财富才能享用。德行为根本，财富为末端。如若本末倒置，民众就会互相争斗、抢夺。因此，财富聚集在国君手中，就可以使百姓离散，财富疏散给百姓，百姓就会聚在国君身边。所以你用不合情理的言语说别人，别人也会用不合情理的言语说你，用不合情理的方法获取的财富，也会被人用不合情理的方法夺走。《康诰》上说：“天命不是始终如一的。”德行好的就会得天命，德行不好就会失掉天命。《楚书》上说：“楚国没有什么可以当做珍宝的，只是把德行当做珍宝。”舅犯说：“流亡的人没有什么可以当做珍宝的，只是把挚爱亲人当做珍宝。”
    《秦誓》上说：“如果有这样一个大臣，他虽没有什么才能，但心地诚实宽大，能够容纳他人。别人有才能，如同他自己有一样；别人德才兼备，他诚心诚意喜欢，不只是口头上说说而已。能够留用这人，便能够保护我的子孙百姓。这对百姓是多么有利啊。如果别人有才能，就嫉妒厌恶；别人德才兼备，就阻拦他施展才干。不能留用这样的人，他不能保护我的子孙百姓，这种人也实在是危险啊。”只有仁德的人能把这种嫉妒贤人的人流放，驱逐到边远地区，使他们不能留在国家的中心地区。这叫做只有仁德的人能够爱人，能够恨人。看到贤人而不举荐，举荐了但不尽快使用，这是怠慢。看到不好的人却不能摈弃，摈弃了却不能放逐到远方，这是过错。喜欢人所厌恶的，厌恶人所喜欢的，这是违背了人性，灾害必然会降临到他的身上。因此，君子所有的高尚德行，一定要忠诚老实才能够获得，骄纵放肆便会失去。
    发财致富有这样一条原则：生产财富的人要多，消耗财富的人要少；干得要快，用得要慢，这样就可以永远保持富足了。有德行的人会舍财修身，没有德行的人会舍身求财。没有居上位的人喜爱仁慈而下位的人不喜爱忠义的；没有喜爱忠义而完不成自己事业的；没有国库里的财富最终不归属于国君的。孟献子说：“拥有一车四马的人，不应计较一鸡一猪的财物；卿大夫家不饲养牛羊；拥有马车百辆的人家，不豢养收敛财富的家臣。与其有聚敛民财的家臣，还不如有盗贼式的家臣。”这是说，国家不应把财物当做利益，而应把仁义作为利益。掌管国家大事的人只致力于财富的聚敛，这一定是来自小人的主张。假如认为这种做法是好的，小人被用来为国家服务，那么灾害就会一起来到，纵使有贤臣，也无济于事啊！这就是说国家不要把财利当做利益，而应把仁义当做利益。
    '''.replace('\n', '')
    implication = '''
    《大学》着重阐述了提高个人修养、培养良好的道德品质与治国平天下之间的重要关系。中心思想可以概括为“修己以安百姓”，并以三纲领“明明德、亲民、止于至善”和八条目“格物、致知、诚意、正心、修身、齐家、治国、平天下”为主题。
    《大学》提出的人生观与儒家思想有千丝万缕的联系，基本上是儒家人生观的进一步扩展。这种人生观要求注重个人修养，怀抱积极的奋斗目标，这一修养和要求是以儒家的道德观为主要内涵的。三纲八目又有阶级性， “明德”、“至善”都是封建主义对君主的政治要求和伦理标准；“格物”、“致知”等八条目是在修养问题上要求与三纲领中的政治理念和伦理思想相结合。
    《大学》还继承了孔子的仁政学说与孟子的民本论，《大学》里的统治者都是以“尊长”、“民之父母”的身份自居，但实际上他们还是站在剥削者的立场上这么说的，他们所谓的“爱民”、“不暴戾”只是为了维护他们上层建筑的经济基础——生产力。只有这样，他们无生产能力的剥削生活才能得以巩固。
    '''.replace('\n', '')
    key_words = '''
    大学之道|明明德|亲民|知止|静|安|虑|得|齐其家|修其身|致其知|格物|诚其意|益|中|外|严|润屋|润身|心广体胖|切、磋|琢、磨|克明峻德|自明|作|新民|其命|极|修身|好乐|亲爱|如保赤子|让|恕|夭夭|蓁蓁|宜|宜兄宜弟|正|法|絜矩之道|乐|岩岩|赫赫|克|配|峻|外本内末|道|絜|矩
    '''.replace('\n', '')
    key_words_imp = {
        "明明德": "第一个“明”是动词，彰显、发扬之意。第二个“明”是形容词，含有高尚、光辉的意思。",
        "静": "心不妄动。",
        "安": "所处而安。",
        "虑": "处事精详。",
        "修其身": "锻造、修炼自己的品行和人格。",
        "致其知": "让自己得到知识和智慧。",
        "格物": "研究、认识世间万物。",
        "诚其意": "指意念真诚。",
        "益": "益处，好处。",
        "严": "严峻，令人敬畏。",
        "润身": "修炼自己。",
        "心广体胖": "心胸宽广，身体舒适。",
        "胖": "舒适之意",
        "琢、磨": "雕琢打磨玉石。这里用来比喻研究学问，修养品德。",
        "克明峻德": "崇高之意。",
        "自明": "自己去发扬光明的德性。",
        "其命": "在这里指周朝所秉承的天命。",
        "极": "完善、极致。",
        "修身": "指的是修养良好的品德。",
        "好乐": "喜好，偏好。",
        "亲爱": "亲近、偏爱之意。",
        "让": "谦让，礼让。",
        "夭夭": "鲜美的样子。",
        "蓁蓁": "浓密茂盛的样子。",
        "宜": "适宜，和睦。",
        "宜兄宜弟": "是尊敬兄长、爱护兄弟之意。",
        "正": "匡正，教正。",
        "法": "效法。",
        "絜矩之道": "儒家的伦理思想，指一言一行要有模范作用。",
        "乐": "欢快、喜悦之意。",
        "岩岩": "险峻之意。",
        "赫赫": "显赫，显著的样子。",
        "克": "能够。",
        "配": "与……相符。",
        "峻": "大。",
        "道": "说。",
        "絜": "度量之意。",
        "矩": "画矩形所用的尺子，是规则、法度之意。"
    }

    tags = {"event": "大学之道", "event_time": "先秦", "event_location": "中国", "event_holiday": "无",
            "event_season": "无", "event_weather": "无", "sentiment": "积极",
            "metaphor": {"明明德": "第一个“明”是动词，彰显、发扬之意。第二个“明”是形容词，含有高尚、光辉的意思。",
                         "静": "心不妄动。", "安": "所处而安。", "虑": "处事精详。",
                         "修其身": "锻造、修炼自己的品行和人格。",
                         "致其知": "让自己得到知识和智慧。", "格物": "研究、认识世间万物。", "诚其意": "指意念真诚。",
                         "益": "益处，好处。", "严": "严峻，令人敬畏。", "润身": "修炼自己。",
                         "心广体胖": "心胸宽广，身体舒适。",
                         "胖": "舒适之意", "琢、磨": "雕琢打磨玉石。这里用来比喻研究学问，修养品德。",
                         "克明峻德": "崇高之意。",
                         "自明": "自己去发扬光明的德性。", "其命": "在这里指周朝所秉承的天命。", "极": "完善、极致。",
                         "修身": "指的是修养良好的品德。", "好乐": "喜好，偏好。", "亲爱": "亲近、偏爱之意。",
                         "让": "谦让，礼让。", "夭夭": "鲜美的样子。", "蓁蓁": "浓密茂盛的样子。", "宜": "适宜，和睦。",
                         "宜兄宜弟": "是尊敬兄长、爱护兄弟之意。", "正": "匡正，教正。", "法": "效法。",
                         "絜矩之道": "儒家的伦理思想，指一言一行要有模范作用。", "乐": "欢快、喜悦之意。",
                         "岩岩": "险峻之意。",
                         "赫赫": "显赫，显著的样子。", "克": "能够。", "配": "与……相符。", "峻": "大。", "道": "说。",
                         "絜": "度量之意。", "矩": "画矩形所用的尺子，是规则、法度之意。"}}
    df_ = pd.DataFrame([{
        'title': title,
        'dynasty': dynasty,
        'author': author,
        'content': content,
        'explain': explain,
        'implication': implication,
        'key_words': key_words,
        'key_words_imp': str(key_words_imp),
        'tags': str(tags)
    }])
    return df_


if __name__ == '__main__':
    date_ = ['0507', '0511', '0513']
    # date_ = ['0513']
    for d in date_:
        f_all = 'poems_all_v2_{}.csv'.format(d)
        merge_poems_all_files(f_v2_os, f_all.split('.')[0])  # 合并文件，并进行基础处理
        df = pd.read_csv(f_v2_os + f_all, low_memory=False)
        try:
            df.drop('tags_all', axis=1, inplace=True)
        except:
            pass
        try:
            df.drop('metaphor', axis=1, inplace=True)
        except:
            pass
        try:
            df.drop('merge_info', axis=1, inplace=True)
        except:
            pass
        try:
            df.drop('retrieval_info', axis=1, inplace=True)
        except:
            pass
        if d == '0507':
            df_dx = add_daxue()
            df = pd.concat([df, df_dx], axis=0, ignore_index=True)
            # df[df['title'] == '大学']
        '''
            0513之前的数据中，对解释有些未进行深度清洗，会有存在一些解释不对的情况，需要重新清洗
            0507:113367===》83518 ===》82858,
            0511:171455===》75332===》55451,
            0513:137566===》137541===》137522
        '''
        if d < '0513':
            df = drop_exp_noise(df)
        print(d, '：', df.shape)
        new_f_test = 'poems_all_v2_{}test.csv'.format(d)  # tags做合并处理
        build_db(df, f_v2_os + new_f_test)
    print('Done!')

    '''
        合并数据
        poems_all_v2_0513test.csv
        poems_all_v2_0511test.csv
        poems_all_v2_0507test.csv
        ->poems_all_v2.csv
    '''
    df_1 = pd.read_csv(f_v2_os + 'poems_all_v2_0513test.csv', low_memory=False)
    df_2 = pd.read_csv(f_v2_os + 'poems_all_v2_0511test.csv', low_memory=False)
    df_3 = pd.read_csv(f_v2_os + 'poems_all_v2_0507test.csv', low_memory=False)
    df_all = pd.concat([df_1, df_2, df_3], axis=0, ignore_index=True)
    df_all.drop_duplicates(subset=['content'], keep='first', inplace=True)
    # # explain
    # df_all.dropna(subset=['explain'], inplace=True)
    df_all.to_csv(f_v2_os + 'poems_all_v2.csv', index=False, encoding='utf-8')
    # df_all.to_csv(f_os_base + 'poems_all_v2.csv', index=False, encoding='utf-8')  # 178150
    # # 查找数据
    # df_all = pd.read_csv(f_os_base + 'poems_all_v2.csv', low_memory=False)
    # # 通过朝代和title筛出对应诗歌
    # df_tmp = df_all[(df_all['dynasty'] == '唐代') & (df_all['title'].str.contains('客至',regex=False))]
    # df_tmp = df_all[df_all['author'].str.contains('杜甫',regex=False)]

    # 汉字和诗词关联
    poemsLinkHanzi(hanzi_file=f_os_base + 'hanzi_all_simple.csv', poems_file=f_v2_os + 'poems_all_v2.csv',
                   new_poem_file=f_v3_os + 'poems_all_v3.csv', new_hp_file=param.hanzi_os + 'hanzi_poems_v3.csv')
    print('汉字和诗词关联Done!')

    # 五行关联
    addWuxingcol(poems_file=f_v3_os + 'poems_all_v3.csv')
    print('五行关联Done!')
