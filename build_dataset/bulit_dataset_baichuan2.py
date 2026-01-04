import torch
# seed = 4242
seed = 4240
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
import numpy as np

np.random.seed(seed)
import random

random.seed(seed)
torch.backends.cudnn.deterministic = True
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.utils import GenerationConfig
from tqdm import tqdm
import re
import pandas as pd

import json


def complete_implication_tags(df_poems):
    system_prompt = '''
     请参考示例，根据新任务中的古诗信息生成其对应的结果。
     其中“implication”表示古诗鉴赏，即对古诗中每句话的鉴赏以及整体的鉴赏；
     tags(标签)是一个字典，包括8项内容：{event(事件),event_time(事件时间),event_location(事件地点),event_holiday(事件节日),event_season(事件季节),event_weather(事件天气),sentiment(情感),metaphor(隐喻)}，其中metaphor(隐喻)是{"古诗中富有隐喻的关键词":"其相应的隐喻/寓意/情感"}的组合，请注意关键词不要超过20个。
     最终返回json格式，即:{"implication":"....","tags":{"event":"...","event_time":"...","event_location":"...","event_holiday":"...","event_season":"...","event_weather":"...","sentiment":"...","metaphor":{...}}}
     '''.replace('\n', '').replace(' ', '')
    shot1_1 = '''
     示例1：
     *古诗信息：
     {"title":"元日", "dynasty":"宋", "author","王安石", "content":"爆竹声中一岁除|春风送暖入屠苏|千门万户曈曈日|总把新桃换旧符","explain":"阵阵轰鸣的爆竹声中，旧的一年已经过去。和暖的春风吹来了新年，人们欢乐地畅饮着新酿的屠苏酒。初升的太阳照耀着千家万户，他们都忙着把旧的桃符取下，换上新的桃符。"}
     *生成结果（以json格式返回）：
     {"implication":"这首诗描写新年元日热闹、欢乐和万象更新的动人景象，抒发了作者革新政治的思想感情。‘爆竹声中一岁除’描述了农历新年到来；‘春风送暖入屠苏’描绘了春天的温暖气息；‘千门万户曈曈日’象征新的希望；‘总把新桃换旧符’达了人们对新一年的期待和祝福。整首诗充满了对新年的期待和祝福。","tags":{"event": "元日/春节", "event_time": "农历新年第一天", "event_location": "未知", "event_holiday": "春节", "event_season": "冬季", "event_weather": "温暖", "sentiment": "欢乐、期待、信心", "metaphor": {"爆竹":"春节辞旧迎新","屠苏":"喝酒庆祝新年，人们欢乐喜悦","曈曈日":"新生活的开始，变法带给百姓的是一片光明"}}}
     '''.replace('\n', '').replace(' ', '')
    shot1_2 = '''
    示例2：
    *古诗信息：
    {"title":"泊岳阳城下/泊岳阳楼下", "dynasty":"唐", "author","杜甫", "content":"江国逾千里|山城仅百层|岸风翻夕浪|舟雪洒寒灯|留滞才难尽|艰危气益增|图南未可料|变化有鲲鹏", "explain":"南国的江河众多，水程超过一千。岳阳城在巴陵山上，将近百层。 湖岸的风翻起晚浪，舟外的雪飘落灯前。 留滞他乡，有才无用，艰危时局，气节弥坚。 打算乘风破浪，放舟南下，说不定就像扶摇直上的九天鲲鹏。"}
    *生成结果（以json格式返回）：
    {"implication":"‘江国逾千里’描述了长江流域的广阔；‘山城仅百层’描绘了岳阳城的壮丽；‘岸风翻夕浪’形容傍晚时分江岸上的风吹起浪花；‘舟雪洒寒灯’描绘了船上的雪在灯光下的景象；‘留滞才难尽’表达了诗人滞留他乡的无奈；‘艰危气益增’则表达了诗人在艰难困苦中坚韧不屈的精神；‘图南未可料’表示对未来的迷茫和不确定；‘变化有鲲鹏’则借用神话传说中的鲲鹏，寓意诗人对变化的期待和希望。整首诗充满了诗人对生活的感慨和对未来的期待。", "tags":{"event": "泊舟", "event_time": "傍晚", "event_location": "岳阳城下/岳阳楼下", "event_holiday": "未知", "event_season": "冬季", "event_weather": "风、雪", "sentiment": "感慨、期待", "metaphor": {"江国":"广阔地域","舟雪":"艰难旅途","图南":"未来迷茫","鲲鹏":"代表希望与期待"}}}
    '''.replace('\n', '').replace(' ', '')

    # system_prompt = '''
    #     请参考示例，根据新任务中的古诗信息生成其对应的结果。
    #     最终返回json格式，即:{"implication":"....","tags":{"event":"...","event_time":"...","event_location":"...","event_holiday":"...","event_season":"...","event_weather":"...","sentiment":"...","metaphor":{...}}}。
    #     其中“implication”表示古诗鉴赏，即对古诗中每句话的鉴赏以及整体的鉴赏；tags(标签)是一个字典，包括8项内容：{event(事件),event_time(事件时间),event_location(事件地点),event_holiday(事件节日),event_season(事件季节),event_weather(事件天气),sentiment(情感),metaphor(隐喻)}，其中metaphor(隐喻)是{"古诗中富有隐喻的关键词":"其相应的隐喻/寓意/情感"}的组合，请注意关键词不要超过20个。
    #     '''.replace('\n', '').replace(' ', '')
    # shot1_1 = '''
    #     示例1：
    #     *古诗信息：
    #     {"title":"元日", "dynasty":"宋", "author","王安石", "content":"爆竹声中一岁除|春风送暖入屠苏|千门万户曈曈日|总把新桃换旧符","explain":"阵阵轰鸣的爆竹声中，旧的一年已经过去。和暖的春风吹来了新年，人们欢乐地畅饮着新酿的屠苏酒。初升的太阳照耀着千家万户，他们都忙着把旧的桃符取下，换上新的桃符。"}
    #     *生成结果（以json格式返回）：
    #     {"implication":"这首诗描写新年元日热闹、欢乐和万象更新的动人景象，抒发了作者革新政治的思想感情。‘爆竹声中一岁除’描述了农历新年到来；‘春风送暖入屠苏’描绘了春天的温暖气息；‘千门万户曈曈日’象征新的希望；‘总把新桃换旧符’达了人们对新一年的期待和祝福。整首诗充满了对新年的期待和祝福。","tags":{"event": "元日/春节", "event_time": "农历新年第一天", "event_location": "未知", "event_holiday": "春节", "event_season": "冬季", "event_weather": "温暖", "sentiment": "欢乐、期待、信心", "metaphor": {"爆竹":"春节辞旧迎新","屠苏":"喝酒庆祝新年，人们欢乐喜悦","曈曈日":"新生活的开始，变法带给百姓的是一片光明"}}}
    #     '''.replace('\n', '').replace(' ', '')
    # shot1_2 = '''
    #    示例2：
    #    *古诗信息：
    #    {"title":"泊岳阳城下/泊岳阳楼下", "dynasty":"唐", "author","杜甫", "content":"江国逾千里|山城仅百层|岸风翻夕浪|舟雪洒寒灯|留滞才难尽|艰危气益增|图南未可料|变化有鲲鹏", "explain":"南国的江河众多，水程超过一千。岳阳城在巴陵山上，将近百层。 湖岸的风翻起晚浪，舟外的雪飘落灯前。 留滞他乡，有才无用，艰危时局，气节弥坚。 打算乘风破浪，放舟南下，说不定就像扶摇直上的九天鲲鹏。"}
    #    *生成结果（以json格式返回）：
    #    {"implication":"‘江国逾千里’描述了长江流域的广阔；‘山城仅百层’描绘了岳阳城的壮丽；‘岸风翻夕浪’形容傍晚时分江岸上的风吹起浪花；‘舟雪洒寒灯’描绘了船上的雪在灯光下的景象；‘留滞才难尽’表达了诗人滞留他乡的无奈；‘艰危气益增’则表达了诗人在艰难困苦中坚韧不屈的精神；‘图南未可料’表示对未来的迷茫和不确定；‘变化有鲲鹏’则借用神话传说中的鲲鹏，寓意诗人对变化的期待和希望。整首诗充满了诗人对生活的感慨和对未来的期待。", "tags":{"event": "泊舟", "event_time": "傍晚", "event_location": "岳阳城下/岳阳楼下", "event_holiday": "未知", "event_season": "冬季", "event_weather": "风、雪", "sentiment": "感慨、期待", "metaphor": {"江国":"广阔地域","舟雪":"艰难旅途","图南":"未来迷茫","鲲鹏":"代表希望与期待"}}}
    #    '''.replace('\n', '').replace(' ', '')

    # # 筛选出已经包含explain和implication的古诗
    # df_poems = df_poems[df_poems['explain'].notna() & df_poems['implication'].notna()]
    # 创建一个新的df_tmp
    f_os = './poems_all_v2_0513_'
    fw_os = './poems_all_v2_wrong_0513_'
    df_tmp = pd.DataFrame(columns=df_poems.columns)
    df_tmp.to_csv(f_os + f'{run_num[0]}_{run_num[1]}.csv', index=False)
    # columns = ['title', 'dynasty', 'author', 'content', 'explain', 'implication', 'metaphor', 'tags']
    columns = df_poems.columns
    df_tmp_wrong = pd.DataFrame(columns=columns)
    df_tmp_wrong.to_csv(fw_os + f'{run_num[0]}_{run_num[1]}.csv', index=False)
    cnt = 0
    cnt_wrong = 0
    # 取出需要处理的古诗数量
    df_poems_tmp = df_poems[run_num[0]:run_num[1]]
    # 遍历df_poems提取古诗信息
    for index, row in tqdm(df_poems_tmp.iterrows(), total=df_poems_tmp.shape[0]):
        # if index == 6:
        #     exit(0)
        if index < run_num[0]:
            continue
        elif index > run_num[1]:
            exit(0)
        title = row['title']
        dynasty = row['dynasty']
        author = row['author']
        content = row['content']
        explain = row['explain']

        query1 = f'''
        新任务：
        *新的古诗信息:
        {{"title":"{title}", "dynasty":"{dynasty}", "author","{author}", "content":"{content}", "explain":"{explain}"}}
        *生成结果（以json格式返回）：
        '''.replace('\n', '').replace(' ', '')
        prompt = system_prompt + '\n' + shot1_1 + '\n' + shot1_2 + '\n' + query1
        # prompt = shot1_1 + '\n' + shot1_2 + '\n' + system_prompt + '\n' + query1
        # prompt = shot1_1 + '\n' + system_prompt + '\n' + query1
        messages = []
        messages.append({"role": "user", "content": f"{prompt}"})
        # print(f'prompt: {prompt}')
        try_num = 0
        response = model.chat(tokenizer, messages)
        response = response.replace('\n', '').replace('\t', '').replace(' ', '').replace('，', ',').replace('：', ':')
        # response = response[response.index('{"implication"'):response.rindex('}') + 1]
        try:
            response = re.sub(r'implication":"(.*?)"\s*,"tags',
                              lambda m: r'implication":"{}","tags'.format(m.group(1).replace('"', r'\"')), response)

            response = response[:response.rindex('"') + 1] + '}}}'
            res_dict = eval(response)
            # print(f'=======================================response: {response}')
            # 保存结果
            df_tmp.loc[index] = row
            df_tmp.loc[index, 'implication'] = res_dict['implication']
            df_tmp.loc[index, 'tags'] = str(res_dict['tags'])
            if cnt % 100 == 0:
                df_tmp.to_csv(f_os + f'{run_num[0]}_{run_num[1]}.csv', mode='a', index=False,
                              header=False)
                df_tmp = pd.DataFrame(columns=columns)
            cnt += 1
        except:
            print(f'index: {index}, response: {response}')
            df_tmp_wrong.loc[index] = row
            if cnt_wrong % 100 == 0:
                df_tmp_wrong.to_csv(fw_os + f'{run_num[0]}_{run_num[1]}.csv', mode='a', index=False,
                                    header=False)
                df_tmp_wrong = pd.DataFrame(columns=columns)
            cnt_wrong += 1

    df_tmp.to_csv(f_os + f'{run_num[0]}_{run_num[1]}.csv', mode='a', index=False, header=False)
    df_tmp_wrong.to_csv(fw_os + f'{run_num[0]}_{run_num[1]}.csv', mode='a', index=False, header=False)
    print('通过的个数：', cnt)
    print('错误的个数：', cnt_wrong)
    # return response


if __name__ == '__main__':
    server = 'zsl'  # b101/zsl  # TODO: 修改server, TODO: 修改sftp.json
    model_size = 7  # 7/13, 13B
    if model_size == 7:
        if server == 'b101':
            model_path = "/nfs-data/user14/cache_huggingface_hub/hub/models--baichuan-inc--Baichuan2-7B-Chat/snapshots/ea66ced17780ca3db39bc9f8aa601d8463db3da5"
        else:
            model_path = "/home/zsl/audrey_code/AI_name/AI_name/Baichuan2-7B-Chat/models--baichuan-inc--Baichuan2-7B-Chat/snapshots/ea66ced17780ca3db39bc9f8aa601d8463db3da5"
    else:  # 13
        if server == 'b101':
            model_path = "/nfs-data/user14/cache_huggingface_hub/hub/models--baichuan-inc--Baichuan2-13B-Chat/snapshots/c8d877c7ca596d9aeff429d43bff06e288684f45"
        else:
            model_path = "/home/zsl/audrey_code/AI_name/AI_name/Baichuan2-13B-Chat/models--baichuan-inc--Baichuan2-13B-Chat/snapshots/c8d877c7ca596d9aeff429d43bff06e288684f45"
    tokenizer = AutoTokenizer.from_pretrained(model_path,
                                              revision="v2.0",
                                              use_fast=False,
                                              trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_path,
                                                 revision="v2.0",
                                                 device_map="auto",
                                                 torch_dtype=torch.bfloat16,
                                                 trust_remote_code=True)
    model.generation_config = GenerationConfig.from_pretrained(model_path, revision="v2.0")
    # model = model.cuda()
    if server == 'b101':
        # poems_file = './poems_All_0504_exp_imp.csv'
        poems_file = './poems_All_0504_exp.csv'
    else:
        # poems_file = '/home/zsl/audrey_code/AI_Naming/dataset/poems_All_0504_exp_imp.csv'
        poems_file = '/home/zsl/audrey_code/AI_Naming/dataset/poems_All_0504_exp.csv'
        # poems_file = '/home/zsl/audrey_code/AI_Naming/build_dataset/poems_all_v2.csv'

    df_poems = pd.read_csv(poems_file, low_memory=False)  # 读取数据
    # poems_All_0504_exp_imp.csv
    # run_num = [0,100000]  # 修改run_num, 共171451
    # run_num = [100000, 150000]
    # run_num = [150000, 180000]
    # simplify_implication_complete_tags(df_poems)  # 对已经包含explain和implication的古诗,简化implication,补充tags

    # poems_All_0504_exp.csv'
    # run_num = [0, 10]  # TODO: 修改run_num, 共152813
    # run_num = [0, 60000]
    # run_num = [60000, 120000]
    # run_num = [120000, 160000]
    run_num = [140000, 160000]
    # run_num = [160000, 180000]
    print(f'run_num: {run_num}')
    complete_implication_tags(df_poems)  # 对已经包含explain的古诗, 补充implication, tags

    # complete_poem_all(df_poems)
    # poems = None
    # simplify_implication(poem)
    # complete_implication(poem)
    # complete_all(poems)
