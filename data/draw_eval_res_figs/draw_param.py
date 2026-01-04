import matplotlib.pyplot as plt
import pandas as pd

# methods_order_all = ['Base', 'CoT', 'TDB', 'Few-shot', 'Q2Kw', 'LLM Discussion', 'BabyNamer']
methods_order_all = ['Base', 'CoT', 'TDB', 'Few-shot', 'Q2Kw', 'LLM-D', 'NAMeGEn']


class DrawParam:
    color_list = plt.cm.tab20
    methods_= methods_order_all
    # backbones_ = ['Baichuan', 'Qwen', 'GLM-4', 'DeepSeek', 'Mistral', 'Gemini', 'GPT4o']
    backbones_ = ['Qwen', 'GLM-4', 'DeepSeek', 'Mistral', 'Gemini', 'GPT4o']
    # backbones_order = ['Qwen', 'GLM-4', 'DeepSeek', 'Mistral', 'Gemini']
    backbone_color_map = {
        'Qwen': color_list(0),
        'GLM-4': color_list(2),
        'DeepSeek': color_list(8),
        'Mistral': color_list(6),
        'Gemini': color_list(4),
        'GPT4o': color_list(12),
    }
    # 0: 蓝，2：橙，4：绿，6：红，8：紫，10：棕，12：粉
    # 'Qwen': color_list(2),
    # 'DeepSeek': color_list(4),
    # 'GLM-4': color_list(8),
    # 'Mistral': color_list(12),
    # 'Gemini': color_list(0),
    # 'GPT4o': color_list(6),
    methods_color_map = {
        'Base': color_list(0),
        'CoT': color_list(2),
        'TDB': color_list(4),
        'Few-shot': color_list(6),
        'Q2Kw': color_list(8),
        'LLM Discussion': color_list(10),
        'BabyNamer': color_list(12)
    }

    def pre_process_Upper(self, df_):
        # 找出对比实验的数据
        df_new = df_[df_['method'].isin(['base', 'CoT', 'TDB', 'fewshot', 'query2keyword', 'llm_discussion', 'magic_moo'])]
        # 对backbone名做修改
        df_new['backbone'] = df_new['backbone'].apply(
            lambda x: x.title() if x in ['baichuan', 'qwen', 'mistral', 'gemini'] else x)  # 首字母大写
        df_new['backbone'] = df_new['backbone'].apply(lambda x: 'GLM-4' if x in ['glm4'] else x)
        df_new['backbone'] = df_new['backbone'].apply(lambda x: 'GPT4o' if x in ['gpt4o'] else x)
        df_new.loc[:, 'backbone'] = df_new['backbone'].apply(lambda x: 'DeepSeek' if x == 'deepseek' else x)
        return df_new
    def pre_process(self, df_):
        # 预处理
        try:
            df2 = self.pre_process_Upper(df_)
        except:
            pass
        # 对方法名做修改
        df2.loc[:, 'method'] = df2['method'].apply(lambda x: 'Base' if x == 'base' else x)
        df2.loc[:,'method'] = df2['method'].apply(lambda x: 'Few-shot' if x == 'fewshot' else x)
        try:
            df2.loc[:,'method'] = df2['method'].apply(lambda x: 'Q2Kw' if x == 'query2keyword' else x)
        except:
            pass
        df2.loc[:,'method'] = df2['method'].apply(lambda x: 'LLM-D' if x == 'llm_discussion' else x)
        # df2.loc[:,'method'] = df2['method'].apply(lambda x: 'BabyNamer' if x == 'magic_moo' else x)
        df2.loc[:,'method'] = df2['method'].apply(lambda x: 'NAMeGEn' if x == 'magic_moo' else x)
        # 重新按照method_order排序
        method_order = methods_order_all
        df2['method'] = pd.Categorical(df2['method'], categories=method_order, ordered=True)
        df2.sort_values(by=['backbone', 'method'], inplace=True)
        return df2