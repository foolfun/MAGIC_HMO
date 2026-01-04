# 重新导入所需库和数据（由于代码执行状态已重置）
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

# 构建数据
data = {
    'method': ['TDB', 'Few-shot', 'LLM Discussion', 'BabyNamer'],
    'o1_human': [38.89, 52.67, 40.89, 84.22],
    'o2_human': [51.11, 88.89, 50.00, 93.56],
    'o3_human': [65.78, 72.22, 49.11, 98.45],
    'o4_human': [39.56, 91.33, 31.56, 97.33],
    'crc_human': [46.00, 98.89, 39.33, 100.0],
    'lc_human': [50.67, 48.22, 50.22, 80.67],
    'o1_llm': [83.20, 90.07, 85.60, 96.93],
    'o2_llm': [97.27, 99.67, 96.40, 99.13],
    'o3_llm': [91.27, 87.07, 83.60, 93.67],
    'o4_llm': [67.93, 89.07, 62.60, 92.33],
    'crc_llm': [86.40, 90.07, 77.69, 96.36],
    'lc_llm': [82.68, 78.73, 82.87, 87.19]
}

df = pd.DataFrame(data)

# 展平数据为整体散点格式
indicators = ['o1', 'o2', 'o3', 'o4', 'crc', 'lc']
scatter_data = []

for i in range(len(df)):
    for ind in indicators:
        scatter_data.append({
            'Human Score': df.loc[i, f'{ind}_human'],
            'LLM Score': df.loc[i, f'{ind}_llm']
        })

scatter_df = pd.DataFrame(scatter_data)

# 计算整体皮尔逊相关系数
pearson_corr_all, p_value_all = pearsonr(scatter_df['LLM Score'], scatter_df['Human Score'])

# 绘图
fig = plt.figure(figsize=(12, 8.5))
sns.set(style="whitegrid")

sns.scatterplot(data=scatter_df, x='LLM Score', y='Human Score', s=550, color='#FF7F0E', alpha=0.7)
sns.regplot(x='LLM Score', y='Human Score', data=scatter_df, scatter=False, color='gray', ci=None)

plt.title(f"LLM vs. Human Evaluation Scores\nPearson r = {pearson_corr_all:.2f}, p = {p_value_all:.4f}",
          fontsize=46,pad=18, fontweight='bold', fontname='Microsoft YaHei')
plt.xlabel("LLM Score", fontsize=40, labelpad=12)
plt.ylabel("Human Score", fontsize=40, labelpad=-13)
plt.xticks(fontsize=35)
plt.yticks(fontsize=35)
plt.grid(True, linestyle='--')
# plt.tight_layout()
plt.subplots_adjust(left=0.12, right=0.95, top=0.8, bottom=0.15)
plt.show()

from matplotlib.backends.backend_pdf import PdfPages
from tools.base_param import BaseParam
params = BaseParam()
# 保存pdf
pdf_pages = PdfPages(params.f_draw_pdf_os + 'fig_c6_human.pdf')
pdf_pages.savefig(fig)
pdf_pages.close()