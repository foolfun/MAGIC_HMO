'''
从古诗句网爬取数据
https://www.gushiju.net/shici
'''

# 从https://www.gushiju.net/shici这个网站爬取数据
from bs4 import BeautifulSoup
from selenium import webdriver
from tqdm import tqdm
import pandas as pd
from utils.base_param import BaseParam

params = BaseParam()
f_web = params.web_data_os
# 这些始终要用到，作为全局，可以加快code运行速度
chrome_options = webdriver.ChromeOptions()
# 使用headless无界面浏览器模式
chrome_options.add_argument('--headless')  # 增加无界面选项
chrome_options.add_argument('--disable-gpu')  # 如果不加这个选项，有时定位会出现问题
# 1\启动浏览器
driver = webdriver.Chrome(options=chrome_options)
# 隐式等待
driver.implicitly_wait(10)
base_link = 'https://www.gushiju.net/'


# # 防止被识别， 设置随机等待秒数
# rand_seconds = random.choice([1, 3]) + random.random()
def get_soup(url):  # 给一个链接返回封装之后的结果
    driver.get(url)
    content = driver.page_source
    soup = BeautifulSoup(content, 'lxml')
    return soup


def get_poem_link():
    page_urls = ['https://www.gushiju.net/shici/chaodai-%E4%B8%8D%E9%99%90/p-' + str(i) for i in range(1, 13248)]
    poem_info = []
    df = pd.DataFrame(poem_info, columns=['title', 'dynasty', 'author', 'poem_link'])
    df.to_csv(f_web + 'gushiju_poem_links0427.csv', index=False)
    for p_url in tqdm(page_urls):
        # p_url = page_urls[0]
        soup = get_soup(p_url)
        page_poems = soup.find_all('div', 'item-shici')
        poem_info = []
        for poem in page_poems:
            # poem=page_poems[0]
            try:
                ind = poem.find('a').text.rindex("》")
                ti = poem.find('a').text[:ind + 1]
                ti = ti.replace('《', '').replace('》', '')
                da_text = poem.find('a').text[ind + 2:].strip()
                ind_p = da_text.index('·')
                dy = da_text[:ind_p].strip()
                au = da_text[ind_p + 1:].strip()
                poem_link = base_link + poem.find('a')['href']
            except:
                print(poem.find('a').text)
                continue

            poem_info.append([ti, dy, au, poem_link])
            df_tmp = pd.DataFrame(poem_info, columns=['title', 'dynasty', 'author', 'poem_link'])
        # 每读一页在文件后面append一次数据
        df_tmp.to_csv(f_web + 'gushiju_poem_links0427.csv', mode='a', header=False, index=False)


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


def get_detail(gsh, links_df, flag_new=0):
    p_data = []
    if flag_new:
        # 新建
        con_df = pd.DataFrame(p_data,
                              columns=['title', 'dynasty', 'author', 'content', 'explain', 'implication',
                                       'key_words',
                                       'key_words_imp', 'tags'])
        con_df.to_csv(f_web + gsh, index=False)
    cnt = 0
    n = 17
    base = 91000
    start_n = base + 10000 * n
    end_n = min(base + 10000 * (n+1), links_df.shape[0])
    # start_n = 0
    # end_n = links_df.shape[0]
    print(n, start_n, end_n)
    for i in tqdm(links_df.iloc[start_n:end_n].values):
        cnt += 1
        title = None
        dynasty = None
        author = None
        content = None
        explain = None
        implication = None
        key_words = None
        key_words_imp = None
        tags = get_tags()
        title, dynasty, author, poem_link = i
        # if title !='木兰诗/木兰辞': # test
        #     continue
        soup_inner = get_soup(poem_link)
        # soup_inner = get_soup('https://www.gushiju.net/shici/1')  # test
        # title = '关雎'
        # article = soup_inner.find('article', 'col-md-8')
        # 获取诗句内容
        try:
            content = soup_inner.find('div', 'shici-text').text.strip()
        except:
            print(i)
            continue
        # 获取翻译及注释
        ziliao_contents = soup_inner.find('div', 'shici-ziliao').find_all('div', 'ziliao')
        if len(ziliao_contents) != 0:
            for zc in ziliao_contents:
                # print('=====', i)
                cont = zc.text
                flag_fz = [cont.find(title + '翻译及注释'), cont.find(title + '翻译')]
                if sum(flag_fz) > -2:
                    if flag_fz[0] != -1 or flag_fz[1] != -1:  # 有翻译
                        explain = cont.rsplit('翻译', 1)[1]
                        if cont.rfind('注释') != -1:  # 去掉注释
                            explain = explain.rsplit('注释', 1)[0]
                    continue  # 找到翻译就结束

                if cont.find(title + '赏析') != -1:
                    implication = cont.rsplit('赏析', 1)[1].strip()
                    break  # 找到赏析就结束
                elif cont.find(title + '鉴赏') != -1:
                    implication = cont.rsplit('鉴赏', 1)[1]
                    break  # 找到鉴赏就结束
        else:
            ziliao_contents = soup_inner.find('div', 'shici-ziliao').text.strip()
            ind1 = ziliao_contents.find('\n诗意和赏析：')
            ind2 = ziliao_contents.find('\n诗意：')
            ind3 = ziliao_contents.find('\n赏析：')
            ind = -1
            if ind1 != -1:
                ind = ind1
            elif ind2 != -1:
                ind = ind2
            elif ind3 != -1:
                ind = ind3
            if ind != -1:
                # s_ind = ziliao_contents.find('\n')
                explain = ziliao_contents[:ind].strip()
                implication = ziliao_contents[ind:].strip()
            else:
                explain = ziliao_contents

        p_data.append([title, dynasty, author, content, explain, implication, key_words, key_words_imp, tags])

        # 保存数据
        if cnt % 2000 == 0:
            df_ = pd.DataFrame(p_data,
                               columns=['title', 'dynasty', 'author', 'content', 'explain', 'implication',
                                        'key_words',
                                        'key_words_imp', 'tags'])
            df_.to_csv(f_web + gsh, mode='a', header=False, index=False)
            p_data = []
            print('已保存{0}条数据'.format(cnt))
    df_ = pd.DataFrame(p_data,
                       columns=['title', 'dynasty', 'author', 'content', 'explain', 'implication', 'key_words',
                                'key_words_imp', 'tags'])
    df_.to_csv(f_web + gsh, mode='a', header=False, index=False)
    print('save！done！')


if __name__ == '__main__':
    step1 = 0
    step2 = 1

    if step1:  # 从gushiju网站爬取link数据
        get_poem_link()
        driver.quit()
        print('links done')
        # df1 = pd.read_csv(f_web + 'gushiju_poem_links0427_13247.csv')
        # df2 = pd.read_csv(f_web + 'gushiju_poem_links0427_6624.csv')
        # df1.columns= ['title', 'dynasty', 'author', 'poem_link']
        # df2.columns= ['title', 'dynasty', 'author', 'poem_link']
        # df = pd.concat([df1, df2], ignore_index=True)
        # df_ = df.drop_duplicates(subset=['poem_link'], keep='first')
        # df.to_csv(f_web + 'gushiju_poem_links0427.csv', index=False)
    if step2:
        # 从gushiju_poem_link.csv读取数据
        links_df = pd.read_csv(f_web + 'gushiju_poem_links0427.csv')
        gsh = 'gushiju_poem_raw0427.csv'
        flag_new = 0
        get_detail(gsh=gsh, links_df=links_df, flag_new=flag_new)
