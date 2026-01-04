'''
info-process
将出生年月
转为对应的生肖、八字、季节、节日、节气
https://github.com/OPN48/cnlunar
'''
import datetime
import cnlunar
from collections import Counter


class GetMoreInfo:
    def __init__(self, birth, season=None):
        self.birth = birth.split('-')
        # 将从出生日期抽取出年、月、日、时、分、秒
        self.year = int(self.birth[0])
        self.month = int(self.birth[1])
        self.day = int(self.birth[2])
        self.hour = int(self.birth[3])
        self.minute = int(self.birth[4])
        self.second = int(self.birth[5])
        self.season = season
        # 如果没有年月日，就默认为2024年1月1日
        year = 2024 if (self.year == 0) else self.year
        month = 1 if (self.month == 0) else self.month
        day = 1 if (self.day == 0) else self.day
        self.cn = cnlunar.Lunar(datetime.datetime(year, month, day, self.hour, self.minute, self.second)
                                , godType='8char')

    def birth_to_info(self):
        dic = {
            '日期': self.cn.date,
            '农历数字': (
                self.cn.lunarYear, self.cn.lunarMonth, self.cn.lunarDay, '闰' if self.cn.isLunarLeapMonth else ''),
            '农历': '%s %s[%s]年 %s%s' % (
                self.cn.lunarYearCn, self.cn.year8Char, self.cn.chineseYearZodiac, self.cn.lunarMonthCn,
                self.cn.lunarDayCn),
            '生肖': self.cn.chineseYearZodiac,
            '星期': self.cn.weekDayCn,
            '季节': self.cn.lunarSeason,
            '八字': ' '.join([self.cn.year8Char, self.cn.month8Char, self.cn.day8Char, self.cn.twohour8Char]),
            '星座': self.cn.starZodiac,
            # '今日节日': self.cn.get_legalHolidays() + self.cn.get_otherHolidays() + self.cn.get_otherLunarHolidays(),
            '今日节日': self.cn.get_legalHolidays() + '，' + self.cn.get_otherHolidays(),
            # '今年节气表': self.cn.thisYearSolarTermsDic,
            # '今日节日': self.cn.get_otherLunarHolidays(),
            '今日节气': self.cn.todaySolarTerms,
            '今日时辰': self.cn.twohour8CharList,
            '时辰凶吉': self.cn.get_twohourLuckyList(),
            '生肖冲煞': self.cn.chineseZodiacClash,

            '星次': self.cn.todayEastZodiac,

            '彭祖百忌': self.cn.get_pengTaboo(),
            '彭祖百忌精简': self.cn.get_pengTaboo(long=4, delimit='<br>'),
            '十二神': self.cn.get_today12DayOfficer(),
            '廿八宿': self.cn.get_the28Stars(),

            '今日三合': self.cn.zodiacMark3List,
            '今日六合': self.cn.zodiacMark6,
            '今日五行': self.cn.get_today5Elements(),

            '纳音': self.cn.get_nayin(),
            '九宫飞星': self.cn.get_the9FlyStar(),
            '吉神方位': self.cn.get_luckyGodsDirection(),
            '今日胎神': self.cn.get_fetalGod(),
            '神煞宜忌': self.cn.angelDemon,
            '今日吉神': self.cn.goodGodName,
            '今日凶煞': self.cn.badGodName,
            '宜忌等第': self.cn.todayLevelName,
            '宜': self.cn.goodThing,
            '忌': self.cn.badThing,
            '时辰经络': self.cn.meridians
        }
        # for i in dic:
        #     midstr = '\t' * (2 - len(i) // 2) + ':' + '\t'
        #     print(i, midstr, dic[i])
        return dic

    def get_naming_radicals(self, zodiac):
        zodiac_radicals = {
            "鼠": {
                "good": ["宀", "米", "豆", "禾", "艹"],
                "bad": ["午", "马", "火", "羊", "犭"]
            },
            "牛": {
                "good": ["艹", "辶", "车", "田", "禾"],
                "bad": ["马", "羊", "礻", "纟", "示"]
            },
            "虎": {
                "good": ["山", "王", "令", "火", "礻"],
                "bad": ["辶", "廴", "人", "门", '虎']
            },
            "兔": {
                "good": ["口", "艹", "宀", "豆", "米"],
                "bad": ["亻", "几", "忄", "日 ", "王"]
            },
            "龙": {
                "good": ["长", "申", "王", "大", "马"],
                "bad": ["人", "田", "口", "羊", "犭"],
            },
            "蛇": {
                "good": ["口", "宀", "冖", "虫", "几"],
                "bad": ["艹", "水", "人", "豆", "米"],
            },
            "马": {
                "good": ["艹", "采", "麦", "叔", "禾"],
                "bad": ["田", "口", "米", "山", "奇"],
            },
            "羊": {
                "good": ["几", "卯", "门", "足", "叔"],
                "bad": ["心", "彡", "纟", "车", "山"],
            },
            "猴": {
                "good": ["木", "宀", "冖", "言", "巾"],
                "bad": ["金", "酉", "西", "月", "禾"],
            },
            "鸡": {
                "good": ["米", "豆", "山", "彡", "纟"],
                "bad": ["东", "月", "兔", "心", "小"],
            },
            "狗": {
                "good": ["人", "入", "巾", "宀", "主"],
                "bad": ["禾", "米", "豆", "麦", "口"],
            },
            "猪": {
                "good": ["米", "豆", "禾", "宀", "冖"],
                "bad": ["辶", "廴", "川", "一", "王"],
            }
        }

        if zodiac in zodiac_radicals:
            # print(zodiac_radicals[zodiac])
            return zodiac_radicals[zodiac]
        else:
            return {"error": "Invalid zodiac sign"}

    def trans_season(self, season):
        if "春" in season:
            return '春季'
        elif "夏" in season:
            return '夏季'
        elif "秋" in season:
            return '秋季'
        elif "冬" in season:
            return '冬季'

    def get_baby_info(self):
        dic = self.birth_to_info()
        lunar_day = dic['农历']
        shengxiao = dic['生肖']
        season = self.trans_season(dic['季节'])
        temp = '，'.join([item for item in dic['今日节日'].split('，') if item])
        holiday = temp if temp != '' else '无'
        solar_term = dic['今日节气']
        bazi = dic['八字']
        if self.year == 0 or self.month == 0 or self.day == 0:
            lunar_day = '无'
            shengxiao = '无'
            season = self.season if (self.season is not None) else '无'
            holiday = '无'
            solar_term = '无'
            bazi = '无'
        dic_new = {
            '农历': lunar_day,
            '生肖': shengxiao,
            # '属'+dic['生肖']+'的取名推荐部首': self.get_naming_radicals(dic['生肖'])['good'],
            # '推荐部首': self.get_naming_radicals(dic['生肖'])['good'],
            # '属'+dic['生肖']+'的取名忌用部首': self.get_naming_radicals(dic['生肖'])['bad'],
            # '忌用部首': self.get_naming_radicals(dic['生肖'])['bad'],
            '季节': season,
            '节日': holiday,
            '节气': solar_term,
            '八字': bazi
        }
        return dic_new

    def get_baby_info_new(self):
        dic = self.birth_to_info()
        lunar_day = dic['农历']
        shengxiao = dic['生肖']
        season = self.trans_season(dic['季节'])
        temp = '，'.join([item for item in dic['今日节日'].split('，') if item])
        holiday = temp if temp != '' else '无'
        solar_term = dic['今日节气']
        bw, lack_wx = self.get_wuxing(dic['八字'])
        # lack_wx = '、'.join(lack_wx) if lack_wx != [] else '' # 空代表五行齐全
        if self.year == 0 or self.month == 0 or self.day == 0:
            lunar_day = '无'
            # shengxiao = '无'
            if self.month == 0:
                season = self.season if (self.season is not None) else '无'
            holiday = '无'
            solar_term = '无'
            bw = '无'
            lack_wx = []
        dic_new = {
            '农历': lunar_day,
            '生肖': shengxiao,
            '季节': season,
            '节日': holiday,
            '节气': solar_term,
            '八字和五行': bw,
            '五行缺失': lack_wx
        }
        return dic_new

    def get_wuxing(self, bazi):
        # 计算缺失五行的方法
        # 1\排出八字：
        gz_wu_xing = {
            '甲': '木',
            '乙': '木',
            '丙': '火',
            '丁': '火',
            '戊': '土',
            '己': '土',
            '庚': '金',
            '辛': '金',
            '壬': '水',
            '癸': '水',
            '子': '水',
            '丑': '土',
            '寅': '木',
            '卯': '木',
            '辰': '土',
            '巳': '火',
            '午': '火',
            '未': '土',
            '申': '金',
            '酉': '金',
            '戌': '土',
            '亥': '水',
        }
        wuxing = ['木', '火', '土', '金', '水']
        bazi = list(bazi.replace(' ', ''))
        # 2\分析五行属性：
        bazi_wuxing = [gz_wu_xing[bazi[i]] for i in range(0, 8)]
        bw_s = ''
        for i in range(0, 8, 2):
            bw_s += bazi[i] + bazi[i + 1] + '(' + bazi_wuxing[i] + bazi_wuxing[i + 1] + ')' + ' '
        bw_s = bw_s.strip()
        # print(bazi_wuxing)
        # 3\计算五行旺衰：接下来，要根据八字中五行的分布情况，判断哪种五行最为旺盛，哪种五行相对较弱。这一步通常涉及到对八字中各五行力量的平衡与对比。
        bw_c = dict(Counter(bazi_wuxing))
        # print(bw_c)
        # 4\计算缺失五行：最后，根据五行的旺衰情况，判断八字中哪种五行的力量最为不足，即为缺失五行。
        lack_wuxing = []
        for i in wuxing:
            if i not in bw_c.keys():
                lack_wuxing.append(i)
        return bw_s, lack_wuxing


if __name__ == '__main__':
    # info = Get_more_info('2024-01-22')
    # info = GetMoreInfo('2024-02-09')
    # info = GetMoreInfo('2023-12-09-00-00-00')
    info = GetMoreInfo('1988-02-00-00-00-00')
    # info = GetMoreInfo('2028-06-01-00-00-00')
    # info = GetMoreInfo('2024-06-01-10-00-00')
    # info = GetMoreInfo('2029-03-24-08-00-00')
    dic = info.birth_to_info()
    print(dic)
    base_dic = info.get_baby_info()
    print(base_dic)
    bazi = dic['八字']
    bw_s, lack_wuxing = info.get_wuxing(bazi)
    print(bw_s, lack_wuxing)
    info.get_naming_radicals(base_dic['生肖'])

    print(info.get_baby_info_new())

'''
立春、雨水、惊蛰、春分、清明、谷雨、
立夏、小满、芒种、夏至、小暑、大暑、
立秋、处暑、白露、秋分、寒露、霜降、
立冬、小雪、大雪、冬至、小寒、大寒。
'''
