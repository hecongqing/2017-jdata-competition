
# coding: utf-8

# In[1]:

#!/usr/bin/env python

import time
from datetime import datetime
from datetime import timedelta
import pandas as pd
import os
import math
import numpy as np
from sklearn import preprocessing
from  sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

action_1_path = "./data/JData_Action_201602.csv"
action_2_path = "./data/JData_Action_201603.csv"
action_3_path = "./data/JData_Action_201604.csv"
comment_path = "./data/JData_Comment.csv"
product_path = "./data/JData_Product.csv"
user_path = "./data/JData_User.csv"

def convert_age(age_str):
    if age_str == u'-1':
        return 0
    elif age_str == u'15岁以下':
        return 1
    elif age_str == u'16-25岁':
        return 2
    elif age_str == u'26-35岁':
        return 3
    elif age_str == u'36-45岁':
        return 4
    elif age_str == u'46-55岁':
        return 5
    elif age_str == u'56岁以上':
        return 6
    else:
        return -1


# 用户的基本信息
def get_basic_user_feat():
    dump_path = './cache/basic_user.csv'
    if os.path.exists(dump_path):
        user = pd.read_csv(dump_path)
    else:
        user = pd.read_csv(user_path, encoding='gbk')
        user['age'] = user['age'].map(convert_age)
        age_df = pd.get_dummies(user["age"], prefix="age")
        sex_df = pd.get_dummies(user["sex"], prefix="sex")
        user_lv_df = pd.get_dummies(user["user_lv_cd"], prefix="user_lv_cd")
        user = pd.concat([user['user_id'], age_df, sex_df, user_lv_df], axis=1)
        user.to_csv(dump_path, index=False)
    return user


# 商品的基本信息
def get_basic_product_feat():
    dump_path = './cache/basic_product.csv'
    if os.path.exists(dump_path):
        product = pd.read_csv(dump_path)
    else:
        product = pd.read_csv(product_path)
        attr1_df = pd.get_dummies(product["a1"], prefix="a1")
        attr2_df = pd.get_dummies(product["a2"], prefix="a2")
        attr3_df = pd.get_dummies(product["a3"], prefix="a3")
        product = pd.concat([product[['sku_id', 'cate', 'brand']], attr1_df, attr2_df, attr3_df], axis=1)
        product.to_csv(dump_path, index=False)
    return product


def get_actions_1():
    action = pd.read_csv(action_1_path)
    return action


def get_actions_2():
    action2 = pd.read_csv(action_2_path)
    return action2


def get_actions_3():
    action3 = pd.read_csv(action_3_path)
    return action3


# 行为数据
def get_actions(start_date, end_date):
    """

    :param start_date:
    :param end_date:
    :return: actions: pd.Dataframe
    """
    dump_path = './cache/all_action_%s_%s.csv' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = pd.read_csv(dump_path)
    else:
        action_1 = get_actions_1()
        action_1 = action_1[(action_1.time >= start_date) & (action_1.time < end_date)]
        action_2 = get_actions_2()
        action_2 = action_2[(action_2.time >= start_date) & (action_2.time < end_date)]
        actions = pd.concat([action_1, action_2])
        action_3 = get_actions_3()
        action_3 = action_3[(action_3.time >= start_date) & (action_3.time < end_date)]
        actions = pd.concat([actions, action_3])  # type: pd.DataFrame
        actions = actions[(actions.time >= start_date) & (actions.time < end_date)]
        actions.to_csv(dump_path, index=False)
    return actions

# 评论数据
comment_date = ["2016-02-01", "2016-02-08", "2016-02-15", "2016-02-22", "2016-02-29", "2016-03-07", "2016-03-14",
                "2016-03-21", "2016-03-28",
                "2016-04-04", "2016-04-11", "2016-04-15"]
def get_comments_product_feat(start_date, end_date):
    dump_path = './cache/comments_accumulate_%s_%s.csv' % (start_date, end_date)
    if os.path.exists(dump_path):
        comments = pd.read_csv(dump_path)
    else:
        comments = pd.read_csv(comment_path)
        comment_date_end = end_date
        comment_date_begin = comment_date[0]
        for date in reversed(comment_date):
            if date < comment_date_end:
                comment_date_begin = date
                break
        comments = comments[(comments.dt >= comment_date_begin) & (comments.dt < comment_date_end)]
        df = pd.get_dummies(comments['comment_num'], prefix='comment_num')
        comments = pd.concat([comments, df], axis=1)  # type: pd.DataFrame
        # del comments['dt']
        # del comments['comment_num']
        comments = comments[
            ['sku_id', 'has_bad_comment', 'bad_comment_rate', 'comment_num_1', 'comment_num_2', 'comment_num_3',
             'comment_num_4']]
        comments.to_csv(dump_path, index=False)
    return comments


# 获取两个时间相差几天
def get_day_chaju(x, end_date):
    #     x=x.split(' ')[0]
    x = datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
    end_date = datetime.strptime(end_date, '%Y-%m-%d')
    return (end_date - x).days
print("US model  finish  part_0")


# In[ ]:




# In[2]:

#!/usr/bin/env python


#from basic_feat0518 import *






# top k 天的行为次数总和(滑窗处理)
def get_user_feat(start_date, end_date, k):
    dump_path = './cache/u_action_%s_%s_%s.csv' % (start_date, end_date, k)
    if os.path.exists(dump_path):
        actions = pd.read_csv(dump_path)
    else:
        start_days = pd.to_datetime(end_date) - timedelta(days=k)
        start_days = str(start_days).split(' ')[0]
        actions = get_actions(start_days, end_date)
        actions = actions[['user_id', 'type']]
        df = pd.get_dummies(actions['type'], prefix='type')
        actions = pd.concat([actions, df], axis=1)  # type: pd.DataFrame
        actions = actions.groupby('user_id', as_index=False).sum()
        min_max_scaler = preprocessing.MinMaxScaler()
        df = min_max_scaler.fit_transform(actions.drop(['user_id', 'type'], axis=1).values)
        df = pd.DataFrame(df)
        df.columns = ['u_action_' + str(k) + '_' + str(i) for i in range(1, df.shape[1] + 1)]
        actions = pd.concat([actions[['user_id']], df], axis=1)
        actions.to_csv(dump_path, index=False)
    return actions


# 用户的行为转化率
def get_action_user_feat1(start_date, end_date):
    feature = ['user_id', 'user_action_1_ratio', 'user_action_2_ratio', 'user_action_3_ratio',
               'user_action_5_ratio', 'user_action_6_ratio']
    dump_path = './cache/user_feat_accumulate_xiugai_%s_%s.csv' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = pd.read_csv(dump_path)
    else:
        actions = get_actions(start_date, end_date)
        df = pd.get_dummies(actions['type'], prefix='action')
        actions = pd.concat([actions['user_id'], df], axis=1)
        actions = actions.groupby(['user_id'], as_index=False).sum()
        actions['user_action_1_ratio'] = actions['action_4'] / actions['action_1']
        actions['user_action_2_ratio'] = actions['action_4'] / actions['action_2']
        #         actions['user_action_3_ratio'] = actions['action_4'] / actions['action_3']
        actions['user_action_3_ratio'] = actions['action_3'] / actions['action_2']
        actions['user_action_5_ratio'] = actions['action_4'] / actions['action_5']
        actions['user_action_6_ratio'] = actions['action_4'] / actions['action_6']
        #         3.购物车删除
        actions = actions[feature]
        actions.to_csv(dump_path, index=False)
    return actions


# 用户购买/加入购物车/关注前访问天数
def get_action_user_feat2(start_date, end_date):
    dump_path = './cache/user_feat2_after_%s_%s.csv' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = pd.read_csv(dump_path)

    else:
        # 用户购买前访问天数
        def user_feat_2_1(start_date, end_date):
            actions = get_actions(start_date, end_date)[['user_id', 'type', 'time']]
            actions['time'] = pd.to_datetime(actions['time']).dt.date
            # actions=actions.drop_duplicates(['user_id','time'],keep='first')
            visit = actions[actions['type'] == 1]
            visit = visit.drop_duplicates(['user_id', 'time'], keep='first')
            del visit['time']
            del actions['time']
            visit = visit.groupby('user_id', as_index=False).count()
            visit.columns = ['user_id', 'visit']
            buy = actions[actions['type'] == 4]
            buy = buy.groupby('user_id', as_index=False).count()
            buy.columns = ['user_id', 'buy']
            actions = pd.merge(visit, buy, on='user_id', how='left')
            actions['visit_day_before_buy'] = actions['visit'] / actions['buy']
            del actions['buy']
            del actions['visit']
            return actions

        # 用户加入购物车前访问天数
        def user_feat_2_2(start_date, end_date):
            actions = get_actions(start_date, end_date)[['user_id', 'type', 'time']]
            actions['time'] = pd.to_datetime(actions['time']).dt.date
            # actions=actions.drop_duplicates(['user_id','time'],keep='first')
            visit = actions[actions['type'] == 1]
            visit = visit.drop_duplicates(['user_id', 'time'], keep='first')
            del visit['time']
            del actions['time']
            visit = visit.groupby('user_id', as_index=False).count()
            visit.columns = ['user_id', 'visit']
            addtoshopping = actions[actions['type'] == 2]
            addtoshopping = addtoshopping.groupby('user_id', as_index=False).count()
            addtoshopping.columns = ['user_id', 'addtoshopping']
            actions = pd.merge(visit, addtoshopping, on='user_id', how='left')
            actions['visit_day_before_addtoshopping'] = actions['visit'] / actions['addtoshopping']
            del actions['addtoshopping']
            del actions['visit']
            return actions

        # 用户关注前访问天数
        def user_feat_2_3(start_date, end_date):
            actions = get_actions(start_date, end_date)[['user_id', 'type', 'time']]
            actions['time'] = pd.to_datetime(actions['time']).dt.date
            # actions=actions.drop_duplicates(['user_id','time'],keep='first')
            visit = actions[actions['type'] == 1]
            visit = visit.drop_duplicates(['user_id', 'time'], keep='first')
            del visit['time']
            del actions['time']
            visit = visit.groupby('user_id', as_index=False).count()
            visit.columns = ['user_id', 'visit']
            guanzhu = actions[actions['type'] == 5]
            guanzhu = guanzhu.groupby('user_id', as_index=False).count()
            guanzhu.columns = ['user_id', 'guanzhu']
            actions = pd.merge(visit, guanzhu, on='user_id', how='left')
            actions['visit_day_before_guanzhu'] = actions['visit'] / actions['guanzhu']
            del actions['guanzhu']
            del actions['visit']
            return actions

        # 用户购买前加入购物车天数
        def user_feat_2_4(start_date, end_date):
            actions = get_actions(start_date, end_date)[['user_id', 'type', 'time']]
            actions['time'] = pd.to_datetime(actions['time']).dt.date
            # actions=actions.drop_duplicates(['user_id','time'],keep='first')
            addtoshopping = actions[actions['type'] == 2]
            addtoshopping = addtoshopping.drop_duplicates(['user_id', 'time'], keep='first')
            del addtoshopping['time']
            del actions['time']
            addtoshopping = addtoshopping.groupby('user_id', as_index=False).count()
            addtoshopping.columns = ['user_id', 'addtoshopping']
            buy = actions[actions['type'] == 4]
            buy = buy.groupby('user_id', as_index=False).count()
            buy.columns = ['user_id', 'buy']
            actions = pd.merge(addtoshopping, buy, on='user_id', how='left')
            actions['addtoshopping_day_before_buy'] = actions['addtoshopping'] / actions['buy']
            del actions['buy']
            del actions['addtoshopping']
            return actions

        # 用户购买前关注天数
        def user_feat_2_5(start_date, end_date):
            actions = get_actions(start_date, end_date)[['user_id', 'type', 'time']]
            actions['time'] = pd.to_datetime(actions['time']).dt.date
            guanzhu = actions[actions['type'] == 5]
            guanzhu = guanzhu.drop_duplicates(['user_id', 'time'], keep='first')
            del guanzhu['time']
            del actions['time']
            guanzhu = guanzhu.groupby('user_id', as_index=False).count()
            guanzhu.columns = ['user_id', 'guanzhu']
            buy = actions[actions['type'] == 4]
            buy = buy.groupby('user_id', as_index=False).count()
            buy.columns = ['user_id', 'buy']
            actions = pd.merge(guanzhu, buy, on='user_id', how='left')
            actions['guanzhu_day_before_buy'] = actions['guanzhu'] / actions['buy']
            del actions['buy']
            del actions['guanzhu']
            return actions

        actions = pd.merge(user_feat_2_1(start_date, end_date), user_feat_2_2(start_date, end_date), on='user_id',
                           how='outer')
        actions = pd.merge(actions, user_feat_2_3(start_date, end_date), on='user_id', how='outer')
        actions = pd.merge(actions, user_feat_2_4(start_date, end_date), on='user_id', how='outer')
        actions = pd.merge(actions, user_feat_2_5(start_date, end_date), on='user_id', how='outer')
        user_id = actions['user_id']
        del actions['user_id']
        actions = actions.fillna(0)
        min_max_scale = preprocessing.MinMaxScaler()
        actions = min_max_scale.fit_transform(actions.values)
        actions = pd.concat([user_id, pd.DataFrame(actions)], axis=1)
        actions.to_csv(dump_path, index=False)
    actions.columns = ['user_id'] + ['u_feat2_' + str(i) for i in range(1, actions.shape[1])]
    return actions

#print get_action_user_feat2('2016-02-01','2016-04-11')


# 用户平均访问间隔
def get_action_user_feat6(start_date, end_date):
    dump_path = './cache/user_feat6_%s_%s.csv' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = pd.read_csv(dump_path)
    else:

        df = get_actions(start_date, end_date)[['user_id', 'time']]
        # df['user_id']=df['user_id'].astype('int')
        df['time'] = df['time'].map(lambda x: x.split(' ')[0])
        df = df.drop_duplicates(['user_id', 'time'], keep='first')
        df['time'] = df['time'].map(lambda x: datetime.strptime(x, '%Y-%m-%d'))
        actions = df.groupby('user_id', as_index=False).agg(lambda x: x['time'].diff().mean())
        actions['avg_visit'] = actions['time'].dt.days
        del actions['time']
        actions.to_csv(dump_path, index=False)
    actions.columns = ['user_id'] + ['u_feat6_' + str(i) for i in range(1, actions.shape[1])]
    return actions


# 用户平均六种行为的访问间隔
def get_action_user_feat6_six(start_date, end_date):
    dump_path = './cache/user_feat6_six_%s_%s.csv' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = pd.read_csv(dump_path)
    else:
        df = get_actions(start_date, end_date)[['user_id', 'time', 'type']]
        df['time'] = df['time'].map(lambda x: (-1) * get_day_chaju(x, start_date))
        df = df.drop_duplicates(['user_id', 'time', 'type'], keep='first')
        actions = df.groupby(['user_id', 'type']).agg(lambda x: np.diff(x).mean())
        actions = actions.unstack()
        actions.columns = list(range(actions.shape[1]))
        actions = actions.reset_index()
        actions.to_csv(dump_path, index=False)
    actions.columns = ['user_id'] + ['u_feat6_six_' + str(i) for i in range(1, actions.shape[1])]
    return actions


# 用户购买频率
def get_action_user_feat7(start_date, end_date):
    dump_path = './cache/user_feat7_six_%s_%s.csv' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = pd.read_csv(dump_path)
    else:
        df = get_actions(start_date, end_date)[['user_id', 'type', 'time']]
        actions = df.groupby(['user_id', 'type'], as_index=False).count()

        time_min = df.groupby(['user_id', 'type'], as_index=False).min()
        time_max = df.groupby(['user_id', 'type'], as_index=False).max()

        time_cha = pd.merge(time_max, time_min, on=['user_id', 'type'], how='left')
        time_cha['time_x'] = time_cha['time_x'].map(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
        time_cha['time_y'] = time_cha['time_y'].map(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))

        time_cha['cha_hour'] = 1 + (time_cha['time_x'] - time_cha['time_y']).dt.days * 24 + (time_cha['time_x'] -
                                                                                             time_cha[
                                                                                                 'time_y']).dt.seconds // 3600
        del time_cha['time_x']
        del time_cha['time_y']
        # time_cha=time_cha.fillna(1)

        actions = pd.merge(time_cha, actions, on=['user_id', 'type'], how="left")
        actions = actions.groupby(['user_id', 'type']).sum()
        actions['cnt/time'] = actions['time'] / actions["cha_hour"]
        actions = actions.unstack()
        actions.columns = list(range(actions.shape[1]))
        actions = actions.reset_index()
        actions = actions.fillna(0)
        actions.to_csv(dump_path, index=False)
    actions.columns = ['user_id'] + ['u_feat7_' + str(i) for i in range(1, actions.shape[1])]
    return actions


def user_top_k_0_1(start_date, end_date):
    actions = get_actions(start_date, end_date)
    actions = actions[['user_id', 'sku_id', 'type']]
    df = pd.get_dummies(actions['type'], prefix='%s-%s-action' % (start_date, end_date))
    actions = pd.concat([actions, df], axis=1)  # type: pd.DataFrame
    actions = actions.groupby('user_id', as_index=False).sum()
    del actions['type']
    del actions['sku_id']
    user_id = actions['user_id']
    del actions['user_id']
    actions = actions.applymap(lambda x: 1 if x > 0 else 0)
    actions = pd.concat([user_id, actions], axis=1)
    return actions


# 用户最近K天行为0/1提取
def get_action_user_feat8(start_date, end_date):
    dump_path = './cache/user_feat8_%s_%s.csv' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = pd.read_csv(dump_path)
    else:
        actions = None
        for i in (1, 2, 3, 4, 5, 6, 7, 15, 30):
            print(i)
            start_days = datetime.strptime(end_date, '%Y-%m-%d') - timedelta(days=i)
            start_days = start_days.strftime('%Y-%m-%d')
            if actions is None:
                actions = user_top_k_0_1(start_days, end_date)
            else:
                actions = pd.merge(actions, user_top_k_0_1(start_days, end_date), how='outer', on='user_id')
        actions.to_csv(dump_path, index=False)
    actions.columns = ['user_id'] + ['u_feat8_' + str(i) for i in range(1, actions.shape[1])]
    return actions


# 获取用户的重复购买率
def get_action_user_feat8_2(start_date, end_date):
    dump_path = './cache/product_feat8_2_%s_%s.csv' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = pd.read_csv(dump_path)
    else:
        df = get_actions(start_date, end_date)[['user_id', 'sku_id', 'type']]
        df = df[df['type'] == 4]  # 购买的行为
        df = df.groupby(['user_id', 'sku_id'], as_index=False).count()
        df.columns = ['user_id', 'sku_id', 'count1']
        df['count1'] = df['count1'].map(lambda x: 1 if x > 1 else 0)
        grouped = df.groupby(['user_id'], as_index=False)
        actions = grouped.count()[['user_id', 'count1']]
        actions.columns = ['user_id', 'count']
        re_count = grouped.sum()[['user_id', 'count1']]
        re_count.columns = ['user_id', 're_count']
        actions = pd.merge(actions, re_count, on='user_id', how='left')
        re_buy_rate = actions['re_count'] / actions['count']
        actions = pd.concat([actions['user_id'], re_buy_rate], axis=1)
        actions.columns = ['user_id', 're_buy_rate']
        actions.to_csv(dump_path, index=False)
    actions.columns = ['user_id'] + ['u_feat8_2_' + str(i) for i in range(1, actions.shape[1])]
    return actions


# 获取最近一次行为的时间距离当前时间的差距
def get_action_user_feat9(start_date, end_date):
    dump_path = './cache/user_feat9_%s_%s.csv' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = pd.read_csv(dump_path)
    else:
        df = get_actions(start_date, end_date)[['user_id', 'time', 'type']]
        # df['time'] = df['time'].map(lambda x: (-1)*get_day_chaju(x,start_date))
        df = df.drop_duplicates(['user_id', 'type'], keep='last')
        df['time'] = df['time'].map(lambda x: get_day_chaju(x, end_date) + 1)
        actions = df.groupby(['user_id', 'type']).sum()
        actions = actions.unstack()
        actions.columns = list(range(actions.shape[1]))
        actions = actions.reset_index()
        actions = actions.fillna(30)
        actions.to_csv(dump_path, index=False)
    actions.columns = ['user_id'] + ['u_feat9_' + str(i) for i in range(1, actions.shape[1])]
    return actions


# 获取最后一次行为的次数并且进行归一化
def get_action_user_feat10(start_date, end_date):
    dump_path = './cache/user_feat10_%s_%s.csv' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = pd.read_csv(dump_path)
    else:

        df = get_actions(start_date, end_date)[['user_id', 'time', 'type']]
        df['time'] = df['time'].map(lambda x: get_day_chaju(x, end_date) + 1)

        idx = df.groupby(['user_id', 'type'])['time'].transform(min)
        idx1 = idx == df['time']
        actions = df[idx1].groupby(["user_id", "type"]).count()
        actions = actions.unstack()
        actions.columns = list(range(actions.shape[1]))
        actions = actions.fillna(0)
        actions = actions.reset_index()

        user_sku = actions[['user_id']]
        del actions['user_id']
        min_max_scaler = preprocessing.MinMaxScaler()
        actions = min_max_scaler.fit_transform(actions.values)
        actions = pd.DataFrame(actions)
        actions = pd.concat([user_sku, actions], axis=1)

        actions.to_csv(dump_path, index=False)
    actions.columns = ['user_id'] + ['u_feat10_' + str(i) for i in range(1, actions.shape[1])]
    return actions


# 获取人物该层级最后一层的各种行为的统计数量
def get_action_user_feat11(start_date, end_date, n):
    dump_path = './cache/user_feat11_%s_%s_%s.csv' % (start_date, end_date, n)
    if os.path.exists(dump_path):
        actions = pd.read_csv(dump_path)
    else:

        df = get_actions(start_date, end_date)[['user_id', 'time', 'type']]
        df['time'] = df['time'].map(lambda x: get_day_chaju(x, end_date) // n)
        df = df[df['time'] == 0]
        del df['time']
        temp = pd.get_dummies(df['type'], prefix='type')
        del df['type']
        actions = pd.concat([df, temp], axis=1)
        actions = actions.groupby(['user_id'], as_index=False).sum()
        user_sku = actions[['user_id']]
        del actions['user_id']
        min_max_scaler = preprocessing.MinMaxScaler()
        actions = min_max_scaler.fit_transform(actions.values)
        actions = pd.DataFrame(actions)
        actions = pd.concat([user_sku, actions], axis=1)
        actions.to_csv(dump_path, index=False)
    actions.columns = ['user_id'] + ['u_feat11_' + str(n) + '_' + str(i) for i in range(1, actions.shape[1])]
    return actions


def get_action_user_feat12(start_date, end_date):
    dump_path = './cache/user_feat12_%s_%s.csv' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = pd.read_csv(dump_path)
    else:
        actions = get_actions(start_date, end_date)[['user_id', 'time', 'type']]
        actions['time'] = actions['time'].map(lambda x: x.split(' ')[0])
        actions = actions.drop_duplicates(['user_id', 'time', 'type'], keep='first')
        actions['day'] = actions['time'].map(
            lambda x: (datetime.strptime(end_date, '%Y-%m-%d') - datetime.strptime(x, '%Y-%m-%d')).days)
        result = None
        for i in (2, 3, 7, 14, 28):  # 层级个数
            print ('i%s' % i)
            actions['level%s' % i] = actions['day'].map(lambda x: x // i)
            a = set(actions['level%s' % i].tolist())
            for j in (1, 2, 3, 4, 5, 6):  # type
                print ('j%s' % j)
                df = actions[actions['type'] == j][['user_id', 'level%s' % i, 'time']]
                df = df.groupby(['user_id', 'level%s' % i]).count()
                df = df.unstack()
                b = df.columns.levels[1].tolist()
                df.columns = ['u_feat12_' + str('level%s_' % i) + str(j) + '_' + str(k) for k in
                              df.columns.levels[1].tolist()]
                if len(list(a - set(b))) != 0:
                    c = list(a - set(b))
                    for k in c:
                        df['u_feat12_' + str('level%s_' % i) + str(j) + '_' + str(k)] = 0
                columns = df.columns
                dict = {}
                for column in columns:
                    k = int(column.split('_')[-1])
                    dict[column] = k
                columns = sorted(dict.items(), key=lambda x: x[1])
                columns = [(columns[t])[0] for t in range(len(columns))]
                df = df[columns]
                df = df.reset_index()
                if result is None:
                    result = df
                else:
                    result = pd.merge(result, df, on='user_id', how='left')
        columns = result.columns
        user_id = result['user_id']
        del result['user_id']
        actions = result.fillna(0)

        min_max_scaler = preprocessing.MinMaxScaler()
        actions = min_max_scaler.fit_transform(actions.values)
        actions = pd.DataFrame(actions)
        actions = pd.concat([user_id, actions], axis=1)
        actions.columns = columns
        actions.to_csv(dump_path, index=False)
    return actions


# 层级的天数
def get_action_user_feat13(start_date, end_date, n):
    dump_path = './cache/user_feat13_%s_%s_%s.csv' % (start_date, end_date, n)
    if os.path.exists(dump_path):
        actions = pd.read_csv(dump_path)
    else:
        df = get_actions(start_date, end_date)[['user_id', 'time', 'type']]
        df['time'] = df['time'].map(lambda x: get_day_chaju(x, end_date) // n)
        df = df.drop_duplicates(['user_id', 'type', 'time'], keep='first')
        actions = df.groupby(['user_id', 'type']).count()
        actions = actions.unstack()
        actions.columns = list(range(actions.shape[1]))
        actions = actions.fillna(0)
        actions = actions.reset_index()
        user_sku = actions[['user_id']]
        del actions['user_id']
        min_max_scaler = preprocessing.MinMaxScaler()
        actions = min_max_scaler.fit_transform(actions.values)
        actions = pd.DataFrame(actions)
        actions = pd.concat([user_sku, actions], axis=1)
        actions.to_csv(dump_path, index=False)
    actions.columns = ['user_id'] + ['u_feat13_' + str(n) + '_' + str(i) for i in range(1, actions.shape[1])]
    return actions


def get_action_user_feat14(start_date, end_date):
    dump_path = './cache/user_feat14_%s_%s.csv' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = pd.read_csv(dump_path)
    else:
        n = 5
        df = get_actions(start_date, end_date)[['user_id', 'time', 'type']]
        df = df[df['type'] == 4][['user_id', 'time']]
        df['time'] = df['time'].map(lambda x: get_day_chaju(x, end_date) // n)
        days = np.max(df['time'])

        df['cnt'] = 0
        actions = df.groupby(['user_id', 'time']).count()

        actions = actions.unstack()

        actions.columns = list(range(actions.shape[1]))
        actions = actions.reset_index()

        actions = actions.fillna(0)
        user_sku = actions[['user_id']]
        del actions['user_id']
        min_max_scaler = preprocessing.MinMaxScaler()
        actions = min_max_scaler.fit_transform(actions.values)
        actions = pd.DataFrame(actions)
        actions = pd.concat([user_sku, actions], axis=1)
        actions.to_csv(dump_path, index=False)
    actions.columns = ['user_id'] + ['u_feat14_' + str(i) for i in range(1, actions.shape[1])]
    return actions


# 用户购买/加入购物车/关注前访问次数
def get_action_user_feat15(start_date, end_date):
    dump_path = './cache/user_feat15_%s_%s.csv' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = pd.read_csv(dump_path)
    else:
        # 用户购买前访问次数
        def user_feat_15_1(start_date, end_date):
            actions = get_actions(start_date, end_date)[['user_id', 'type']]
            visit = actions[actions['type'] == 1]
            visit = visit.groupby('user_id', as_index=False).count()
            visit.columns = ['user_id', 'visit']
            buy = actions[actions['type'] == 4]
            buy = buy.groupby('user_id', as_index=False).count()
            buy.columns = ['user_id', 'buy']
            actions = pd.merge(visit, buy, on='user_id', how='left')
            actions['visit_num_before_buy'] = actions['visit'] / actions['buy']
            del actions['buy']
            del actions['visit']
            return actions

        # 用户加入购物车前访问次数
        def user_feat_15_2(start_date, end_date):
            actions = get_actions(start_date, end_date)[['user_id', 'type']]
            visit = actions[actions['type'] == 1]
            visit = visit.groupby('user_id', as_index=False).count()
            visit.columns = ['user_id', 'visit']
            addtoshopping = actions[actions['type'] == 2]
            addtoshopping = addtoshopping.groupby('user_id', as_index=False).count()
            addtoshopping.columns = ['user_id', 'addtoshopping']
            actions = pd.merge(visit, addtoshopping, on='user_id', how='left')
            actions['visit_num_before_addtoshopping'] = actions['visit'] / actions['addtoshopping']
            del actions['addtoshopping']
            del actions['visit']
            return actions

        # 用户关注前访问次数
        def user_feat_15_3(start_date, end_date):
            actions = get_actions(start_date, end_date)[['user_id', 'type']]
            visit = actions[actions['type'] == 1]
            visit = visit.groupby('user_id', as_index=False).count()
            visit.columns = ['user_id', 'visit']
            guanzhu = actions[actions['type'] == 5]
            guanzhu = guanzhu.groupby('user_id', as_index=False).count()
            guanzhu.columns = ['user_id', 'guanzhu']
            actions = pd.merge(visit, guanzhu, on='user_id', how='left')
            actions['visit_num_before_guanzhu'] = actions['visit'] / actions['guanzhu']
            del actions['guanzhu']
            del actions['visit']
            return actions

        # 用户购买前加入购物车次数
        def user_feat_15_4(start_date, end_date):
            actions = get_actions(start_date, end_date)[['user_id', 'type']]
            addtoshopping = actions[actions['type'] == 2]
            addtoshopping = addtoshopping.groupby('user_id', as_index=False).count()
            addtoshopping.columns = ['user_id', 'addtoshopping']
            buy = actions[actions['type'] == 4]
            buy = buy.groupby('user_id', as_index=False).count()
            buy.columns = ['user_id', 'buy']
            actions = pd.merge(addtoshopping, buy, on='user_id', how='left')
            actions['addtoshopping_num_before_buy'] = actions['addtoshopping'] / actions['buy']
            del actions['buy']
            del actions['addtoshopping']
            return actions

        # 用户购买前关注次数
        def user_feat_15_5(start_date, end_date):
            actions = get_actions(start_date, end_date)[['user_id', 'type']]
            guanzhu = actions[actions['type'] == 5]
            guanzhu = guanzhu.groupby('user_id', as_index=False).count()
            guanzhu.columns = ['user_id', 'guanzhu']
            buy = actions[actions['type'] == 4]
            buy = buy.groupby('user_id', as_index=False).count()
            buy.columns = ['user_id', 'buy']
            actions = pd.merge(guanzhu, buy, on='user_id', how='left')
            actions['guanzhu_num_before_buy'] = actions['guanzhu'] / actions['buy']
            del actions['buy']
            del actions['guanzhu']
            return actions

        actions = pd.merge(user_feat_15_1(start_date, end_date), user_feat_15_2(start_date, end_date), on='user_id',
                           how='outer')
        actions = pd.merge(actions, user_feat_15_3(start_date, end_date), on='user_id', how='outer')
        actions = pd.merge(actions, user_feat_15_4(start_date, end_date), on='user_id', how='outer')
        actions = pd.merge(actions, user_feat_15_5(start_date, end_date), on='user_id', how='outer')
        user_id = actions['user_id']
        del actions['user_id']
        actions = actions.fillna(0)
        min_max_scale = preprocessing.MinMaxScaler()
        actions = min_max_scale.fit_transform(actions.values)
        actions = pd.concat([user_id, pd.DataFrame(actions)], axis=1)

        actions.to_csv(dump_path, index=False)
    actions.columns = ['user_id'] + ['u_feat15_' + str(i) for i in range(1, actions.shape[1])]
    return actions


# 用户行为的交叉
def get_action_user_feat16(start_date, end_date):
    dump_path = './cache/user_feat16_%s_%s.csv' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = pd.read_csv(dump_path)
    else:
        actions = get_actions(start_date, end_date)[['user_id', 'type']]
        actions['cnt'] = 0
        action1 = actions.groupby(['user_id', 'type']).count()
        action1 = action1.unstack()
        index_col = list(range(action1.shape[1]))
        action1.columns = index_col
        action1 = action1.reset_index()
        action2 = actions.groupby('user_id', as_index=False).count()
        del action2['type']
        action2.columns = ['user_id', 'cnt']
        actions = pd.merge(action1, action2, how='left', on='user_id')
        for i in index_col:
            actions[i] = actions[i] / actions['cnt']
        del actions['cnt']
        actions.to_csv(dump_path, index=False)
    actions.columns = ['user_id'] + ['u_feat16_' + str(i) for i in range(1, actions.shape[1])]
    return actions

#get_action_u0509_feat_17  =======>>>无

# 最近k天用户访问P集合的商品数/用户访问总体的商品数（k小于7天，不除总体的商品数，反之，除）
def get_action_u0509_feat_18(start_date, end_date, n):
    dump_path = './cache/user_feat0509_1_30_%s_%s_%s.csv' % (start_date, end_date, n)
    if os.path.exists(dump_path):
        actions = pd.read_csv(dump_path)
    else:

        start_days = datetime.strptime(end_date, '%Y-%m-%d') - timedelta(days=n)
        start_days = datetime.strftime(start_days, '%Y-%m-%d')

        actions = get_actions(start_days, end_date)[['user_id', 'sku_id', 'type']]
        actions_dummy = pd.get_dummies(actions['type'], prefix='actions')
        actions = pd.concat([actions, actions_dummy], axis=1)
        del actions['type']

        P = get_basic_product_feat()[['sku_id']]
        P['label'] = 1
        actions_sub = pd.merge(actions, P, on='sku_id', how='left')
        actions_sub = actions_sub[actions_sub['label'] == 1]
        del actions_sub['label']

        actions_sub = actions_sub.groupby(['user_id'], as_index=False).sum()
        del actions_sub['sku_id']
        actions_all = actions.groupby(['user_id'], as_index=False).sum()
        del actions_all['sku_id']

        if n > 7:
            actions = pd.merge(actions_all, actions_sub, on=['user_id'], how='left')
            # print actions.head()
            for i in range(1, 7):
                actions['actions_%s' % i] = actions['actions_%s_y' % i] / actions['actions_%s_x' % i]
                # actions=actions[['user_id','actions_1','actions_2','actions_3','actions_4','actions_5','actions_6']]

        else:
            actions = pd.merge(actions_all, actions_sub, on=['user_id'], how='left')
        actions.to_csv(dump_path, index=False)
    actions.columns = ['user_id'] + ['u_feat18_' + str(n) + '_' + str(i) for i in range(1, actions.shape[1])]
    #     user_id = actions[['user_id']]
    #     del actions['user_id']
    #     actions = actions.fillna(0)
    #     actions=actions.replace(np.inf,0)
    # #         print(actions.head())
    #     columns = actions.columns

    #     min_max_scale = preprocessing.MinMaxScaler()
    #     actions=actions.replace(np.inf,0)
    #     actions = min_max_scale.fit_transform(actions.values)
    #     actions = pd.concat([user_id, pd.DataFrame(actions,columns = columns)], axis=1)
    return actions


# 用户点击到购买的时间间隔
def get_action_u0509_feat_19(start_date, end_date):
    dump_path = './cache/get_action_user_feat0515_2_1_%s_%s.csv' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = pd.read_csv(dump_path)
    else:
        actions = get_actions(start_date, end_date)
        actions_dianji = actions[actions['type'] == 6][['user_id', 'sku_id', 'time']]
        actions_dianji['time_dianji'] = actions_dianji['time'].map(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
        actions_dianji = actions_dianji[['user_id', 'sku_id', 'time_dianji']]
        actions_dianji = actions_dianji.drop_duplicates(['user_id', 'sku_id'], keep='first')

        actions_goumai = actions[actions['type'] == 4][['user_id', 'sku_id', 'time']]
        actions_goumai['time_goumai'] = actions_goumai['time'].map(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
        actions_goumai = actions_goumai[['user_id', 'sku_id', 'time_goumai']]
        actions_goumai = actions_goumai.drop_duplicates(['user_id', 'sku_id'], keep='last')

        actions = pd.merge(actions_dianji, actions_goumai, on=['user_id', 'sku_id'], how='inner')
        actions['time_jiange'] = actions['time_goumai'] - actions['time_dianji']
        actions = actions.drop(['sku_id', 'time_goumai', 'time_dianji'], axis=1)
        actions['time_jiange'] = actions['time_jiange'].map(lambda x: x.days * 24 + x.seconds // 3600 + 1)

        actions_min = actions.groupby('user_id').min().reset_index()
        actions_min.columns = ['user_id', 'time_min']
        # actions_mean = actions.groupby('user_id').mean().reset_index()
        # actions_mean.columns = ['user_id','time_mean']
        actions_max = actions.groupby('user_id').max().reset_index()
        actions_max.columns = ['user_id', 'time_max']
        actions = pd.merge(actions_min, actions_max, on='user_id', how='left')

        user_id = actions[['user_id']]
        del actions['user_id']
        actions = actions.fillna(0)
        columns = actions.columns
        min_max_scale = preprocessing.MinMaxScaler()
        actions = min_max_scale.fit_transform(actions.values)
        actions = pd.concat([user_id, pd.DataFrame(actions, columns=columns)], axis=1)
        actions.to_csv(dump_path, index=False)
    actions.columns = ['user_id'] + ['u_feat19_' + str(i) for i in range(1, actions.shape[1])]
    return actions


# 用户购买每种cate的数量
def get_action_u0509_feat_20(start_date, end_date):
    dump_path = './cache/get_action_user_feat0515_2_2_%s_%s.csv' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = pd.read_csv(dump_path)
    else:
        actions = get_actions(start_date, end_date)
        actions = get_actions(start_date, end_date)[['user_id', 'cate']]
        cate_col = pd.get_dummies(actions['cate'], prefix='cate')
        actions = pd.concat([actions[['user_id']], cate_col], axis=1)
        actions = actions.groupby('user_id').sum().reset_index()

        user_id = actions[['user_id']]
        del actions['user_id']
        actions = actions.fillna(0)
        columns = actions.columns
        min_max_scale = preprocessing.MinMaxScaler()
        actions = min_max_scale.fit_transform(actions.values)
        actions = pd.concat([user_id, pd.DataFrame(actions, columns=columns)], axis=1)
        actions.to_csv(dump_path, index=False)
    return actions


# 获取某人某段时间内加入购物车的数量以及关注的数量
def get_action_u0509_feat_21(start_date, end_date, n):
    dump_path = './cache/u0509_feat_21_%s_%s_%s.csv' % (start_date, end_date, n)
    if os.path.exists(dump_path):
        actions = pd.read_csv(dump_path)
    else:

        start_days = datetime.strptime(end_date, '%Y-%m-%d') - timedelta(days=n)
        start_days = datetime.strftime(start_days, '%Y-%m-%d')

        actions = get_actions(start_days, end_date)[['user_id', 'type', 'cate']]
        actions_gouwuche = actions[actions['type'] == 2]
        actions_gouwuche_1 = actions_gouwuche[['user_id', 'type']]
        actions_gouwuche_1 = actions_gouwuche_1.groupby('user_id').count().reset_index()
        actions_gouwuche_1.columns = ['user_id', str(n) + 'gouwuche_add']

        actions_gouwuche = actions_gouwuche[actions_gouwuche['cate'] == 8]
        actions_gouwuche_2=actions_gouwuche[['user_id', 'type']]
        actions_gouwuche_2 = actions_gouwuche_2.groupby('user_id').count().reset_index()
        actions_gouwuche_2.columns = ['user_id', str(n) + 'gouwuche_add_cate_8']

        actions_guanzhu = actions[actions['type'] == 5]
        actions_guanzhu_1 = actions_guanzhu[['user_id', 'type']]
        actions_guanzhu_1 = actions_guanzhu_1.groupby('user_id').count().reset_index()
        actions_guanzhu_1.columns = ['user_id', str(n) + 'guanzhu_add']

        actions_guanzhu = actions_guanzhu[actions_guanzhu['cate'] == 8]
        actions_guanzhu_2=actions_guanzhu[['user_id', 'type']]
        actions_guanzhu_2 = actions_guanzhu_2.groupby('user_id').count().reset_index()
        actions_guanzhu_2.columns = ['user_id', str(n) + 'guanzhu_add_cate_8']

        actions = pd.merge(actions_gouwuche_1, actions_gouwuche_2, on='user_id', how='outer')
        actions = pd.merge(actions, actions_guanzhu_1, on='user_id', how='outer')
        actions = pd.merge(actions, actions_guanzhu_2, on='user_id', how='outer')
        actions = actions.fillna(0)

        user_id = actions[['user_id']]
        del actions['user_id']
        actions = actions.fillna(0)
        columns = actions.columns
        min_max_scale = preprocessing.MinMaxScaler()
        actions = min_max_scale.fit_transform(actions.values)
        actions = pd.concat([user_id, pd.DataFrame(actions, columns=columns)], axis=1)
        actions.to_csv(dump_path, index=False)
    return actions


# top n 中 某人使用了多少天产生了该行为
def get_action_u0509_feat_22(start_date, end_date, n):
    dump_path = './cache/get_action_user_feat0515_2_4_%s_%s_%s.csv' % (start_date, end_date, n)
    if os.path.exists(dump_path):
        actions = pd.read_csv(dump_path)
    else:

        start_days = datetime.strptime(end_date, '%Y-%m-%d') - timedelta(days=n)
        start_days = datetime.strftime(start_days, '%Y-%m-%d')

        actions = get_actions(start_days, end_date)[['user_id', 'type', 'time']]
        actions['time'] = actions['time'].map(
            lambda x: (datetime.strptime(end_date, '%Y-%m-%d') - datetime.strptime(x, '%Y-%m-%d %H:%M:%S')).days)
        actions = actions.drop_duplicates(['user_id', 'type', 'time'])
        actions = actions.groupby(['user_id', 'type']).count()
        actions.columns = [str(n) + 'day_nums']
        actions = actions.unstack()
        actions = actions.reset_index()
        actions.columns = ['user_id'] + ['get_action_user_feat0515_2_4_' + str(n) + '_' + str(i) for i in
                                         range(1, actions.shape[1])]
        actions = actions.fillna(0)

        user_id = actions[['user_id']]
        del actions['user_id']
        actions = actions.fillna(0)
        columns = actions.columns
        min_max_scale = preprocessing.MinMaxScaler()
        actions = min_max_scale.fit_transform(actions.values)
        actions = pd.concat([user_id, pd.DataFrame(actions, columns=columns)], axis=1)
        actions.to_csv(dump_path, index=False)
    return actions


# 用户总购买/加购/关注/点击/浏览品牌数
def get_action_user_feat5(start_date, end_date):
    dump_path = './cache/user_feat5_a_%s_%s.csv' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = pd.read_csv(dump_path)
    else:
        actions = get_actions(start_date, end_date)
        action = None
        for i in (1, 2, 4, 5, 6):
            df = actions[actions['type'] == i][['user_id', 'sku_id']]
            df = df.drop_duplicates(['user_id', 'sku_id'], keep='first')
            df = df.groupby('user_id', as_index=False).count()
            df.columns = ['user_id', 'num_%s' % i]
            if i == 1:
                action = df
            else:
                action = pd.merge(action, df, on='user_id', how='outer')
        actions = action.fillna(0)
        actions = actions.astype('float')
        user = actions[['user_id']]
        min_max_scaler = preprocessing.MinMaxScaler()
        actions = min_max_scaler.fit_transform(actions.drop(['user_id'], axis=1).values)
        actions = pd.DataFrame(actions)
        actions = pd.concat([user, actions], axis=1)
        actions.to_csv(dump_path, index=False)
    actions.columns = ['user_id'] + ['u_feat5_' + str(i) for i in range(1, actions.shape[1])]
    return actions

# top  k 用户总购买/加购/关注/点击/浏览品牌数
def get_action_u0509_feat_23(start_date, end_date, k):
    dump_path = './cache/u0515_feat5_%s_%s_%s.csv' % (start_date, end_date,k)
    if os.path.exists(dump_path):
        actions = pd.read_csv(dump_path)
    else:
        start_days = pd.to_datetime(end_date) - timedelta(days=k)
        start_days = str(start_days).split(' ')[0]
        actions = get_action_user_feat5(start_days, end_date)
        actions.to_csv(dump_path, index=False)
    actions.columns = ['user_id'] + ['u0515_feat5_' + str(k) + '_' + str(i) for i in range(1, actions.shape[1])]
    return actions


# 点击模块
def get_action_u0509_feat_24(start_date, end_date, n):
    dump_path='./cache/user_feat0509_1_31_%s_%s_%s.csv'%(start_date,end_date,n)
    if os.path.exists(dump_path):
        actions = pd.read_csv(dump_path)
    else:
        start_days = datetime.strptime(end_date, '%Y-%m-%d') - timedelta(days=n)
        start_days = datetime.strftime(start_days, '%Y-%m-%d')
        actions = get_actions(start_days, end_date)
        actions = actions[actions['type'] == 6][['user_id', 'model_id']]

        #         actions = actions.drop('type',axis=1)

        actions_click_sum = actions[['user_id', 'model_id']].groupby('user_id').count().reset_index()
        actions_click_sum.columns = ['user_id', str(n) + 'click_sum_all']
        actions[str(n) + 'u_click14_history'] = actions['model_id'].map(lambda x: int(x == 14))
        actions[str(n) + 'u_click21_history'] = actions['model_id'].map(lambda x: int(x == 21))
        actions[str(n) + 'u_click28_history'] = actions['model_id'].map(lambda x: int(x == 28))
        actions[str(n) + 'u_click110_history'] = actions['model_id'].map(lambda x: int(x == 110))
        actions[str(n) + 'u_click210_history'] = actions['model_id'].map(lambda x: int(x == 210))
        actions = actions.groupby('user_id').sum().reset_index().drop('model_id', axis=1)
        #         actions.to_csv(dump_path,index=False)
        actions = pd.merge(actions, actions_click_sum, how='left', on='user_id')

        actions[str(n) + 'u_click14/click_sum_history'] = actions[str(n) + 'u_click14_history'] / actions[
            str(n) + 'click_sum_all']
        actions[str(n) + 'u_click21/click_sum_history'] = actions[str(n) + 'u_click21_history'] / actions[
            str(n) + 'click_sum_all']
        actions[str(n) + 'u_click28/click_sum_history'] = actions[str(n) + 'u_click28_history'] / actions[
            str(n) + 'click_sum_all']
        actions[str(n) + 'u_click110/click_sum_history'] = actions[str(n) + 'u_click110_history'] / actions[
            str(n) + 'click_sum_all']
        actions[str(n) + 'u_click210/click_sum_history'] = actions[str(n) + 'u_click210_history'] / actions[
            str(n) + 'click_sum_all']

        user_id = actions[['user_id']]
        del actions['user_id']
        actions = actions.fillna(0)
        columns = actions.columns
        min_max_scale = preprocessing.MinMaxScaler()
        actions = min_max_scale.fit_transform(actions.values)
        actions = pd.concat([user_id, pd.DataFrame(actions, columns=columns)], axis=1)
        actions.to_csv(dump_path, index=False)
    return actions


# u模型cate=8的购买者和不是cate=8的购买者
def get_action_u0509_feat_25(start_date, end_date):
    dump_path = './cache/u0509_feat25_%s_%s.csv' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = pd.read_csv(dump_path)
    else:
        df = get_actions(start_date, end_date)[['user_id', 'type', 'cate']]
        df = df[df['type'] == 4]
        df = df.groupby(['user_id', 'cate']).count()
        df = df.unstack().reset_index()
        df.columns = ['user_id'] + ['cate' + str(i) for i in range(4, 12)]
        df = df.fillna(0)
        sum1 = df.drop(['user_id', 'cate8'], axis=1).apply(sum, axis=1)
        sum2 = df.drop(['user_id'], axis=1).apply(sum, axis=1)
        actions = pd.concat([df[['user_id', 'cate8']], sum1, sum2], axis=1)
        actions.columns = ['user_id', 'cate8', 'sum_other_cate', 'sum']
        actions['cate8_rate'] = actions['cate8'] / actions['sum']
        actions['sum_other_cate_rate'] = actions['sum_other_cate'] / actions['sum']
        del actions['sum']
        actions=actions[['user_id','cate8_rate','sum_other_cate_rate']]
        actions.to_csv(dump_path, index=False)
    actions.columns=['user_id']+['u_feat25_' + str(i) for i in range(1, actions.shape[1])]
    return actions



# 用户层级特征(注意：'get_action_u0509_feat_27'调用'get_action_u0509_feat_26')
#def get_action_user_feat_six_xingwei(start_date, end_date, n)
def get_action_u0509_feat_26(start_date, end_date, n):
    dump_path = './cache/user_six_action_%s_%s_%s_int.csv' % (start_date, end_date, n)
    if os.path.exists(dump_path):
        actions = pd.read_csv(dump_path)
        print("user_zlzl" + str(n))

    else:
        actions = get_actions(start_date, end_date)
        actions['time'] = actions['time'].map(lambda x: get_day_chaju(x, end_date) // n)
        num_day = np.max(actions['time'])
        df = None
        print(num_day)
        for i in range(min(num_day + 1, 6)):
            in_temp = pd.get_dummies(actions['type'], prefix="user_action_time_" + str(i))
            temp = actions[actions['time'] == i]
            temp = pd.concat([temp['user_id'], in_temp], axis=1)

            feature = ['user_id']
            for j in range(1, 7, 1):
                feature.append('user_action_time_' + str(i) + '_' + str(j))

            temp = temp.groupby(['user_id'], as_index=False).sum()
            temp.columns = feature
            if df is None:
                df = temp
            else:
                df = pd.merge(df, temp, how='outer', on='user_id')
        df.columns = ['user_id'] + ['get_action_user_feat_six_xingwei_' + str(n) + '_' + str(i) for i in
                                    range(1, df.shape[1])]
        df.to_csv(dump_path, index=False)
        actions = df

    # user_id = actions[['user_id']]
    #     del actions['user_id']
    #     actions = actions.fillna(0)
    #     actions=actions.replace(np.inf,0)
    # #         print(actions.head())
    #     columns = actions.columns

    #     min_max_scale = preprocessing.MinMaxScaler()
    #     actions=actions.replace(np.inf,0)
    #     actions = min_max_scale.fit_transform(actions.values)
    #     actions = pd.concat([user_id, pd.DataFrame(actions,columns = columns)], axis=1)
    actions.columns = ['user_id'] + ['get_action_user_feat_six_xingwei_' + str(n) + '_' + str(i) for i in
                                     range(1, actions.shape[1])]
    return actions

#def deal_user_six_deal(start_date, end_date, n):
def get_action_u0509_feat_27(start_date, end_date, n):
    dump_path = './cache/deal_user_six_action_%s_%s_%s_int.csv' % (start_date, end_date, n)
    if os.path.exists(dump_path):
        actions = pd.read_csv(dump_path)
        actions.columns = ['user_id'] + ['u_featsix_deal_' + str(n) + '_' + str(i) for i in range(1, actions.shape[1])]
        return actions
    else:
        temp = get_action_u0509_feat_26(start_date, end_date, n)  # 修改
        time1 = datetime.now()
        columns = ["user_id"]
        all_col = temp.shape[1] - 1
        temp.columns = columns + list(range(all_col))
        temp = temp.fillna(0)
        columns = ['user_id']
        for j in range(0, 6, 1):
            temp["zl_" + str(j)] = 0
            columns.append("zl_" + str(j))
            for k in range(j, all_col, 6):
                temp["zl_" + str(j)] = temp["zl_" + str(j)] + temp[k].map(lambda x: x * ((k // 6 + 1) ** (-0.67)))
            temp["zl_" + str(j)] = temp["zl_" + str(j)].map(lambda x: (x - np.min(temp["zl_" + str(j)])) / (
                np.max(temp["zl_" + str(j)]) - np.min(temp["zl_" + str(j)])))
        temp = temp[columns]
        temp.to_csv(dump_path, index=False)
        return temp
    




print("US model  finish  part_1")



# In[ ]:




# In[3]:

#!/usr/bin/env python

#from basic_feat0518 import *


# top k 天的行为次数总和(滑窗处理)
def get_action_p0509_feat(start_date, end_date, k):
    dump_path = './cache/p_action_%s_%s_%s.csv' % (start_date, end_date, k)
    if os.path.exists(dump_path):
        actions = pd.read_csv(dump_path)
    else:
        start_days = pd.to_datetime(end_date) - timedelta(days=k)
        start_days = str(start_days).split(' ')[0]
        actions = get_actions(start_days, end_date)
        actions = actions[['sku_id', 'type']]
        df = pd.get_dummies(actions['type'], prefix='type')
        actions = pd.concat([actions, df], axis=1)  # type: pd.DataFrame
        actions = actions.groupby('sku_id', as_index=False).sum()
        min_max_scaler = preprocessing.MinMaxScaler()
        df = min_max_scaler.fit_transform(actions.drop(['sku_id', 'type'], axis=1).values)
        df = pd.DataFrame(df)

        actions = pd.concat([actions[['sku_id']], df], axis=1)
        actions.columns=['sku_id']+['p0509_' + str(k) + '_' + str(i) for i in range(1, actions.shape[1])]
        actions.to_csv(dump_path, index=False)
    return actions

#print get_action_p0509_feat('2016-02-01','2016-04-11',1)

# 商品的行为转化率
def get_action_product_feat_1(start_date, end_date):
    feature = ['sku_id', 'product_action_1_ratio', 'product_action_2_ratio', 'product_action_3_ratio',
               'product_action_5_ratio', 'product_action_6_ratio']
    dump_path = './cache/product_feat_accumulate_xiugai_%s_%s.csv' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = pd.read_csv(dump_path)
    else:
        actions = get_actions(start_date, end_date)
        df = pd.get_dummies(actions['type'], prefix='action')
        actions = pd.concat([actions['sku_id'], df], axis=1)
        actions = actions.groupby(['sku_id'], as_index=False).sum()
        actions['product_action_1_ratio'] = actions['action_4'] / actions['action_1']
        actions['product_action_2_ratio'] = actions['action_4'] / actions['action_2']
        #         actions['product_action_3_ratio'] = actions['action_4'] / actions['action_3']
        actions['product_action_3_ratio'] = actions['action_3'] / actions['action_2']
        actions['product_action_5_ratio'] = actions['action_4'] / actions['action_5']
        actions['product_action_6_ratio'] = actions['action_4'] / actions['action_6']
        actions = actions[feature]
        actions.to_csv(dump_path, index=False)
    actions.columns = ['sku_id'] + ['p_feat1_' + str(i) for i in range(1, actions.shape[1])]
    return actions

# 商品购买/加入购物车/关注前访问天数
def get_action_p0509_feat_2(start_date, end_date):
    dump_path = './cache/product_feat15_%s_%s.csv' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = pd.read_csv(dump_path)
    else:
        # 商品购买前访问天数
        def product_feat_2_1(start_date, end_date):
            actions = get_actions(start_date, end_date)[['sku_id', 'type', 'time']]
            actions['time'] = actions['time'].map(lambda x: x.split(' ')[0])
            visit = actions[actions['type'] == 1]
            visit = visit.drop_duplicates(['sku_id', 'time'], keep='first')
            del visit['time']
            del actions['time']
            visit = visit.groupby('sku_id', as_index=False).count()
            visit.columns = ['sku_id', 'visit']
            buy = actions[actions['type'] == 4]
            buy = buy.groupby('sku_id', as_index=False).count()
            buy.columns = ['sku_id', 'buy']
            actions = pd.merge(visit, buy, on='sku_id', how='left')
            actions['visit_day_before_buy'] = actions['visit'] / actions['buy']
            del actions['buy']
            del actions['visit']
            return actions

        # 商品加入购物车前访问天数
        def product_feat_2_2(start_date, end_date):
            actions = get_actions(start_date, end_date)[['sku_id', 'type', 'time']]
            actions['time'] = actions['time'].map(lambda x: x.split(' ')[0])
            visit = actions[actions['type'] == 1]
            visit = visit.drop_duplicates(['sku_id', 'time'], keep='first')
            del visit['time']
            del actions['time']
            visit = visit.groupby('sku_id', as_index=False).count()
            visit.columns = ['sku_id', 'visit']
            addtoshopping = actions[actions['type'] == 2]
            addtoshopping = addtoshopping.groupby('sku_id', as_index=False).count()
            addtoshopping.columns = ['sku_id', 'addtoshopping']
            actions = pd.merge(visit, addtoshopping, on='sku_id', how='left')
            actions['visit_day_before_addtoshopping'] = actions['visit'] / actions['addtoshopping']
            del actions['addtoshopping']
            del actions['visit']
            return actions

        # 商品关注前访问天数
        def product_feat_2_3(start_date, end_date):
            actions = get_actions(start_date, end_date)[['sku_id', 'type', 'time']]
            actions['time'] = actions['time'].map(lambda x: x.split(' ')[0])
            visit = actions[actions['type'] == 1]
            visit = visit.drop_duplicates(['sku_id', 'time'], keep='first')
            del visit['time']
            del actions['time']
            visit = visit.groupby('sku_id', as_index=False).count()
            visit.columns = ['sku_id', 'visit']
            guanzhu = actions[actions['type'] == 5]
            guanzhu = guanzhu.groupby('sku_id', as_index=False).count()
            guanzhu.columns = ['sku_id', 'guanzhu']
            actions = pd.merge(visit, guanzhu, on='sku_id', how='left')
            actions['visit_day_before_guanzhu'] = actions['visit'] / actions['guanzhu']
            del actions['guanzhu']
            del actions['visit']
            return actions

        # 用户购买前加入购物车天数
        def product_feat_2_4(start_date, end_date):
            actions = get_actions(start_date, end_date)[['sku_id', 'type', 'time']]
            actions['time'] = actions['time'].map(lambda x: x.split(' ')[0])
            # actions=actions.drop_duplicates(['user_id','time'],keep='first')
            addtoshopping = actions[actions['type'] == 2]
            addtoshopping = addtoshopping.drop_duplicates(['sku_id', 'time'], keep='first')
            del addtoshopping['time']
            del actions['time']
            addtoshopping = addtoshopping.groupby('sku_id', as_index=False).count()
            addtoshopping.columns = ['sku_id', 'addtoshopping']
            buy = actions[actions['type'] == 4]
            buy = buy.groupby('sku_id', as_index=False).count()
            buy.columns = ['sku_id', 'buy']
            actions = pd.merge(addtoshopping, buy, on='sku_id', how='left')
            actions['addtoshopping_day_before_buy'] = actions['addtoshopping'] / actions['buy']
            del actions['buy']
            del actions['addtoshopping']
            return actions

        # 用户购买前关注天数
        def product_feat_2_5(start_date, end_date):
            actions = get_actions(start_date, end_date)[['sku_id', 'type', 'time']]
            actions['time'] = actions['time'].map(lambda x: x.split(' ')[0])
            guanzhu = actions[actions['type'] == 5]
            guanzhu = guanzhu.drop_duplicates(['sku_id', 'time'], keep='first')
            del guanzhu['time']
            del actions['time']
            guanzhu = guanzhu.groupby('sku_id', as_index=False).count()
            guanzhu.columns = ['sku_id', 'guanzhu']
            buy = actions[actions['type'] == 4]
            buy = buy.groupby('sku_id', as_index=False).count()
            buy.columns = ['sku_id', 'buy']
            actions = pd.merge(guanzhu, buy, on='sku_id', how='left')
            actions['guanzhu_day_before_buy'] = actions['guanzhu'] / actions['buy']
            del actions['buy']
            del actions['guanzhu']
            return actions

        actions = pd.merge(product_feat_2_1(start_date, end_date), product_feat_2_2(start_date, end_date),
                           on='sku_id', how='outer')
        actions = pd.merge(actions, product_feat_2_3(start_date, end_date), on='sku_id', how='outer')
        actions = pd.merge(actions, product_feat_2_4(start_date, end_date), on='sku_id', how='outer')
        actions = pd.merge(actions, product_feat_2_5(start_date, end_date), on='sku_id', how='outer')
        sku_id = actions['sku_id']
        del actions['sku_id']
        actions = actions.fillna(0)
        min_max_scale = preprocessing.MinMaxScaler()
        actions = min_max_scale.fit_transform(actions.values)
        actions = pd.concat([sku_id, pd.DataFrame(actions)], axis=1)
        actions.to_csv(dump_path, index=False)
    actions.columns = ['sku_id'] + ['p_feat2_' + str(i) for i in range(1, actions.shape[1])]
    return actions



# 商品平均访问间隔
def get_action_p0509_feat_6(start_date, end_date):
    dump_path = './cache/product_feat7_%s_%s.csv' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = pd.read_csv(dump_path)
    else:
        df = get_actions(start_date, end_date)[['sku_id', 'time']]
        df['time'] = df['time'].map(lambda x: x.split(' ')[0])
        df = df.drop_duplicates(['sku_id', 'time'], keep='first')
        df['time'] = df['time'].map(lambda x: datetime.strptime(x, '%Y-%m-%d'))
        actions = df.groupby('sku_id', as_index=False).agg(lambda x: x['time'].diff().mean())
        actions['avg_visit'] = actions['time'].dt.days
        del actions['time']
        actions.to_csv(dump_path, index=False)
    actions.columns = ['sku_id'] + ['p_feat6_' + str(i) for i in range(1, actions.shape[1])]
    return actions

# 商品六种行为平均访问间隔
def get_action_p0509_feat_6_six(start_date, end_date):
    dump_path = './cache/product_feat7_six_%s_%s.csv' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = pd.read_csv(dump_path)
    else:
        df = get_actions(start_date, end_date)[['sku_id', 'time', 'type']]
        df['time'] = df['time'].map(lambda x: (-1) * get_day_chaju(x, start_date))
        df = df.drop_duplicates(['sku_id', 'time', 'type'], keep='first')
        actions = df.groupby(['sku_id', 'type']).agg(lambda x: np.diff(x).mean())
        actions = actions.unstack()
        actions.columns = list(range(actions.shape[1]))
        actions = actions.reset_index()
        actions.to_csv(dump_path, index=False)
    actions.columns = ['sku_id'] + ['p_feat6_six_' + str(i) for i in range(1, actions.shape[1])]
    return actions



# 最近K天
def product_top_k_0_1(start_date, end_date):
    actions = get_actions(start_date, end_date)
    actions = actions[['user_id', 'sku_id', 'type']]
    df = pd.get_dummies(actions['type'], prefix='%s-%s-action' % (start_date, end_date))
    actions = pd.concat([actions, df], axis=1)  # type: pd.DataFrame
    actions = actions.groupby('sku_id', as_index=False).sum()
    del actions['type']
    del actions['user_id']
    sku_id = actions['sku_id']
    del actions['sku_id']
    actions = actions.applymap(lambda x: 1 if x > 0 else 0)
    actions = pd.concat([sku_id, actions], axis=1)
    return actions


# 最近K天行为0/1提取
def get_action_p0509_feat_8(start_date, end_date):
    dump_path = './cache/product_feat9_%s_%s.csv' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = pd.read_csv(dump_path)
    else:
        actions = None
        for i in (1, 2, 3, 4, 5, 6, 7, 15, 30):
            print(i)
            start_days = datetime.strptime(end_date, '%Y-%m-%d') - timedelta(days=i)
            start_days = start_days.strftime('%Y-%m-%d')
            if actions is None:
                actions = product_top_k_0_1(start_days, end_date)
            else:
                actions = pd.merge(actions, product_top_k_0_1(start_days, end_date), how='outer', on='sku_id')
        actions.to_csv(dump_path, index=False)
    actions.columns = ['sku_id'] + ['p_feat8_' + str(i) for i in range(1, actions.shape[1])]
    return actions



# 商品的重复购买率
def get_action_p0509_feat_8_2(start_date, end_date):
    dump_path = './cache/product_feat8_%s_%s.csv' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = pd.read_csv(dump_path)
    else:
        df = get_actions(start_date, end_date)[['user_id', 'sku_id', 'type']]
        df = df[df['type'] == 4]  # 购买的行为
        df = df.groupby(['user_id', 'sku_id'], as_index=False).count()
        df.columns = ['user_id', 'sku_id', 'count1']
        df['count1'] = df['count1'].map(lambda x: 1 if x > 1 else 0)
        grouped = df.groupby(['sku_id'], as_index=False)
        actions = grouped.count()[['sku_id', 'count1']]
        actions.columns = ['sku_id', 'count']
        re_count = grouped.sum()[['sku_id', 'count1']]
        re_count.columns = ['sku_id', 're_count']
        actions = pd.merge(actions, re_count, on='sku_id', how='left')
        re_buy_rate = actions['re_count'] / actions['count']
        actions = pd.concat([actions['sku_id'], re_buy_rate], axis=1)
        actions.columns = ['sku_id', 're_buy_rate']
        actions.to_csv(dump_path, index=False)
    actions.columns = ['sku_id'] + ['p_feat8_2_' + str(i) for i in range(1, actions.shape[1])]
    return actions




# 获取货物最近一次行为的时间距离当前时间的差距
def get_action_p0509_feat_9(start_date, end_date):
    dump_path = './cache/product_feat9_2_%s_%s.csv' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = pd.read_csv(dump_path)
    else:
        df = get_actions(start_date, end_date)[['sku_id', 'time', 'type']]
        # df['time'] = df['time'].map(lambda x: (-1)*get_day_chaju(x,start_date))
        df = df.drop_duplicates(['sku_id', 'type'], keep='last')
        df['time'] = df['time'].map(lambda x: get_day_chaju(x, end_date) + 1)
        actions = df.groupby(['sku_id', 'type']).sum()
        actions = actions.unstack()
        actions.columns = list(range(actions.shape[1]))
        actions = actions.reset_index()
        actions = actions.fillna(30)
        actions.to_csv(dump_path, index=False)
    actions.columns = ['sku_id'] + ['p_feat9_' + str(i) for i in range(1, actions.shape[1])]
    return actions



# 获取货品最后一次行为的次数并且进行归一化
def get_action_product_feat_10(start_date, end_date):
    dump_path = './cache/product_feat10_%s_%s.csv' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = pd.read_csv(dump_path)
    else:

        df = get_actions(start_date, end_date)[['sku_id', 'time', 'type']]
        df['time'] = df['time'].map(lambda x: get_day_chaju(x, end_date) + 1)

        idx = df.groupby(['sku_id', 'type'])['time'].transform(min)
        idx1 = idx == df['time']
        actions = df[idx1].groupby(["sku_id", "type"]).count()
        actions = actions.unstack()
        actions.columns = list(range(actions.shape[1]))
        actions = actions.fillna(0)
        actions = actions.reset_index()

        user_sku = actions[['sku_id']]
        del actions['sku_id']
        min_max_scaler = preprocessing.MinMaxScaler()
        actions = min_max_scaler.fit_transform(actions.values)
        actions = pd.DataFrame(actions)
        actions = pd.concat([user_sku, actions], axis=1)

        actions.to_csv(dump_path, index=False)
    actions.columns = ['sku_id'] + ['p_feat10_' + str(i) for i in range(1, actions.shape[1])]
    return actions



# 获取物品该层级最后一层的各种行为的统计数量
def get_action_product_feat_11(start_date, end_date, n):
    dump_path = './cache/product_feat11_%s_%s_%s.csv' % (start_date, end_date, n)
    if os.path.exists(dump_path):
        actions = pd.read_csv(dump_path)
    else:
        df = get_actions(start_date, end_date)[['sku_id', 'time', 'type']]
        df['time'] = df['time'].map(lambda x: get_day_chaju(x, end_date) // n)
        df = df[df['time'] == 0]
        del df['time']
        temp = pd.get_dummies(df['type'], prefix='type')
        del df['type']
        actions = pd.concat([df, temp], axis=1)
        actions = actions.groupby(['sku_id'], as_index=False).sum()
        user_sku = actions[['sku_id']]
        del actions['sku_id']
        min_max_scaler = preprocessing.MinMaxScaler()
        actions = min_max_scaler.fit_transform(actions.values)
        actions = pd.DataFrame(actions)
        actions = pd.concat([user_sku, actions], axis=1)
        actions.to_csv(dump_path, index=False)
    actions.columns = ['sku_id'] + ['p_feat11_' + str(n) + '_' + str(i) for i in range(1, actions.shape[1])]
    return actions


#(hcq)用户2/3/7/14/28层级行为次数
def get_action_product_feat_12(start_date, end_date):
    dump_path = './cache/p0509_feat12_%s_%s.csv' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = pd.read_csv(dump_path)
    else:
        actions = get_actions(start_date, end_date)[['sku_id', 'time', 'type']]
        actions['time'] = actions['time'].map(lambda x: x.split(' ')[0])
        actions = actions.drop_duplicates(['sku_id', 'time', 'type'], keep='first')
        actions['day'] = actions['time'].map(
            lambda x: (datetime.strptime(end_date, '%Y-%m-%d') - datetime.strptime(x, '%Y-%m-%d')).days)
        result = None
        for i in (2, 3, 7, 14, 28):  # 层级个数
            print ('i%s' % i)
            actions['level%s' % i] = actions['day'].map(lambda x: x // i)
            a = set(actions['level%s' % i].tolist())
            for j in (1, 2, 3, 4, 5, 6):  # type
                print ('j%s' % j)
                df = actions[actions['type'] == j][['sku_id', 'level%s' % i, 'time']]
                df = df.groupby(['sku_id', 'level%s' % i]).count()
                df = df.unstack()
                b = df.columns.levels[1].tolist()
                df.columns = ['p_feat12_' + str('level%s_' % i) + str(j) + '_' + str(k) for k in
                              df.columns.levels[1].tolist()]
                if len(list(a - set(b))) != 0:
                    c = list(a - set(b))
                    for k in c:
                        df['p_feat12_' + str('level%s_' % i) + str(j) + '_' + str(k)] = 0
                columns = df.columns
                dict = {}
                for column in columns:
                    k = int(column.split('_')[-1])
                    dict[column] = k
                columns = sorted(dict.items(), key=lambda x: x[1])
                columns = [(columns[t])[0] for t in range(len(columns))]
                df = df[columns]
                df = df.reset_index()
                if result is None:
                    result = df
                else:
                    result = pd.merge(result, df, on='sku_id', how='left')
        columns = result.columns
        sku_id = result['sku_id']
        del result['sku_id']
        actions = result.fillna(0)

        min_max_scaler = preprocessing.MinMaxScaler()
        actions = min_max_scaler.fit_transform(actions.values)
        actions = pd.DataFrame(actions)
        actions = pd.concat([sku_id, actions], axis=1)
        actions.columns = columns
        actions.to_csv(dump_path, index=False)
    return actions





# 层级天数
def get_action_product_feat_13(start_date, end_date, n):
    dump_path = './cache/sku_feat13_%s_%s_%s.csv' % (start_date, end_date, n)
    if os.path.exists(dump_path):
        actions = pd.read_csv(dump_path)
    else:
        df = get_actions(start_date, end_date)[['sku_id', 'time', 'type']]
        df['time'] = df['time'].map(lambda x: get_day_chaju(x, end_date) // n)
        df = df.drop_duplicates(['sku_id', 'type', 'time'], keep='first')
        actions = df.groupby(['sku_id', 'type']).count()
        actions = actions.unstack()
        actions.columns = list(range(actions.shape[1]))
        actions = actions.fillna(0)
        actions = actions.reset_index()
        user_sku = actions[['sku_id']]
        del actions['sku_id']
        min_max_scaler = preprocessing.MinMaxScaler()
        actions = min_max_scaler.fit_transform(actions.values)
        actions = pd.DataFrame(actions)
        actions = pd.concat([user_sku, actions], axis=1)
        actions.to_csv(dump_path, index=False)
    actions.columns = ['sku_id'] + ['p_feat13_' + str(n) + '_' + str(i) for i in range(1, actions.shape[1])]
    return actions

#用户每隔5天购买购买次数
def get_action_product_feat_14(start_date, end_date):
    dump_path = './cache/sku_feat14_%s_%s.csv' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = pd.read_csv(dump_path)
    else:
        n = 5
        df = get_actions(start_date, end_date)[['sku_id', 'time', 'type']]
        df = df[df['type'] == 4][['sku_id', 'time']]
        df['time'] = df['time'].map(lambda x: get_day_chaju(x, end_date) // n)
        days = np.max(df['time'])

        df['cnt'] = 0
        actions = df.groupby(['sku_id', 'time']).count()

        actions = actions.unstack()

        actions.columns = list(range(actions.shape[1]))
        actions = actions.reset_index()

        actions = actions.fillna(0)
        user_sku = actions[['sku_id']]
        del actions['sku_id']
        min_max_scaler = preprocessing.MinMaxScaler()
        actions = min_max_scaler.fit_transform(actions.values)
        actions = pd.DataFrame(actions)
        actions = pd.concat([user_sku, actions], axis=1)
        actions.to_csv(dump_path, index=False)
    actions.columns = ['sku_id'] + ['p_feat14_' + str(i) for i in range(1, actions.shape[1])]
    return actions


# 商品购买/加入购物车/关注前访问次数
def get_action_p0509_feat_15(start_date, end_date):
    dump_path = './cache/p0509_feat15_%s_%s.csv' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = pd.read_csv(dump_path)
    else:
        # 商品购买前访问次数
        def p0509_feat_15_1(start_date, end_date):
            actions = get_actions(start_date, end_date)[['sku_id', 'type']]
            visit = actions[actions['type'] == 1]
            visit = visit.groupby('sku_id', as_index=False).count()
            visit.columns = ['sku_id', 'visit']
            buy = actions[actions['type'] == 4]
            buy = buy.groupby('sku_id', as_index=False).count()
            buy.columns = ['sku_id', 'buy']
            actions = pd.merge(visit, buy, on='sku_id', how='left')
            actions['visit_num_before_buy'] = actions['visit'] / actions['buy']
            del actions['buy']
            del actions['visit']
            return actions

        # 商品加入购物车前访问次数
        def p0509_feat_15_2(start_date, end_date):
            actions = get_actions(start_date, end_date)[['sku_id', 'type']]
            visit = actions[actions['type'] == 1]
            visit = visit.groupby('sku_id', as_index=False).count()
            visit.columns = ['sku_id', 'visit']
            addtoshopping = actions[actions['type'] == 2]
            addtoshopping = addtoshopping.groupby('sku_id', as_index=False).count()
            addtoshopping.columns = ['sku_id', 'addtoshopping']
            actions = pd.merge(visit, addtoshopping, on='sku_id', how='left')
            actions['visit_num_before_addtoshopping'] = actions['visit'] / actions['addtoshopping']
            del actions['addtoshopping']
            del actions['visit']
            return actions

        # 商品关注前访问次数
        def p0509_feat_15_3(start_date, end_date):
            actions = get_actions(start_date, end_date)[['sku_id', 'type']]
            visit = actions[actions['type'] == 1]
            visit = visit.groupby('sku_id', as_index=False).count()
            visit.columns = ['sku_id', 'visit']
            guanzhu = actions[actions['type'] == 5]
            guanzhu = guanzhu.groupby('sku_id', as_index=False).count()
            guanzhu.columns = ['sku_id', 'guanzhu']
            actions = pd.merge(visit, guanzhu, on='sku_id', how='left')
            actions['visit_num_before_guanzhu'] = actions['visit'] / actions['guanzhu']
            del actions['guanzhu']
            del actions['visit']
            return actions

        # 用户购买前加入购物车次数
        def p0509_feat_15_4(start_date, end_date):
            actions = get_actions(start_date, end_date)[['sku_id', 'type']]
            addtoshopping = actions[actions['type'] == 2]
            addtoshopping = addtoshopping.groupby('sku_id', as_index=False).count()
            addtoshopping.columns = ['sku_id', 'addtoshopping']
            buy = actions[actions['type'] == 4]
            buy = buy.groupby('sku_id', as_index=False).count()
            buy.columns = ['sku_id', 'buy']
            actions = pd.merge(addtoshopping, buy, on='sku_id', how='left')
            actions['addtoshopping_num_before_buy'] = actions['addtoshopping'] / actions['buy']
            del actions['buy']
            del actions['addtoshopping']
            return actions

        # 用户购买前关注次数
        def p0509_feat_15_5(start_date, end_date):
            actions = get_actions(start_date, end_date)[['sku_id', 'type']]
            guanzhu = actions[actions['type'] == 5]
            guanzhu = guanzhu.groupby('sku_id', as_index=False).count()
            guanzhu.columns = ['sku_id', 'guanzhu']
            buy = actions[actions['type'] == 4]
            buy = buy.groupby('sku_id', as_index=False).count()
            buy.columns = ['sku_id', 'buy']
            actions = pd.merge(guanzhu, buy, on='sku_id', how='left')
            actions['guanzhu_num_before_buy'] = actions['guanzhu'] / actions['buy']
            del actions['buy']
            del actions['guanzhu']
            return actions

        actions = pd.merge(p0509_feat_15_1(start_date, end_date), p0509_feat_15_2(start_date, end_date), on='sku_id',
                           how='outer')
        actions = pd.merge(actions, p0509_feat_15_3(start_date, end_date), on='sku_id', how='outer')
        actions = pd.merge(actions, p0509_feat_15_4(start_date, end_date), on='sku_id', how='outer')
        actions = pd.merge(actions, p0509_feat_15_5(start_date, end_date), on='sku_id', how='outer')
        sku_id = actions['sku_id']
        del actions['sku_id']
        actions = actions.fillna(0)
        min_max_scale = preprocessing.MinMaxScaler()
        actions = min_max_scale.fit_transform(actions.values)
        actions = pd.concat([sku_id, pd.DataFrame(actions)], axis=1)

        actions.to_csv(dump_path, index=False)
    actions.columns = ['sku_id'] + ['p_feat15_' + str(i) for i in range(1, actions.shape[1])]
    return actions


# 商品行为的交叉
def get_action_product_feat_16(start_date, end_date):
    dump_path = './cache/product_feat16_%s_%s.csv' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = pd.read_csv(dump_path)
    else:
        actions = get_actions(start_date, end_date)[['sku_id', 'type']]
        actions['cnt'] = 0
        action1 = actions.groupby(['sku_id', 'type']).count()
        action1 = action1.unstack()
        index_col = list(range(action1.shape[1]))
        action1.columns = index_col
        action1 = action1.reset_index()
        action2 = actions.groupby('sku_id', as_index=False).count()
        del action2['type']
        action2.columns = ['sku_id', 'cnt']
        actions = pd.merge(action1, action2, how='left', on='sku_id')
        for i in index_col:
            actions[i] = actions[i] / actions['cnt']
        del actions['cnt']
        actions.to_csv(dump_path, index=False)
    actions.columns = ['sku_id'] + ['p_feat16_' + str(i) for i in range(1, actions.shape[1])]
    return actions


# 老顾客比率
def get_action_p0509_feat_17(start_date, end_date):
    dump_path = './cache/product_feat4_%s_%s.csv' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = pd.read_csv(dump_path)
    else:
        actions = get_actions(start_date, end_date)[['user_id', 'sku_id', 'type']]
        actions = actions[actions['type'] == 4]
        df = actions.groupby(['user_id', 'sku_id'], as_index=False).count()
        df.columns = ['user_id', 'sku_id', 'number']
        df2 = df[df['number'] > 1]
        del df['number']
        del df2['number']
        df1 = df.groupby('sku_id', as_index=False).count()
        df1.columns = ['sku_id', 'all_number']
        df2 = df2.groupby('sku_id', as_index=False).count()
        df2.columns = ['sku_id', 'number']
        actions = pd.merge(df1, df2, on='sku_id', how='left')
        actions['rebuy_rate'] = actions['number'] / actions['all_number']
        del actions['number']
        del actions['all_number']
        actions.to_csv(dump_path, index=False)
    actions.columns = ['sku_id'] + ['p_feat17_' + str(i) for i in range(1, actions.shape[1])]
    return actions

#get_action_p0509_feat_18

# 商品点击到购买的时间间隔
def get_action_p0509_feat_19(start_date, end_date):
    dump_path = './cache/p0509_feat_19_%s_%s.csv' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = pd.read_csv(dump_path)
    else:
        actions = get_actions(start_date, end_date)
        actions_dianji = actions[actions['type'] == 6][['user_id', 'sku_id', 'time']]
        actions_dianji['time_dianji'] = actions_dianji['time'].map(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
        actions_dianji = actions_dianji[['user_id', 'sku_id', 'time_dianji']]
        actions_dianji = actions_dianji.drop_duplicates(['user_id', 'sku_id'], keep='first')

        actions_goumai = actions[actions['type'] == 4][['user_id', 'sku_id', 'time']]
        actions_goumai['time_goumai'] = actions_goumai['time'].map(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
        actions_goumai = actions_goumai[['user_id', 'sku_id', 'time_goumai']]
        actions_goumai = actions_goumai.drop_duplicates(['user_id', 'sku_id'], keep='last')

        actions = pd.merge(actions_dianji, actions_goumai, on=['user_id', 'sku_id'], how='inner')
        actions['time_jiange'] = actions['time_goumai'] - actions['time_dianji']
        actions = actions.drop(['user_id', 'time_goumai', 'time_dianji'], axis=1)
        actions['time_jiange'] = actions['time_jiange'].map(lambda x: x.days * 24 + x.seconds // 3600 + 1)

        actions_min = actions.groupby('sku_id').min().reset_index()
        actions_min.columns = ['sku_id', 'time_min']
        # actions_mean = actions.groupby('user_id').mean().reset_index()
        # actions_mean.columns = ['user_id','time_mean']
        actions_max = actions.groupby('sku_id').max().reset_index()
        actions_max.columns = ['sku_id', 'time_max']
        actions = pd.merge(actions_min, actions_max, on='sku_id', how='left')

        sku_id = actions[['sku_id']]
        del actions['sku_id']
        actions = actions.fillna(0)
        actions=actions.astype('float')
        columns = actions.columns
        min_max_scale = preprocessing.MinMaxScaler()
        actions = min_max_scale.fit_transform(actions.values)
        actions = pd.concat([sku_id, pd.DataFrame(actions, columns=columns)], axis=1)
        actions.to_csv(dump_path, index=False)
    actions.columns = ['sku_id'] + ['p_feat19_' + str(i) for i in range(1, actions.shape[1])]
    return actions




# 获取某商品某段时间内加入购物车的数量以及关注的数量
def get_action_p0509_feat_21(start_date, end_date, n):
    dump_path = './cache/p0509_feat_21_%s_%s_%s.csv' % (start_date, end_date, n)
    if os.path.exists(dump_path):
        actions = pd.read_csv(dump_path)
    else:
        start_days = datetime.strptime(end_date, '%Y-%m-%d') - timedelta(days=n)
        start_days = datetime.strftime(start_days, '%Y-%m-%d')

        actions = get_actions(start_days, end_date)[['sku_id', 'type', 'cate']]
        actions_gouwuche = actions[actions['type'] == 2]
        actions_gouwuche_1 = actions_gouwuche[['sku_id', 'type']]
        actions_gouwuche_1 = actions_gouwuche_1.groupby('sku_id').count().reset_index()
        actions_gouwuche_1.columns = ['sku_id', str(n) + 'gouwuche_add']

        actions_gouwuche = actions_gouwuche[actions_gouwuche['cate'] == 8]
        actions_gouwuche_2=actions_gouwuche[['sku_id', 'type']]

        actions_gouwuche_2 = actions_gouwuche_2.groupby('sku_id').count().reset_index()
        actions_gouwuche_2.columns = ['sku_id', str(n) + 'gouwuche_add_cate_8']

        actions_guanzhu = actions[actions['type'] == 5]
        actions_guanzhu_1 = actions_guanzhu[['sku_id', 'type']]
        actions_guanzhu_1 = actions_guanzhu_1.groupby('sku_id').count().reset_index()
        actions_guanzhu_1.columns = ['sku_id', str(n) + 'guanzhu_add']

        actions_guanzhu = actions_guanzhu[actions_guanzhu['cate'] == 8]
        actions_guanzhu_2=actions_guanzhu[['sku_id', 'type']]
        actions_guanzhu_2 = actions_guanzhu_2.groupby('sku_id').count().reset_index()
        actions_guanzhu_2.columns = ['sku_id', str(n) + 'guanzhu_add_cate_8']

        actions = pd.merge(actions_gouwuche_1, actions_gouwuche_2, on='sku_id', how='outer')
        actions = pd.merge(actions, actions_guanzhu_1, on='sku_id', how='outer')
        actions = pd.merge(actions, actions_guanzhu_2, on='sku_id', how='outer')
        actions = actions.fillna(0)

        sku_id = actions[['sku_id']]
        del actions['sku_id']
        actions = actions.fillna(0)
        actions=actions.astype('float')
        columns = actions.columns
        min_max_scale = preprocessing.MinMaxScaler()
        actions = min_max_scale.fit_transform(actions.values)
        actions = pd.concat([sku_id, pd.DataFrame(actions, columns=columns)], axis=1)
        actions.to_csv(dump_path, index=False)
    return actions





# 商品总购买/加购/关注/点击/浏览品牌数
def get_action_p0509_feat5(start_date, end_date):
    dump_path = './cache/p0509_feat5_a_%s_%s.csv' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = pd.read_csv(dump_path)
    else:
        actions = get_actions(start_date, end_date)
        action = None
        for i in (1, 2, 4, 5, 6):
            df = actions[actions['type'] == i][['user_id', 'sku_id']]
            df = df.drop_duplicates(['user_id', 'sku_id'], keep='first')
            df = df.groupby('sku_id', as_index=False).count()
            df.columns = ['sku_id', 'num_%s' % i]
            if i == 1:
                action = df
            else:
                action = pd.merge(action, df, on='sku_id', how='outer')
        actions = action.fillna(0)
        sku = actions[['sku_id']]
        min_max_scaler = preprocessing.MinMaxScaler()
        actions = min_max_scaler.fit_transform(actions.drop(['sku_id'], axis=1).values)
        actions = pd.DataFrame(actions)
        actions = pd.concat([sku, actions], axis=1)
        actions.to_csv(dump_path, index=False)
    actions.columns = ['sku_id'] + ['p_feat5_' + str(i) for i in range(1, actions.shape[1])]
    return actions


# top  k 商品总购买/加购/关注/点击/浏览品牌数
def get_action_p0509_feat_23(start_date, end_date, k):
    dump_path = './cache/p0509_feat23_%s_%s_%s.csv' % (start_date, end_date, k)
    if os.path.exists(dump_path):
        actions = pd.read_csv(dump_path)
    else:
        start_days = pd.to_datetime(end_date) - timedelta(days=k)
        start_days = str(start_days).split(' ')[0]
        actions = get_action_p0509_feat5(start_days, end_date)
        actions.to_csv(dump_path, index=False)
    actions.columns = ['sku_id'] + ['p0509_feat_23_' + str(k) + '_' + str(i) for i in range(1, actions.shape[1])]
    return actions



# 点击模块
def get_action_p0509_feat_24(start_date, end_date, n):
    dump_path = './cache/p0509_feat_24_%s_%s_%s.csv' % (start_date, end_date, n)
    if os.path.exists(dump_path):
        actions = pd.read_csv(dump_path)
    else:
        start_days = datetime.strptime(end_date, '%Y-%m-%d') - timedelta(days=n)
        start_days = datetime.strftime(start_days, '%Y-%m-%d')
        actions = get_actions(start_days, end_date)
        actions = actions[actions['type'] == 6][['sku_id', 'model_id']]

        actions_click_sum = actions[['sku_id', 'model_id']].groupby('sku_id').count().reset_index()
        actions_click_sum.columns = ['sku_id', str(n) + 'click_sum_all']
        actions[str(n) + 'p_click14_history'] = actions['model_id'].map(lambda x: int(x == 14))
        actions[str(n) + 'p_click21_history'] = actions['model_id'].map(lambda x: int(x == 21))
        actions[str(n) + 'p_click28_history'] = actions['model_id'].map(lambda x: int(x == 28))
        actions[str(n) + 'p_click110_history'] = actions['model_id'].map(lambda x: int(x == 110))
        actions[str(n) + 'p_click210_history'] = actions['model_id'].map(lambda x: int(x == 210))
        actions = actions.groupby('sku_id').sum().reset_index().drop('model_id', axis=1)

        actions = pd.merge(actions, actions_click_sum, how='left', on='sku_id')

        actions[str(n) + 'p_click14/click_sum_history'] = actions[str(n) + 'p_click14_history'] / actions[
            str(n) + 'click_sum_all']
        actions[str(n) + 'p_click21/click_sum_history'] = actions[str(n) + 'p_click21_history'] / actions[
            str(n) + 'click_sum_all']
        actions[str(n) + 'p_click28/click_sum_history'] = actions[str(n) + 'p_click28_history'] / actions[
            str(n) + 'click_sum_all']
        actions[str(n) + 'p_click110/click_sum_history'] = actions[str(n) + 'p_click110_history'] / actions[
            str(n) + 'click_sum_all']
        actions[str(n) + 'p_click210/click_sum_history'] = actions[str(n) + 'p_click210_history'] / actions[
            str(n) + 'click_sum_all']

        sku_id = actions[['sku_id']]
        del actions['sku_id']
        actions = actions.fillna(0)
        columns = actions.columns
        min_max_scale = preprocessing.MinMaxScaler()
        actions = min_max_scale.fit_transform(actions.values)
        actions = pd.concat([sku_id, pd.DataFrame(actions, columns=columns)], axis=1)
        actions.to_csv(dump_path, index=False)
    actions.columns = ['sku_id'] + ['p0509_feat_24_' + str(n) + '_' + str(i) for i in range(1, actions.shape[1])]
    return actions



# 获取每个商品的被购买的六种行为get_action_sku_feat_six_xingwei
def get_action_p0509_feat_26(start_date, end_date, n):
    dump_path = './cache/sku_six_action_%s_%s_%s_int.csv' % (start_date, end_date, n)
    if os.path.exists(dump_path):
        actions = pd.read_csv(dump_path)
        print("sku_zlzl_" + str(n))
        return actions
    else:
        actions = get_actions(start_date, end_date)

        actions['time'] = actions['time'].map(lambda x: get_day_chaju(x, end_date) // n)
        num_day = np.max(actions['time'])
        df = None
        #         print(num_day)
        for i in range(min(num_day + 1, 6)):

            in_temp = pd.get_dummies(actions['type'], prefix="sku_action_time_" + str(i))
            temp = actions[actions['time'] == i]
            temp = pd.concat([temp['sku_id'], in_temp], axis=1)

            feature = ['sku_id']
            for j in range(1, 7, 1):
                feature.append('sku_action_time_' + str(i) + '_' + str(j))

            temp = temp.groupby(['sku_id'], as_index=False).sum()
            #             temp['user_id']=temp['user_id'].astype('int')
            temp.columns = feature
            #             print(temp)
            #           用于归一化
            #             for j in range(1,7,1):
            # #                 min_x=np.min(temp['user_action_time_'+str(j)+'_'+str(i)])
            # #                 max_x=np.max(temp['user_action_time_'+str(j)+'_'+str(i)])
            #                 temp['sku_action_time_'+str(i)+'_'+str(j)]=temp['sku_action_time_'+str(i)+'_'+str(j)].map(lambda x: (x - np.min(temp['sku_action_time_'+str(i)+'_'+str(j)])) / (np.max(temp['sku_action_time_'+str(i)+'_'+str(j)])-np.min(temp['sku_action_time_'+str(i)+'_'+str(j)])))
            if df is None:
                df = temp
            else:
                df = pd.merge(df, temp, how='outer', on='sku_id')
        df.to_csv(dump_path, index=False)
        return df

#deal_sku_six_deal
def get_action_p0509_feat_27(start_date, end_date, n):
    dump_path = './cache/deal_sku_six_action_%s_%s_%s_int.csv' % (start_date, end_date, n)
    if os.path.exists(dump_path):
        actions = pd.read_csv(dump_path)
        print("wxl_" + str(n))
        actions.columns = ['sku_id'] + ['p_featsixdeal_' + str(n) + '_' + str(i) for i in range(1, actions.shape[1])]
        return actions
    else:
        sku_temp = get_action_p0509_feat_26(start_date, end_date, n)  # 修改
        columns = ["sku_id"]
        all_col = sku_temp.shape[1] - 1
        #         print(sku_temp.head(10))
        sku_temp.columns = columns + list(range(all_col))
        #         print(sku_temp.head(10))
        sku_temp = sku_temp.fillna(0)
        columns = ['sku_id']
        for j in range(0, 6, 1):
            sku_temp["zl_" + str(j)] = 0
            columns.append("zl_" + str(j))
            for k in range(j, all_col, 6):
                #                 print(sku_temp[k].head(1))
                #                 print(sku_temp["zl_"+str(j)].head(1))
                sku_temp["zl_" + str(j)] = sku_temp["zl_" + str(j)] + sku_temp[k].map(
                    lambda x: x * ((k // 6 + 1) ** (-0.67)))
            # print(sku_temp["zl_"+str(j)].head(1))
            sku_temp["zl_" + str(j)] = sku_temp["zl_" + str(j)].map(lambda x: (x - np.min(sku_temp["zl_" + str(j)])) / (
                np.max(sku_temp["zl_" + str(j)]) - np.min(sku_temp["zl_" + str(j)])))
        sku_temp = sku_temp[columns]
        sku_temp.to_csv(dump_path, index=False)
        return sku_temp



# 商品的六种行为的频率
def get_action_p0509_feat_28(start_date, end_date):
    dump_path = './cache/user_feat7_2_six_%s_%s.csv' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = pd.read_csv(dump_path)
    else:
        df = get_actions(start_date, end_date)[['sku_id', 'type', 'time']]
        actions = df.groupby(['sku_id', 'type'], as_index=False).count()
        time_min = df.groupby(['sku_id', 'type'], as_index=False).min()
        time_max = df.groupby(['sku_id', 'type'], as_index=False).max()

        time_cha = pd.merge(time_max, time_min, on=['sku_id', 'type'], how='left')
        time_cha['time_x'] = time_cha['time_x'].map(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
        time_cha['time_y'] = time_cha['time_y'].map(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))

        time_cha['cha_hour'] = 1 + (time_cha['time_x'] - time_cha['time_y']).dt.days * 24 + (time_cha['time_x'] -
                                                                                             time_cha[
                                                                                                 'time_y']).dt.seconds // 3600
        del time_cha['time_x']
        del time_cha['time_y']
        # time_cha=time_cha.fillna(1)

        actions = pd.merge(time_cha, actions, on=['sku_id', 'type'], how="left")
        actions = actions.groupby(['sku_id', 'type']).sum()
        actions['cnt/time'] = actions['time'] / actions["cha_hour"]
        actions = actions.unstack()
        actions.columns = list(range(actions.shape[1]))
        actions = actions.reset_index()
        actions = actions.fillna(0)
        actions.to_csv(dump_path, index=False)
    actions.columns = ['sku_id'] + ['p_feat_28_' + str(i) for i in range(1, actions.shape[1])]
    return actions
print("US model  finish  part_2")


# In[ ]:




# In[4]:

#!/usr/bin/env python

#from basic_feat0518 import *

# 所有行为的总和
def get_action_feat(start_date, end_date):
    dump_path = './cache/action_%s_%s.csv' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = pd.read_csv(dump_path)
    else:
        actions = get_actions(start_date, end_date)
        actions = actions[['user_id', 'sku_id', 'type']]
        df = pd.get_dummies(actions['type'], prefix='action')
        actions = pd.concat([actions, df], axis=1)  # type: pd.DataFrame
        actions = actions.groupby(['user_id', 'sku_id'], as_index=False).sum()
        del actions['type']
        actions.to_csv(dump_path, index=False)
    return actions

# 行为按时间衰减
def get_accumulate_action_feat(start_date, end_date):
    dump_path = './cache/action_accumulate_%s_%s.csv' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = pd.read_csv(dump_path)
    else:
        actions = get_actions(start_date, end_date)
        df = pd.get_dummies(actions['type'], prefix='action')
        actions = pd.concat([actions, df], axis=1)  # type: pd.DataFrame
        actions.columns = ['user_id', 'sku_id', 'time', 'model_id', 'type',
                           'cate', 'brand', 'action_1', 'action_2', 'action_3',
                           'action_4', 'action_5', 'action_6']
        # 近期行为按时间衰减
        actions['weights'] = actions['time'].map(
            lambda x: datetime.strptime(end_date, '%Y-%m-%d') - datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
        # actions['weights'] = time.strptime(end_date, '%Y-%m-%d') - actions['datetime']
        actions['weights_1'] = actions['weights'].map(lambda x: 0.01938 * (x.days + 1) ** (-0.73563))
        actions['weights_2'] = actions['weights'].map(lambda x: 0.054465 * (x.days + 1) ** (-0.9035))
        actions['weights_3'] = actions['weights'].map(lambda x: 0.012186 * (x.days + 1) ** (-0.6953))
        actions['weights_4'] = actions['weights'].map(lambda x: 0.01357 * (x.days + 1) ** (-0.467889))
        actions['weights_5'] = actions['weights'].map(lambda x: 0.05234 * (x.days + 1) ** (-0.80797))
        actions['weights_6'] = actions['weights'].map(lambda x: 0.019337 * (x.days + 1) ** (-0.7341655))
        actions['action_1'] = actions['action_1'] * actions['weights_1']
        actions['action_2'] = actions['action_2'] * actions['weights_2']
        actions['action_3'] = actions['action_3'] * actions['weights_3']
        actions['action_4'] = actions['action_4'] * actions['weights_4']
        actions['action_5'] = actions['action_5'] * actions['weights_5']
        actions['action_6'] = actions['action_6'] * actions['weights_6']
        del actions['model_id']
        del actions['type']
        del actions['time']
        del actions['weights']
        del actions['weights_1']
        del actions['weights_2']
        del actions['weights_3']
        del actions['weights_4']
        del actions['weights_5']
        del actions['weights_6']
        actions = actions.groupby(['user_id', 'sku_id', 'cate', 'brand'], as_index=False).sum()
        actions.to_csv(dump_path, index=False)
    return actions

# U-B对浏览次数/用户总浏览次数
# U_B对行为1，2，4，5进行 浏览次数/用户总浏览次数（或者物品的浏览次数）
def get_action_U_P_feat1(start_date, end_date):
    dump_path = './cache/U_B_feat1_eight_%s_%s.csv' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = pd.read_csv(dump_path)
        actions.columns = ['user_id', 'sku_id'] + ['us_feat1_' + str(i) for i in range(1, actions.shape[1] - 1)]
        return actions
    else:
        temp = None
        df = get_actions(start_date, end_date)[['user_id', 'sku_id', 'type']]
        for i in (1, 2, 4, 5):
            actions = df[df['type'] == i]
            # del actions['type']
            action1 = actions.groupby(['user_id', 'sku_id'], as_index=False).count()
            action1.columns = ['user_id', 'sku_id', 'visit']
            action2 = actions.groupby('user_id', as_index=False).count()
            del action2['type']
            action2.columns = ['user_id', 'user_visits']

            action3 = actions.groupby('sku_id', as_index=False).count()
            del action3['type']
            action3.columns = ['sku_id', 'sku_visits']

            actions = pd.merge(action1, action2, how='left', on='user_id')
            actions = pd.merge(actions, action3, how='left', on='sku_id')
            actions['visit_rate_user'] = actions['visit'] / actions['user_visits']
            actions['visit_rate_sku'] = actions['visit'] / actions['sku_visits']
            del actions['visit']
            del actions['user_visits']
            del actions['sku_visits']
            if temp is None:
                temp = actions
            else:
                temp = pd.merge(temp, actions, how="outer", on=['user_id', 'sku_id'])
        temp.to_csv(dump_path, index=False)
        return temp


# 用户关注或加入购物车，但是不购买，且加入购物车或者关注小于10天
def get_action_U_P_feat2(start_date, end_date):
    dump_path = './cache/U_B_feat2_%s_%s.csv' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = pd.read_csv(dump_path)
    else:
        df = get_actions(start_date, end_date)
        # print df[df['type']==2].shape#296891
        # print df[df['type']==6].shape#16627100
        df1 = df[(df['type'] == 2) | (df['type'] == 6)][['user_id', 'sku_id', 'time']]
        df2 = df[df['type'] == 4][['user_id', 'sku_id']]
        df2['label'] = 0
        actions = pd.merge(df1, df2, on=['user_id', 'sku_id'], how='left')
        actions = actions.fillna(1)
        actions['time'] = (actions['time'].map(
            lambda x: datetime.strptime(end_date, '%Y-%m-%d') - datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))).dt.days
        #         actions[actions['time']>10].loc['label']=0
        actions.loc[actions['time'] > 10, 'label'] = 0
        del actions['time']
        actions.columns = ['user_id', 'sku_id', 'notbuy']
        actions.to_csv(dump_path, index=False)
    actions.columns = ['user_id', 'sku_id'] + ['us_feat2_' + str(i) for i in range(1, actions.shape[1] - 1)]
    return actions


def user_product_top_k_0_1(start_date, end_date):
    actions = get_actions(start_date, end_date)
    actions = actions[['user_id', 'sku_id', 'type']]
    df = pd.get_dummies(actions['type'], prefix='%s-%s-action' % (start_date, end_date))
    actions = pd.concat([actions, df], axis=1)  # type: pd.DataFrame
    actions = actions.groupby(['user_id', 'sku_id'], as_index=False).sum()
    del actions['type']
    user_sku = actions[['user_id', 'sku_id']]
    del actions['sku_id']
    del actions['user_id']
    actions = actions.applymap(lambda x: 1 if x > 0 else 0)
    actions = pd.concat([user_sku, actions], axis=1)
    return actions


# print user_product_top_k_0_1('2016-03-10','2016-04-11')
# 最近K天行为0/1提取
def get_action_U_P_feat3(k, start_date, end_date):
    dump_path = './cache/U_P_feat3_%s_%s_%s.csv' % (k, start_date, end_date)
    if os.path.exists(dump_path):
        actions = pd.read_csv(dump_path)
    else:
        start_days = datetime.strptime(end_date, '%Y-%m-%d') - timedelta(days=k)
        start_days = start_days.strftime('%Y-%m-%d')
        actions = user_product_top_k_0_1(start_days, end_date)
        actions.to_csv(dump_path, index=False)
    actions.columns = ['user_id', 'sku_id'] + ['us_feat3_' + str(k) + '_' + str(i) for i in
                                               range(1, actions.shape[1] - 1)]
    return actions


# 获取货物最近一次行为的时间距离当前时间的差距
def get_action_U_P_feat4(start_date, end_date):
    dump_path = './cache/U_P_feat4_%s_%s.csv' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = pd.read_csv(dump_path)
    else:
        df = get_actions(start_date, end_date)[['user_id', 'sku_id', 'time', 'type']]
        # df['time'] = df['time'].map(lambda x: (-1)*get_day_chaju(x,start_date))
        df = df.drop_duplicates(['user_id', 'sku_id', 'type'], keep='last')
        df['time'] = df['time'].map(lambda x: get_day_chaju(x, end_date) + 1)
        actions = df.groupby(['user_id', 'sku_id', 'type']).sum()
        actions = actions.unstack()
        actions.columns = list(range(actions.shape[1]))
        actions = actions.reset_index()
        actions = actions.fillna(30)
        actions.to_csv(dump_path, index=False)
    actions.columns = ['user_id', 'sku_id'] + ['us_feat4_' + str(i) for i in range(1, actions.shape[1] - 1)]
    return actions


# 获取最后一次行为的次数并且进行归一化
def get_action_U_P_feat5(start_date, end_date):
    dump_path = './cache/U_P_feat5_%s_%s.csv' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = pd.read_csv(dump_path)
    else:

        df = get_actions(start_date, end_date)[['user_id', 'sku_id', 'time', 'type']]
        df['time'] = df['time'].map(lambda x: get_day_chaju(x, end_date) + 1)

        idx = df.groupby(['user_id', 'sku_id', 'type'])['time'].transform(min)
        idx1 = idx == df['time']
        actions = df[idx1].groupby(["user_id", "sku_id", "type"]).count()
        actions = actions.unstack()
        actions.columns = list(range(actions.shape[1]))
        actions = actions.fillna(0)
        actions = actions.reset_index()

        user_sku = actions[['user_id', 'sku_id']]
        del actions['user_id']
        del actions['sku_id']
        min_max_scaler = preprocessing.MinMaxScaler()
        actions = min_max_scaler.fit_transform(actions.values)
        actions = pd.DataFrame(actions)
        actions = pd.concat([user_sku, actions], axis=1)

        actions.to_csv(dump_path, index=False)
    actions.columns = ['user_id', 'sku_id'] + ['us_feat5_' + str(i) for i in range(1, actions.shape[1] - 1)]
    return actions


# 获取人物和商品该层级最后一层的各种行为的统计数量
def get_action_U_P_feat6(start_date, end_date, n):
    dump_path = './cache/U_P_feat6_%s_%s_%s.csv' % (start_date, end_date, n)
    if os.path.exists(dump_path):
        actions = pd.read_csv(dump_path)
    else:
        df = get_actions(start_date, end_date)[['user_id', 'sku_id', 'time', 'type']]
        df['time'] = df['time'].map(lambda x: get_day_chaju(x, end_date) // n)
        df = df[df['time'] == 0]
        del df['time']
        temp = pd.get_dummies(df['type'], prefix='type')
        del df['type']
        actions = pd.concat([df, temp], axis=1)
        actions = actions.groupby(['user_id', 'sku_id'], as_index=False).sum()
        user_sku = actions[['user_id', 'sku_id']]
        del actions['user_id']
        del actions['sku_id']
        min_max_scaler = preprocessing.MinMaxScaler()
        actions = min_max_scaler.fit_transform(actions.values)
        actions = pd.DataFrame(actions)
        actions = pd.concat([user_sku, actions], axis=1)
        actions.to_csv(dump_path, index=False)
    actions.columns = ['user_id', 'sku_id'] + ['us_feat6_' + str(n) + '_' + str(i) for i in
                                               range(1, actions.shape[1] - 1)]
    return actions


# 品牌层级天数
def get_action_U_P_feat7(start_date, end_date):
    dump_path = './cache/U_P_feat7_%s_%s.csv' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = pd.read_csv(dump_path)
    else:
        actions = get_actions(start_date, end_date)[['user_id', 'sku_id', 'time', 'type']]
        actions['time'] = actions['time'].map(lambda x: x.split(' ')[0])
        actions = actions.drop_duplicates(['user_id', 'sku_id', 'time', 'type'], keep='first')
        actions['day'] = actions['time'].map(
            lambda x: (datetime.strptime(end_date, '%Y-%m-%d') - datetime.strptime(x, '%Y-%m-%d')).days)
        result = None
        columns = []
        for i in (2, 3, 7, 14, 28):  # 层级个数
            print ('i%s' % i)
            actions['level%s' % i] = actions['day'].map(lambda x: x // i)
            for j in (1, 2, 3, 4, 5, 6):  # type
                print ('j%s' % j)
                df = actions[actions['type'] == j][['user_id', 'sku_id', 'level%s' % i, 'time']]
                df = df.groupby(['user_id', 'sku_id', 'level%s' % i]).count()
                df = df.unstack()
                df = df.reset_index()
                df.columns = ['user_id', 'sku_id'] + list(range(df.shape[1] - 2))
                print (df.head())
                if result is None:
                    result = df
                else:
                    result = pd.merge(result, df, on=['user_id', 'sku_id'], how='left')
        user_sku = result[['user_id', 'sku_id']]
        del result['sku_id']
        del result['user_id']
        actions = result.fillna(0)
        min_max_scaler = preprocessing.MinMaxScaler()
        actions = min_max_scaler.fit_transform(actions.values)
        actions = pd.DataFrame(actions)
        actions.columns = range(actions.shape[1])
        actions = pd.concat([user_sku, actions], axis=1)
        actions.to_csv(dump_path, index=False)
    actions.columns = ['user_id', 'sku_id'] + ['us_feat7_' + str(i) for i in range(1, actions.shape[1] - 1)]
    return actions


# 用户和商品过去购六种行为频数
def get_action_U_P_feat8(start_date, end_date):
    dump_path = './cache/U_P_feat8_six_%s_%s.csv' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = pd.read_csv(dump_path)
    else:
        df = get_actions(start_date, end_date)[['user_id', 'sku_id', 'type', 'time']]
        actions = df.groupby(['user_id', 'sku_id', 'type'], as_index=False).count()
        time_min = df.groupby(['user_id', 'sku_id', 'type'], as_index=False).min()
        time_max = df.groupby(['user_id', 'sku_id', 'type'], as_index=False).max()

        time_cha = pd.merge(time_max, time_min, on=['user_id', 'sku_id', 'type'], how='left')
        time_cha['time_x'] = time_cha['time_x'].map(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
        time_cha['time_y'] = time_cha['time_y'].map(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))

        time_cha['cha_hour'] = 1 + (time_cha['time_x'] - time_cha['time_y']).dt.days * 24 + (time_cha['time_x'] -
                                                                                             time_cha[
                                                                                                 'time_y']).dt.seconds // 3600
        del time_cha['time_x']
        del time_cha['time_y']
        # time_cha=time_cha.fillna(1)

        actions = pd.merge(time_cha, actions, on=['user_id', 'sku_id', 'type'], how="left")
        actions = actions.groupby(['user_id', 'sku_id', 'type']).sum()
        actions['cnt/time'] = actions['time'] / actions["cha_hour"]
        actions = actions.unstack()
        actions.columns = list(range(actions.shape[1]))
        actions = actions.reset_index()
        actions = actions.fillna(0)
        actions.to_csv(dump_path, index=False)
    actions.columns = ['user_id', 'sku_id'] + ['us_feat8_' + str(i) for i in range(1, actions.shape[1] - 1)]
    return actions


# #层级天数  ,一共有几天产生了购买行为
def get_action_U_P_feat9(start_date, end_date, n):
    dump_path = './cache/U_P_feat9_%s_%s_%s.csv' % (start_date, end_date, n)
    if os.path.exists(dump_path):
        actions = pd.read_csv(dump_path)
    else:
        df = get_actions(start_date, end_date)[['user_id', 'sku_id', 'time', 'type']]
        df['time'] = df['time'].map(lambda x: get_day_chaju(x, end_date) // n)
        df = df.drop_duplicates(['user_id', 'sku_id', 'type', 'time'], keep='first')

        actions = df.groupby(['user_id', 'sku_id', 'type']).count()
        actions = actions.unstack()
        actions.columns = list(range(actions.shape[1]))
        actions = actions.fillna(0)
        actions = actions.reset_index()
        user_sku = actions[['user_id', 'sku_id']]
        del actions['user_id']
        del actions['sku_id']
        min_max_scaler = preprocessing.MinMaxScaler()
        actions = min_max_scaler.fit_transform(actions.values)
        actions = pd.DataFrame(actions)
        actions = pd.concat([user_sku, actions], axis=1)
        actions.to_csv(dump_path, index=False)
    actions.columns = ['user_id', 'sku_id'] + ['us_feat9_' + str(n) + '_' + str(i) for i in
                                               range(1, actions.shape[1] - 1)]
    return actions


def get_action_U_P_feat14(start_date, end_date):
    dump_path = './cache/U_P_feat14_%s_%s.csv' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = pd.read_csv(dump_path)
    else:
        n = 5
        df = get_actions(start_date, end_date)[['user_id', 'sku_id', 'time', 'type']]
        df = df[df['type'] == 4][['user_id', 'sku_id', 'time']]
        df['time'] = df['time'].map(lambda x: get_day_chaju(x, end_date) // n)
        days = np.max(df['time'])

        df['cnt'] = 0
        actions = df.groupby(['user_id', 'sku_id', 'time']).count()

        actions = actions.unstack()

        actions.columns = list(range(actions.shape[1]))
        actions = actions.reset_index()

        actions = actions.fillna(0)
        user_sku = actions[['user_id', 'sku_id']]
        del actions['user_id']
        del actions['sku_id']
        min_max_scaler = preprocessing.MinMaxScaler()
        actions = min_max_scaler.fit_transform(actions.values)
        actions = pd.DataFrame(actions)
        actions = pd.concat([user_sku, actions], axis=1)
        actions.to_csv(dump_path, index=False)
    actions.columns = ['user_id', 'sku_id'] + ['us_feat14_' + str(i) for i in range(1, actions.shape[1] - 1)]
    return actions


# 用户和品牌交叉
def get_action_U_P_feat16(start_date, end_date):
    dump_path = './cache/U_P_feat16_%s_%s.csv' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = pd.read_csv(dump_path)
    else:
        actions = get_actions(start_date, end_date)[['user_id', 'sku_id']]
        actions['cnt'] = 0
        action1 = actions.groupby(['user_id', 'sku_id'], as_index=False).count()
        action2 = actions.groupby('user_id', as_index=False).count()
        del action2['sku_id']
        action2.columns = ['user_id', 'user_cnt']

        action3 = actions.groupby('sku_id', as_index=False).count()
        del action3['user_id']
        action3.columns = ['sku_id', 'sku_cnt']

        actions = pd.merge(action1, action2, how='left', on='user_id')
        actions = pd.merge(actions, action3, how='left', on='sku_id')

        actions['user_cnt'] = actions['cnt'] / actions['user_cnt']
        actions['sku_cnt'] = actions['cnt'] / actions['sku_cnt']
        del actions['cnt']
        actions.to_csv(dump_path, index=False)
    actions.columns = ['user_id', 'sku_id'] + ['us_feat16_' + str(i) for i in range(1, actions.shape[1] - 1)]
    return actions



# 点击模块
def get_action_U_P_feat_0509_feat_24(start_date, end_date, n):
    dump_path = './cache/get_action_U_P_feat_0509_feat_24_%s_%s_%s.csv' % (start_date, end_date, n)
    if os.path.exists(dump_path):
        actions = pd.read_csv(dump_path)
    else:
        start_days = datetime.strptime(end_date, '%Y-%m-%d') - timedelta(days=n)
        start_days = datetime.strftime(start_days, '%Y-%m-%d')
        actions = get_actions(start_days, end_date)
        actions = actions[actions['type'] == 6][['user_id','sku_id','model_id']]

        actions_click_sum = actions[['user_id','sku_id', 'model_id']].groupby(['user_id','sku_id']).count().reset_index()
        actions_click_sum.columns = ['user_id','sku_id', str(n) + 'click_sum_all']
        actions[str(n) + 'p_click14_history'] = actions['model_id'].map(lambda x: int(x == 14))
        actions[str(n) + 'p_click21_history'] = actions['model_id'].map(lambda x: int(x == 21))
        actions[str(n) + 'p_click28_history'] = actions['model_id'].map(lambda x: int(x == 28))
        actions[str(n) + 'p_click110_history'] = actions['model_id'].map(lambda x: int(x == 110))
        actions[str(n) + 'p_click210_history'] = actions['model_id'].map(lambda x: int(x == 210))
        actions = actions.groupby(['user_id','sku_id']).sum().reset_index().drop('model_id', axis=1)

        actions = pd.merge(actions, actions_click_sum, how='left', on=['user_id','sku_id'])

        actions[str(n) + 'p_click14/click_sum_history'] = actions[str(n) + 'p_click14_history'] / actions[
            str(n) + 'click_sum_all']
        actions[str(n) + 'p_click21/click_sum_history'] = actions[str(n) + 'p_click21_history'] / actions[
            str(n) + 'click_sum_all']
        actions[str(n) + 'p_click28/click_sum_history'] = actions[str(n) + 'p_click28_history'] / actions[
            str(n) + 'click_sum_all']
        actions[str(n) + 'p_click110/click_sum_history'] = actions[str(n) + 'p_click110_history'] / actions[
            str(n) + 'click_sum_all']
        actions[str(n) + 'p_click210/click_sum_history'] = actions[str(n) + 'p_click210_history'] / actions[
            str(n) + 'click_sum_all']

        sku_id = actions[['user_id','sku_id']]
        del actions['sku_id']
        actions = actions.fillna(0)
        columns = actions.columns
        min_max_scale = preprocessing.MinMaxScaler()
        actions = min_max_scale.fit_transform(actions.values)
        actions = pd.concat([sku_id, pd.DataFrame(actions, columns=columns)], axis=1)
        actions.to_csv(dump_path, index=False)
    actions.columns = ['user_id','sku_id'] + ['p0509_feat_24_' + str(n) + '_' + str(i) for i in range(1, actions.shape[1]-1)]
    return actions












# def sku_tongji_info():
#     dump_1 = './cache/a_tongji/brandsales.csv'
#     dump_2 = './cache/a_tongji/productsales.csv'
#     actions_1 = pd.read_csv(dump_1)[['sales', 'brand', 'cate']]
#     actions_2 = pd.read_csv(dump_2)
#     action = pd.merge([actions_2, actions_1], on=['brand', 'cate'], how='left')
#     return action
print("US model  finish  part_3")


# In[ ]:




# In[5]:

# from basic_feat0518 import *
# from u_feat0518 import *
# from s_feat0518 import *
# from us_feat0518 import *

# 标签
def get_labels(start_date, end_date):
    dump_path = './cache/labels_%s_%s_cate==8.csv' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = pd.read_csv(dump_path)
    else:
        actions = get_actions(start_date, end_date)
        actions = actions[(actions['type'] == 4)&(actions['cate']==8)]
        actions =actions[['user_id','sku_id']].drop_duplicates(['user_id','sku_id']).reset_index()
        actions['label'] = 1
        actions = actions[['user_id', 'sku_id', 'label']]
#         actions.to_csv(dump_path, index=False)
    return actions

# 训练集
def make_train_set(train_start_date, train_end_date, test_start_date, test_end_date):
    dump_path = './cache1/buUStrain_set_%s_%s_%s_%s.csv' % (train_start_date, train_end_date, test_start_date, test_end_date)
    if os.path.exists(dump_path):
        actions = pd.read_csv(dump_path)
    else:
        print ('================>>>train feature starting')
        start_days = datetime.strptime(train_end_date, '%Y-%m-%d') - timedelta(days=10)
        start_days = str(start_days.strftime('%Y-%m-%d'))
        actions = get_actions(start_days, train_end_date)
        actions = actions[actions['cate'] == 8][['user_id', 'sku_id']].drop_duplicates(['user_id', 'sku_id'])
        print (actions.shape)
        actions = pd.merge(actions, get_basic_user_feat(), on='user_id', how='left')
        print (actions.shape)
        start_days = "2016-02-01"
        print ('================>>>merge user feature')
#       for i in (1,2,3,5,7,10,15,21,30):
        for i in(1,2,3,7,14,28):
#             start_day1=datetime.strptime(train_end_date, '%Y-%m-%d') - timedelta(days=i)
#             start_day1 = start_day1.strftime('%Y-%m-%d')
            print('=======>>>>train user_feat')
            actions=pd.merge(actions,get_user_feat(train_start_date, train_end_date, i),on='user_id',how='left')
            print('=======>>>>train user_feat11')
            actions = pd.merge(actions, get_action_user_feat11(train_start_date, train_end_date, i), on='user_id', how='left')
            print('=======>>>>train user_feat13')
            actions = pd.merge(actions, get_action_user_feat13(train_start_date, train_end_date, i), on='user_id', how='left')
            print('=======>>>>train u0509_feat_18')
            actions = pd.merge(actions, get_action_u0509_feat_18(train_start_date, train_end_date, i), on='user_id', how='left')
            print('=======>>>>train u0509_feat_21')
            actions = pd.merge(actions, get_action_u0509_feat_21(train_start_date, train_end_date, i), on='user_id', how='left')
            print('=======>>>>train u0509_feat_22')
            actions = pd.merge(actions, get_action_u0509_feat_22(train_start_date, train_end_date, i), on='user_id',how='left')
            print('=======>>>>train u0509_feat_23')
            actions = pd.merge(actions, get_action_u0509_feat_23(train_start_date, train_end_date, i), on='user_id',how='left')
            print('=======>>>>train u0509_feat_24')
            if(i<=10):
                actions = pd.merge(actions, get_action_u0509_feat_24(train_start_date, train_end_date, i), on='user_id',how='left')
                print('=======>>>>train u0509_feat_27')
            actions = pd.merge(actions, get_action_u0509_feat_27(train_start_date, train_end_date, i), on='user_id', how='left')
        print (actions.shape)
        print('=======>>>>train user_feat1')
        actions = pd.merge(actions, get_action_user_feat1(start_days, train_end_date), on='user_id', how='left')
        print (actions.shape)
        print('=======>>>>train user_feat2')
        actions = pd.merge(actions, get_action_user_feat2(start_days, train_end_date), on='user_id', how='left')
        print (actions.shape)
        print('=======>>>>train user_feat6')
        actions = pd.merge(actions, get_action_user_feat6(start_days, train_end_date), on='user_id', how='left')
        print (actions.shape)
        print('=======>>>>train user_feat6_six')
        actions = pd.merge(actions, get_action_user_feat6_six(start_days, train_end_date), on='user_id', how='left')
        print (actions.shape)
        print('=======>>>>train user_feat7')
        actions = pd.merge(actions, get_action_user_feat7(start_days, train_end_date), on='user_id', how='left')
        print (actions.shape)
        print('=======>>>>train user_feat8')
        actions = pd.merge(actions, get_action_user_feat8(start_days, train_end_date), on='user_id', how='left')
        print (actions.shape)
        print('=======>>>>train user_feat8_2')
        actions = pd.merge(actions, get_action_user_feat8_2(start_days, train_end_date), on='user_id', how='left')
        print (actions.shape)
        print('=======>>>>train user_feat9')
        actions = pd.merge(actions, get_action_user_feat9(start_days, train_end_date), on='user_id', how='left')
        print (actions.shape)
        print('=======>>>>train user_feat10')
        actions = pd.merge(actions, get_action_user_feat10(start_days, train_end_date), on='user_id', how='left')
        print (actions.shape)
        print('=======>>>>train user_feat12')
#         actions = pd.merge(actions, get_action_user_feat12(start_days, train_end_date), on='user_id', how='left')
#         print (actions.shape)
        print('=======>>>>train user_feat14')
        actions = pd.merge(actions, get_action_user_feat14(train_start_date, train_end_date), on='user_id', how='left')
        print (actions.shape)
        print('=======>>>>train user_feat15')
        actions = pd.merge(actions, get_action_user_feat15(start_days, train_end_date), on='user_id', how='left')
        print (actions.shape)
        print('=======>>>>train user_feat16')
        actions = pd.merge(actions, get_action_user_feat16(start_days, train_end_date), on='user_id', how='left')
        print (actions.shape)
        print('=======>>>>train u0509_feat_19')
        actions = pd.merge(actions, get_action_u0509_feat_19(start_days, train_end_date), on='user_id', how='left')
        print (actions.shape)
        print('=======>>>>train u0509_feat_20')
        actions = pd.merge(actions, get_action_u0509_feat_20(start_days, train_end_date), on='user_id', how='left')
        print (actions.shape)
        print('=======>>>>train u0509_feat_25')
        actions = pd.merge(actions, get_action_u0509_feat_25(start_days, train_end_date), on='user_id', how='left')
        print (actions.shape)


        print ('================>>>merge product feature')
        actions = pd.merge(actions, get_basic_product_feat(), on='sku_id', how='left')
        print (actions.shape)
        actions = pd.merge(actions, get_comments_product_feat(train_start_date, train_end_date), on='sku_id', how='left')
        print (actions.shape)
#        for i in (1, 2, 3, 5, 7, 10, 15, 21, 30):

        

        for i in(1,2,3,7,14,28):
            
            print('=======>>>>train p0509_feat')
            actions = pd.merge(actions, get_action_p0509_feat(train_start_date, train_end_date, i), on='sku_id', how='left')
            print('=======>>>>train product_feat_11')
            actions = pd.merge(actions, get_action_product_feat_11(train_start_date, train_end_date, i), on='sku_id', how='left')
            print('=======>>>>train product_feat_13')
            actions = pd.merge(actions, get_action_product_feat_13(train_start_date, train_end_date, i), on='sku_id', how='left')
            print('=======>>>>train p0509_feat_21')
            actions = pd.merge(actions, get_action_p0509_feat_21(train_start_date, train_end_date, i), on='sku_id', how='left')
            print('=======>>>>train p0509_feat_23')
            actions = pd.merge(actions, get_action_p0509_feat_23(train_start_date, train_end_date, i), on='sku_id', how='left')
            print('=======>>>>train p0509_feat24')
            if(i<=10):
                actions = pd.merge(actions, get_action_p0509_feat_24(train_start_date, train_end_date, i), on='sku_id', how='left')
                print('=======>>>>train p0509_feat27')
            actions = pd.merge(actions, get_action_p0509_feat_27(train_start_date, train_end_date, i), on='sku_id', how='left')
        print (actions.shape)
        print('=======>>>>train product_feat_1')
        actions = pd.merge(actions, get_action_product_feat_1(start_days, train_end_date), on='sku_id', how='left')
        print (actions.shape)
        print('=======>>>>train p0509_feat2')
        actions = pd.merge(actions, get_action_p0509_feat_2(start_days, train_end_date), on='sku_id', how='left')
        print (actions.shape)
        print('=======>>>>train p0509_feat6')
        actions = pd.merge(actions, get_action_p0509_feat_6(start_days, train_end_date), on='sku_id', how='left')
        print (actions.shape)
        print('=======>>>>train p0509_feat_6_six')
        actions = pd.merge(actions, get_action_p0509_feat_6_six(start_days, train_end_date), on='sku_id', how='left')
        print (actions.shape)
        print('=======>>>>train p0509_feat8')
        actions = pd.merge(actions, get_action_p0509_feat_8(start_days, train_end_date), on='sku_id', how='left')
        print (actions.shape)
        print('=======>>>>train p0509_feat8_2')
        actions = pd.merge(actions, get_action_p0509_feat_8_2(start_days, train_end_date), on='sku_id', how='left')
        print (actions.shape)
        print('=======>>>>train p0509_feat9')
        actions = pd.merge(actions, get_action_p0509_feat_9(start_days, train_end_date), on='sku_id', how='left')
        print (actions.shape)
        print('=======>>>>train p0509_feat10')
        actions = pd.merge(actions, get_action_product_feat_10(start_days, train_end_date), on='sku_id', how='left')
        print (actions.shape)
#         print('=======>>>>train p0509_feat12')
#         actions = pd.merge(actions, get_action_product_feat_12(start_days, train_end_date), on='sku_id', how='left')
#         print (actions.shape)
        print('=======>>>>train product_feat_14')
        actions = pd.merge(actions, get_action_product_feat_14(train_start_date, train_end_date), on='sku_id', how='left')
        print (actions.shape)
        print('=======>>>>train p0509_feat15')
        actions = pd.merge(actions, get_action_p0509_feat_15(start_days, train_end_date), on='sku_id', how='left')
        print (actions.shape)
        print('=======>>>>train product_feat_16')
        actions = pd.merge(actions, get_action_product_feat_16(start_days, train_end_date), on='sku_id', how='left')
        print (actions.shape)
        print('=======>>>>train p0509_feat17')
        actions = pd.merge(actions, get_action_p0509_feat_17(start_days, train_end_date), on='sku_id', how='left')
        print (actions.shape)
        print('=======>>>>train p0509_feat19')
        actions = pd.merge(actions, get_action_p0509_feat_19(start_days, train_end_date), on='sku_id', how='left')
        print (actions.shape)
        print('=======>>>>train p0509_feat23')

        print('=======>>>>train p0509_feat28')
        actions = pd.merge(actions, get_action_p0509_feat_28(start_days, train_end_date), on='sku_id', how='left')
        print (actions.shape)

        print ('================>>>merge user_sku feature')
        print ('train get_accumulate_action_feat')
        actions = pd.merge(actions, get_accumulate_action_feat(train_start_date, train_end_date), how='left', on=['user_id', 'sku_id'])
        print ('train U_P_feat1')
        
        actions = pd.merge(actions, get_action_U_P_feat1(start_days, train_end_date), how="left",on=['user_id', 'sku_id'])
        print(actions.shape)
        print ('train U_P_feat3')
        for i in (1, 2, 3, 5, 7, 10, 15, 21, 30):
            start_days_2 = datetime.strptime(train_end_date, '%Y-%m-%d') - timedelta(days=i)
            start_days_2 = start_days_2.strftime('%Y-%m-%d')
            actions = pd.merge(actions, get_action_feat(start_days_2, train_end_date), how='left',on=['user_id', 'sku_id'])
        actions.columns=['user_id','sku_id']+['action_feat'+str(i) for i in range(1,actions.shape[1]-1)]
    
        for i in(1,2,3,7,14,28):
            actions = pd.merge(actions, get_action_U_P_feat3(i, start_days, train_end_date), on=['user_id', 'sku_id'],
                               how='left')
            actions=pd.merge(actions, get_action_U_P_feat6( train_start_date, train_end_date,i), on=['user_id', 'sku_id'],
                               how='left')
            actions=pd.merge(actions, get_action_U_P_feat9( train_start_date, train_end_date,i), on=['user_id', 'sku_id'],
                               how='left')
            if(i<=10):
                actions=pd.merge(actions, get_action_U_P_feat_0509_feat_24( train_start_date, train_end_date,i), on=['user_id', 'sku_id'],
                               how='left')
 
        print ('train U_P_feat4')
        actions = pd.merge(actions, get_action_U_P_feat4(start_days, train_end_date), how="left",
                           on=['user_id', 'sku_id'])
        print(actions.shape)
        print ('train U_P_feat5')
        actions = pd.merge(actions, get_action_U_P_feat5(start_days, train_end_date), how="left",
                           on=['user_id', 'sku_id'])
        print(actions.shape)
#         print ('train U_P_feat7')
#         actions = pd.merge(actions, get_action_U_P_feat7(train_start_date, train_end_date), how="left",
#                                     on=['user_id', 'sku_id'])
#         print(actions.shape)
        print ('train U_P_feat8')
        actions = pd.merge(actions, get_action_U_P_feat8(start_days, train_end_date), how="left",
                           on=['user_id', 'sku_id'])
        print(actions.shape)
        print ('train U_P_feat14')
        actions = pd.merge(actions, get_action_U_P_feat14(train_start_date, train_end_date), how="left",
                           on=['user_id', 'sku_id'])
        print ('train U_P_feat16')
        actions = pd.merge(actions, get_action_U_P_feat16(train_start_date, train_end_date), how="left",
                           on=['user_id', 'sku_id'])

        print(actions.shape)
        print("train get_labels")
        actions=pd.merge(actions,get_labels(test_start_date, test_end_date),how='left',on=['user_id','sku_id'])
        print(actions.shape)
        actions = actions.fillna(0)
        
#         actions.to_csv(dump_path,index=False)
    return actions


# 测试集
def make_test_set(train_start_date, train_end_date):
    dump_path = './cache1/buUStest_set_%s_%s.csv' % (train_start_date, train_end_date)
    if os.path.exists(dump_path):
        actions = pd.read_csv(dump_path)
    else:
        print ('================>>>train feature starting')
        start_days = datetime.strptime(train_end_date, '%Y-%m-%d') - timedelta(days=10)
        start_days = str(start_days.strftime('%Y-%m-%d'))
        actions = get_actions(start_days, train_end_date)
        actions = actions[actions['cate'] == 8][['user_id', 'sku_id']].drop_duplicates(['user_id', 'sku_id'])
        print (actions.shape)
        actions = pd.merge(actions, get_basic_user_feat(), on='user_id', how='left')
        print (actions.shape)
        start_days = "2016-02-01"
        print ('================>>>merge user feature')
    # for i in (1, 2, 3, 5, 7, 10, 15, 21, 30):
#        for i in (1, 2, 3, 7, 14, 28):
        for i in(1,2,3,7,14,28):
            start_day1 = datetime.strptime(train_end_date, '%Y-%m-%d') - timedelta(days=i)
            start_day1 = start_day1.strftime('%Y-%m-%d')
            print('=======>>>>train user_feat')
            actions = pd.merge(actions, get_user_feat(train_start_date, train_end_date, i), on='user_id', how='left')
            print('=======>>>>train user_feat11')
            actions = pd.merge(actions, get_action_user_feat11(train_start_date, train_end_date, i), on='user_id', how='left')
            print('=======>>>>train user_feat13')
            actions = pd.merge(actions, get_action_user_feat13(train_start_date, train_end_date, i), on='user_id', how='left')
            print('=======>>>>train u0509_feat_18')
            actions = pd.merge(actions, get_action_u0509_feat_18(train_start_date, train_end_date, i), on='user_id',
                               how='left')
            print('=======>>>>train u0509_feat_21')
            actions = pd.merge(actions, get_action_u0509_feat_21(train_start_date, train_end_date, i), on='user_id',
                               how='left')
            print('=======>>>>train u0509_feat_22')
            actions = pd.merge(actions, get_action_u0509_feat_22(train_start_date, train_end_date, i), on='user_id',
                               how='left')
            print('=======>>>>train u0509_feat_23')
            actions = pd.merge(actions, get_action_u0509_feat_23(train_start_date, train_end_date, i), on='user_id',
                               how='left')
            print('=======>>>>train u0509_feat_24')
            if(i<=10):
                actions = pd.merge(actions, get_action_u0509_feat_24(train_start_date, train_end_date, i), on='user_id',
                                   how='left')
                print('=======>>>>train u0509_feat_27')
            actions = pd.merge(actions, get_action_u0509_feat_27(train_start_date, train_end_date, i), on='user_id',
                               how='left')
        print (actions.shape)
        print('=======>>>>train user_feat1')
        actions = pd.merge(actions, get_action_user_feat1(start_days, train_end_date), on='user_id', how='left')
        print (actions.shape)
        print('=======>>>>train user_feat2')
        actions = pd.merge(actions, get_action_user_feat2(start_days, train_end_date), on='user_id', how='left')
        print (actions.shape)
        print('=======>>>>train user_feat6')
        actions = pd.merge(actions, get_action_user_feat6(start_days, train_end_date), on='user_id', how='left')
        print (actions.shape)
        print('=======>>>>train user_feat6_six')
        actions = pd.merge(actions, get_action_user_feat6_six(start_days, train_end_date), on='user_id', how='left')
        print (actions.shape)
        print('=======>>>>train user_feat7')
        actions = pd.merge(actions, get_action_user_feat7(start_days, train_end_date), on='user_id', how='left')
        print (actions.shape)
        print('=======>>>>train user_feat8')
        actions = pd.merge(actions, get_action_user_feat8(start_days, train_end_date), on='user_id', how='left')
        print (actions.shape)
        print('=======>>>>train user_feat8_2')
        actions = pd.merge(actions, get_action_user_feat8_2(start_days, train_end_date), on='user_id', how='left')
        print (actions.shape)
        print('=======>>>>train user_feat9')
        actions = pd.merge(actions, get_action_user_feat9(start_days, train_end_date), on='user_id', how='left')
        print (actions.shape)
        print('=======>>>>train user_feat10')
        actions = pd.merge(actions, get_action_user_feat10(start_days, train_end_date), on='user_id', how='left')
        print (actions.shape)
        print('=======>>>>train user_feat12')
#         actions = pd.merge(actions, get_action_user_feat12(start_days, train_end_date), on='user_id', how='left')
#         print (actions.shape)
        print('=======>>>>train user_feat14')
        actions = pd.merge(actions, get_action_user_feat14(train_start_date, train_end_date), on='user_id', how='left')
        print (actions.shape)
        print('=======>>>>train user_feat15')
        actions = pd.merge(actions, get_action_user_feat15(start_days, train_end_date), on='user_id', how='left')
        print (actions.shape)
        print('=======>>>>train user_feat16')
        actions = pd.merge(actions, get_action_user_feat16(start_days, train_end_date), on='user_id', how='left')
        print (actions.shape)
        print('=======>>>>train u0509_feat_19')
        actions = pd.merge(actions, get_action_u0509_feat_19(start_days, train_end_date), on='user_id', how='left')
        print (actions.shape)
        print('=======>>>>train u0509_feat_20')
        actions = pd.merge(actions, get_action_u0509_feat_20(start_days, train_end_date), on='user_id', how='left')
        print (actions.shape)
        print('=======>>>>train u0509_feat_25')
        actions = pd.merge(actions, get_action_u0509_feat_25(start_days, train_end_date), on='user_id', how='left')
        print (actions.shape)

        print ('================>>>merge product feature')
        actions = pd.merge(actions, get_basic_product_feat(), on='sku_id', how='left')
        print (actions.shape)
        actions = pd.merge(actions, get_comments_product_feat(train_start_date, train_end_date), on='sku_id',
                           how='left')
        print (actions.shape)
#         for i in (1, 2, 3, 5, 7, 10, 15, 21, 30):
        for i in(1,2,3,7,14,28):
        
            start_day1 = datetime.strptime(train_end_date, '%Y-%m-%d') - timedelta(days=i)
            start_day1 = start_day1.strftime('%Y-%m-%d')
            print('=======>>>>train p0509_feat')
            actions = pd.merge(actions, get_action_p0509_feat(train_start_date, train_end_date, i), on='sku_id', how='left')
            print('=======>>>>train product_feat_11')
            actions = pd.merge(actions, get_action_product_feat_11(train_start_date, train_end_date, i), on='sku_id',
                               how='left')
            print('=======>>>>train product_feat_13')
            actions = pd.merge(actions, get_action_product_feat_13(train_start_date, train_end_date, i), on='sku_id',
                               how='left')
            print('=======>>>>train p0509_feat_21')
            actions = pd.merge(actions, get_action_p0509_feat_21(train_start_date, train_end_date, i), on='sku_id',
                               how='left')
            print('=======>>>>train p0509_feat_23')
            actions = pd.merge(actions, get_action_p0509_feat_23(train_start_date, train_end_date, i), on='sku_id',
                               how='left')
            print('=======>>>>train p0509_feat24')
            if(i<=10):
                actions = pd.merge(actions, get_action_p0509_feat_24(train_start_date, train_end_date, i), on='sku_id',
                               how='left')
            print('=======>>>>train p0509_feat27')
            actions = pd.merge(actions, get_action_p0509_feat_27(train_start_date, train_end_date,i), on='sku_id',
                               how='left')
        print (actions.shape)
        print('=======>>>>train product_feat_1')
        actions = pd.merge(actions, get_action_product_feat_1(start_days, train_end_date), on='sku_id', how='left')
        print (actions.shape)
        print('=======>>>>train p0509_feat2')
        actions = pd.merge(actions, get_action_p0509_feat_2(start_days, train_end_date), on='sku_id', how='left')
        print (actions.shape)
        print('=======>>>>train p0509_feat6')
        actions = pd.merge(actions, get_action_p0509_feat_6(start_days, train_end_date), on='sku_id', how='left')
        print (actions.shape)
        print('=======>>>>train p0509_feat_6_six')
        actions = pd.merge(actions, get_action_p0509_feat_6_six(start_days, train_end_date), on='sku_id', how='left')
        print (actions.shape)
        print('=======>>>>train p0509_feat8')
        actions = pd.merge(actions, get_action_p0509_feat_8(start_days, train_end_date), on='sku_id', how='left')
        print (actions.shape)
        print('=======>>>>train p0509_feat8_2')
        actions = pd.merge(actions, get_action_p0509_feat_8_2(start_days, train_end_date), on='sku_id', how='left')
        print (actions.shape)
        print('=======>>>>train p0509_feat9')
        actions = pd.merge(actions, get_action_p0509_feat_9(start_days, train_end_date), on='sku_id', how='left')
        print (actions.shape)
        print('=======>>>>train p0509_feat10')
        actions = pd.merge(actions, get_action_product_feat_10(start_days, train_end_date), on='sku_id', how='left')
        print (actions.shape)
        print('=======>>>>train p0509_feat12')
#         actions = pd.merge(actions, get_action_product_feat_12(start_days, train_end_date), on='sku_id', how='left')
#         print (actions.shape)
        print('=======>>>>train product_feat_14')
        actions = pd.merge(actions, get_action_product_feat_14(train_start_date, train_end_date), on='sku_id',
                           how='left')
        print (actions.shape)
        print('=======>>>>train p0509_feat15')
        actions = pd.merge(actions, get_action_p0509_feat_15(start_days, train_end_date), on='sku_id', how='left')
        print (actions.shape)
        print('=======>>>>train product_feat_16')
        actions = pd.merge(actions, get_action_product_feat_16(start_days, train_end_date), on='sku_id', how='left')
        print (actions.shape)
        print('=======>>>>train p0509_feat17')
        actions = pd.merge(actions, get_action_p0509_feat_17(start_days, train_end_date), on='sku_id', how='left')
        print (actions.shape)
        print('=======>>>>train p0509_feat19')
        actions = pd.merge(actions, get_action_p0509_feat_19(start_days, train_end_date), on='sku_id', how='left')
        print (actions.shape)
        print('=======>>>>train p0509_feat23')

        print('=======>>>>train p0509_feat28')
        actions = pd.merge(actions, get_action_p0509_feat_28(start_days, train_end_date), on='sku_id', how='left')
        print (actions.shape)

        print ('================>>>merge user_sku feature')
        print ('train get_accumulate_action_feat')
        
        actions = pd.merge(actions, get_accumulate_action_feat(train_start_date, train_end_date), how='left',
                           on=['user_id', 'sku_id'])
        print ('train U_P_feat1')
        actions = pd.merge(actions, get_action_U_P_feat1(start_days, train_end_date), how="left",
                           on=['user_id', 'sku_id'])
        print(actions.shape)
        print ('train U_P_feat3')
        for i in (1, 2, 3, 5, 7, 10, 15, 21, 30):
            start_days_2 = datetime.strptime(train_end_date, '%Y-%m-%d') - timedelta(days=i)
            start_days_2 = start_days_2.strftime('%Y-%m-%d')
            actions = pd.merge(actions, get_action_feat(start_days_2, train_end_date), how='left',on=['user_id', 'sku_id'])
        actions.columns=['user_id','sku_id']+['action_feat'+str(i) for i in range(1,actions.shape[1]-1)]
    
        for i in(1,2,3,7,14,28):
            actions = pd.merge(actions, get_action_U_P_feat3(i, start_days, train_end_date), on=['user_id', 'sku_id'],
                               how='left')
            actions=pd.merge(actions, get_action_U_P_feat6( train_start_date, train_end_date,i), on=['user_id', 'sku_id'],
                               how='left')
            actions=pd.merge(actions, get_action_U_P_feat9( train_start_date, train_end_date,i), on=['user_id', 'sku_id'],
                               how='left')
            if(i<=10):
                actions=pd.merge(actions, get_action_U_P_feat_0509_feat_24( train_start_date, train_end_date,i), on=['user_id', 'sku_id'],
                               how='left')
            
        print ('train U_P_feat4')
        actions = pd.merge(actions, get_action_U_P_feat4(start_days, train_end_date), how="left",
                           on=['user_id', 'sku_id'])
        print(actions.shape)
        print ('train U_P_feat5')
        actions = pd.merge(actions, get_action_U_P_feat5(start_days, train_end_date), how="left",
                           on=['user_id', 'sku_id'])
        print(actions.shape)
        print ('train U_P_feat7')
#         actions = pd.merge(actions, get_action_U_P_feat7(train_start_date, train_end_date), how="left",
#                            on=['user_id', 'sku_id'])
        print(actions.shape)
        print ('train U_P_feat8')
        actions = pd.merge(actions, get_action_U_P_feat8(start_days, train_end_date), how="left",
                           on=['user_id', 'sku_id'])
        print(actions.shape)
        print ('train U_P_feat14')
        actions = pd.merge(actions, get_action_U_P_feat14(train_start_date, train_end_date), how="left",
                           on=['user_id', 'sku_id'])
        print ('train U_P_feat16')
        actions = pd.merge(actions, get_action_U_P_feat16(train_start_date, train_end_date), how="left",
                           on=['user_id', 'sku_id'])

        print(actions.shape)
        actions= actions.fillna(0)
#         actions.to_csv(dump_path, index=False)
    return actions


train_start_date = '2016-03-10'
train_end_date = '2016-04-11'
test_start_date = '2016-04-11'
test_end_date = '2016-04-16'

sub_start_date = '2016-03-15'
sub_end_date = '2016-04-16'


print ('=====================>>>>>>>>生成训练数据集')
actions = make_train_set(train_start_date, train_end_date, test_start_date, test_end_date)

#训练集划分
train,test=train_test_split(actions.values,test_size=0.2,random_state=0)
train=pd.DataFrame(train,columns=actions.columns)
test=pd.DataFrame(test,columns=actions.columns)

X_train=train.drop(['user_id','sku_id','label'],axis=1)
X_test=test.drop(['user_id','sku_id','label'],axis=1)
y_train=train[['label']]
y_test=test[['label']]
train_index=train[['user_id','sku_id']].copy()
test_index=test[['user_id','sku_id']].copy()



#生成测试集
print ('=====================>>>>>>>>生成测试数据集')
sub_test_data = make_test_set(sub_start_date, sub_end_date)
sub_trainning_data=sub_test_data.drop(['user_id','sku_id'],axis=1)
sub_user_index=sub_test_data[['user_id','sku_id']].copy()
print("US model  finish  part_4")




# In[ ]:




# In[6]:

#from gen_data0518 import *
import xgboost as xgb

print ('start running ....')

dtrain = xgb.DMatrix(X_train,label=y_train)
dtest = xgb.DMatrix(X_test,label=y_test)
param = {'learning_rate' : 0.1,
        'n_estimators': 1000,
        'max_depth': 3,
        'min_child_weight': 5,
        'gamma': 0,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'eta': 0.05,
        'silent': 1,
        'objective':
        'binary:logistic',
        'scale_pos_weight':1}

num_round =150
plst = list(param.items())
plst += [('eval_metric', 'logloss')]
evallist = [(dtest, 'eval'), (dtrain, 'train')]
bst=xgb.train(plst,dtrain,num_round,evallist,early_stopping_rounds=100)



#预测测试集
#============================================>>>>
print ('==========>>>predict test data label')
sub_trainning_data_1 = xgb.DMatrix(sub_trainning_data)
y = bst.predict(sub_trainning_data_1)
sub_user_index['label'] = y
print ('==========>>>finish test data label')
P=get_basic_product_feat()[['sku_id']]
P['sku_label']=1
pred=pd.merge(sub_user_index,P,on='sku_id',how='left')
pred=pred[pred['sku_label']==1][['user_id','sku_id','label']]

pred.sort_values(by=['user_id','label'],ascending=[0,0],inplace=True)
pred = pred.groupby('user_id').first().reset_index()
result=pred.sort_values(by=['label'],ascending=[0])

result['user_id']=result['user_id'].astype('int')


result.to_csv('./sub/USModel.csv',index=False,index_label=False )
print("finish")



# In[ ]:



