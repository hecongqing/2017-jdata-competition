
# coding: utf-8

# In[1]:

#!/usr/bin/env python

import time

from datetime import datetime
from datetime import timedelta
import pandas as pd
import pickle
import os
import math
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt

action_1_path = "./data/JData_Action_201602.csv"
action_2_path = "./data/JData_Action_201603.csv"
action_3_path = "./data/JData_Action_201604.csv"
user_path = "./data/JData_User.csv"
product_path = "./data/JData_Product.csv"


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

def sub_get_actions(start_date,end_date):
    dump_path = './cache/sub_action_%s_%s.csv' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = pd.read_csv(dump_path)
    else:
        actions=get_actions(start_date,end_date)
        actions=actions[actions['cate']==8]
        actions.to_csv(dump_path,index=False)
    return actions

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
    # actions['user_id']=actions['user_id'].astype('int')
    return actions

# 获取两个时间相差几天
def get_day_chaju(x, end_date):
    #     x=x.split(' ')[0]
    x = datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
    end_date = datetime.strptime(end_date, '%Y-%m-%d')
    return (end_date - x).days




# # 所有行为的总和
# def get_action_feat(start_date, end_date):
#     dump_path = './cache/action_%s_%s.csv' % (start_date, end_date)
#     if os.path.exists(dump_path):
#         actions = pd.read_csv(dump_path)
#     else:
#         actions = get_actions(start_date, end_date)
#         actions = actions[['user_id', 'sku_id', 'type']]
#         df = pd.get_dummies(actions['type'], prefix='action')
#         actions = pd.concat([actions, df], axis=1)  # type: pd.DataFrame
#         actions = actions.groupby(['user_id', 'sku_id'], as_index=False).sum()
#         del actions['type']
#         actions.to_csv(dump_path, index=False)
#     return actions
# top k 天的行为次数总和(滑窗处理)

#user_id,u_action_1_1,u_action_1_2,u_action_1_3,u_action_1_4,u_action_1_5,u_action_1_6
def get_action_feat(start_date, end_date,k):
    dump_path = './cache/u_action_%s_%s_%s.csv' % (start_date, end_date,k)
    if os.path.exists(dump_path):
        actions = pd.read_csv(dump_path)
    else:
        start_days=pd.to_datetime(end_date)-timedelta(days=k)
        start_days=str(start_days).split(' ')[0]
        actions = get_actions(start_days, end_date)
        actions = actions[['user_id', 'type']]
        df = pd.get_dummies(actions['type'], prefix='type')
        actions = pd.concat([actions, df], axis=1)  # type: pd.DataFrame
        actions = actions.groupby('user_id', as_index=False).sum()
        min_max_scaler = preprocessing.MinMaxScaler()
        df = min_max_scaler.fit_transform(actions.drop(['user_id','type'],axis=1).values)
        df = pd.DataFrame(df)
        df.columns=['u_action_'+str(k)+'_'+str(i) for i in range(1,df.shape[1]+1)]
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


# print get_accumulate_user_feat('2016-03-10','2016-04-11')
# 用户购买前访问天数
# 用户购买/加入购物车/关注前访问天数
def get_action_user_feat2(start_date, end_date):
    dump_path = './cache/user_feat2_after_%s_%s.csv' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = pd.read_csv(dump_path)

    else:
        # 用户购买前访问天数
        def user_feat_2_1(start_date, end_date):
            actions = get_actions(start_date, end_date)[['user_id', 'type', 'time']]
            actions['time'] = actions['time'].map(lambda x: x.split(' ')[0])
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
            actions['time'] = actions['time'].map(lambda x: x.split(' ')[0])
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
            actions['time'] = actions['time'].map(lambda x: x.split(' ')[0])
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
            actions['time'] = actions['time'].map(lambda x: x.split(' ')[0])
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
            actions['time'] = actions['time'].map(lambda x: x.split(' ')[0])
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




# # 用户总购买品牌数
# def get_action_user_feat5(start_date, end_date):
#     dump_path = './cache/user_feat5_%s_%s.csv' % (start_date, end_date)
#     if os.path.exists(dump_path):
#         actions = pd.read_csv(dump_path)
#     else:
#         actions = get_actions(start_date, end_date)[['user_id', 'sku_id']]
#         actions = actions.drop_duplicates(['user_id', 'sku_id'], keep='first')
#         actions = actions.groupby('user_id', as_index=False).count()
#         actions.columns = ['user_id', 'sku_num']
#         actions['sku_num'] = actions['sku_num'].astype('float')
#         actions['sku_num'] = actions['sku_num'].map(
#             lambda x: (x - actions['sku_num'].min()) / (actions['sku_num'].max() - actions['sku_num'].min()))
#         actions.to_csv(dump_path, index=False)
#     actions.columns = ['user_id'] + ['u_feat5_' + str(i) for i in range(1, actions.shape[1])]
#     return actions


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
            a=set(actions['level%s' % i].tolist())
            for j in (1, 2,3,4, 5, 6):  # type
                print ('j%s' % j)
                df = actions[actions['type'] == j][['user_id', 'level%s' % i, 'time']]
                df = df.groupby(['user_id', 'level%s' % i]).count()
                df = df.unstack()
                b=df.columns.levels[1].tolist()
                df.columns = ['u_feat12_' + str('level%s_' % i) + str(j) + '_' + str(k) for k in df.columns.levels[1].tolist()]
                if  len(list(a-set(b)))!=0:
                    c=list(a-set(b))
                    for k in c:
                        df['u_feat12_'+str('level%s_' % i)+str(j)+'_'+ str(k)]=0
                columns=df.columns
                dict={}
                for column in columns:
                    k=int(column.split('_')[-1])
                    dict[column]=k
                columns=sorted(dict.items(),key=lambda x: x[1])
                columns=[(columns[t])[0] for t in range(len(columns))]
                df=df[columns]
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
        actions.columns=columns
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


# 最近k天用户访问P集合的商品数/用户访问总体的商品数（k小于7天，不除总体的商品数，反之，除）
def get_action_user_feat0509_1_30(start_date, end_date, n):
    dump_path = './cache/user_feat0509_1_30_%s_%s_%s.csv' % (start_date, end_date, n)
    if os.path.exists(dump_path):
        actions = pd.read_csv(dump_path)
    else:

        start_days = datetime.strptime(end_dfte, '%Y-%m-%d') - timedelta(days=n)
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
    actions.columns = ['user_id'] + ['u_feat30_' + str(n) + '_' + str(i) for i in range(1, actions.shape[1])]
    
    return actions



#用户行为的交叉
def get_action_user_feat16(start_date,end_date):
    dump_path = './cache/user_feat16_%s_%s.csv' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = pd.read_csv(dump_path)
    else:
        actions=get_actions(start_date, end_date)[['user_id', 'type']]
        actions['cnt']=0
        action1 = actions.groupby(['user_id', 'type']).count()
        action1=action1.unstack()
        index_col=list(range(action1.shape[1]))
        action1.columns=index_col
        action1=action1.reset_index()
        action2 = actions.groupby('user_id', as_index=False).count()
        del action2['type']
        action2.columns = ['user_id', 'cnt']
        actions = pd.merge(action1, action2, how='left', on='user_id')
        for i in index_col:
            actions[i] = actions[i] / actions['cnt']
        del actions['cnt']
        actions.to_csv(dump_path,index=False)
    actions.columns = ['user_id'] + ['u_feat16_' + str(i) for i in range(1, actions.shape[1])]
    return actions

#最近k天用户访问P集合的商品数/用户访问总体的商品数（k小于7天，不除总体的商品数，反之，除）
def get_action_user_feat0509_1_30(start_date,end_date,n):
    dump_path='./cache/user_feat0509_1_30_%s_%s_%s.csv'%(start_date,end_date,n)
    if os.path.exists(dump_path):
        actions = pd.read_csv(dump_path)
    else:

        start_days=datetime.strptime(end_date,'%Y-%m-%d')-timedelta(days=n)
        start_days=datetime.strftime(start_days,'%Y-%m-%d')

        actions=get_actions(start_days,end_date)[['user_id','sku_id','type']]
        actions_dummy=pd.get_dummies(actions['type'],prefix='actions')
        actions=pd.concat([actions,actions_dummy],axis=1)
        del actions['type']

        P = get_basic_product_feat()[['sku_id']]
        P['label']=1
        actions_sub=pd.merge(actions,P,on='sku_id',how='left')
        actions_sub=actions_sub[actions_sub['label']==1]
        del actions_sub['label']

        actions_sub=actions_sub.groupby(['user_id'],as_index=False).sum()
        del actions_sub['sku_id']
        actions_all=actions.groupby(['user_id'],as_index=False).sum()
        del actions_all['sku_id']

        if n>7:
            actions=pd.merge(actions_all,actions_sub,on=['user_id'],how='left')
            #print actions.head()
            for i in range(1,7):
                actions['actions_%s'%i]=actions['actions_%s_y'%i]/actions['actions_%s_x'%i]
            #actions=actions[['user_id','actions_1','actions_2','actions_3','actions_4','actions_5','actions_6']]

        else:
            actions = pd.merge(actions_all, actions_sub, on=['user_id'], how='left')
        actions.to_csv(dump_path,index=False)
    actions.columns = ['user_id'] + ['u_feat30_' +str(n)+'_'+ str(i) for i in range(1, actions.shape[1])]  
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

#用户点击到购买的时间间隔
def get_action_user_feat0515_2_1(start_date,end_date):
    dump_path='./cache/get_action_user_feat0515_2_1_%s_%s.csv'%(start_date,end_date)
    if os.path.exists(dump_path):
        actions = pd.read_csv(dump_path)
    else:
        actions = get_actions(start_date,end_date)
        actions_dianji=actions[actions['type']==6][['user_id','sku_id','time']]
        actions_dianji['time_dianji'] = actions_dianji['time'].map(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
        actions_dianji = actions_dianji[['user_id', 'sku_id','time_dianji']]
        actions_dianji= actions_dianji.drop_duplicates(['user_id', 'sku_id'], keep='first')


        actions_goumai=actions[actions['type']==4][['user_id','sku_id','time']]
        actions_goumai['time_goumai'] = actions_goumai['time'].map(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
        actions_goumai = actions_goumai[['user_id', 'sku_id','time_goumai']]
        actions_goumai= actions_goumai.drop_duplicates(['user_id', 'sku_id'], keep='last')

        actions = pd.merge(actions_dianji,actions_goumai,on=['user_id','sku_id'],how='inner')
        actions['time_jiange']=actions['time_goumai']-actions['time_dianji']
        actions=actions.drop(['sku_id','time_goumai','time_dianji'],axis=1)
        actions['time_jiange']=actions['time_jiange'].map(lambda x:x.days*24+x.seconds//3600+1)

        actions_min = actions.groupby('user_id').min().reset_index()
        actions_min.columns = ['user_id','time_min']
        # actions_mean = actions.groupby('user_id').mean().reset_index()
        # actions_mean.columns = ['user_id','time_mean']
        actions_max = actions.groupby('user_id').max().reset_index()
        actions_max.columns = ['user_id','time_max']
        actions=pd.merge(actions_min,actions_max,on='user_id',how='left')
        
        user_id = actions[['user_id']]
        del actions['user_id']
        actions = actions.fillna(0)
        columns = actions.columns
        min_max_scale = preprocessing.MinMaxScaler()
        actions = min_max_scale.fit_transform(actions.values)
        actions = pd.concat([user_id, pd.DataFrame(actions,columns = columns)], axis=1)  
        actions.to_csv(dump_path,index=False)
    return actions


#用户购买每种cate的数量
def get_action_user_feat0515_2_2(start_date,end_date):
    dump_path='./cache/get_action_user_feat0515_2_2_%s_%s.csv'%(start_date,end_date)
    if os.path.exists(dump_path):
        actions = pd.read_csv(dump_path)
    else:
        actions = get_actions(start_date,end_date)
        actions = get_actions(start_date,end_date)[['user_id','cate']]
        cate_col = pd.get_dummies(actions['cate'],prefix='cate')
        actions=pd.concat([actions[['user_id']],cate_col],axis=1)
        actions= actions.groupby('user_id').sum().reset_index()
        
        user_id = actions[['user_id']]
        del actions['user_id']
        actions = actions.fillna(0)
        columns = actions.columns
        min_max_scale = preprocessing.MinMaxScaler()
        actions = min_max_scale.fit_transform(actions.values)
        actions = pd.concat([user_id, pd.DataFrame(actions,columns = columns)], axis=1)  
        actions.to_csv(dump_path,index=False)
    return actions


#获取某人某段时间内加入购物车的数量以及关注的数量
def get_action_user_feat0515_2_3(start_date, end_date, n):
    dump_path = './cache/get_action_user_feat0515_2_3_%s_%s_%s_1.csv' % (start_date, end_date, n)
    if os.path.exists(dump_path):
        actions = pd.read_csv(dump_path)
    else:

        start_days = datetime.strptime(end_date, '%Y-%m-%d') - timedelta(days=n)
        start_days = datetime.strftime(start_days, '%Y-%m-%d')

        actions = get_actions(start_days,end_date)[['user_id','type','cate']]
        actions_gouwuche=actions[actions['type']==2]
        actions_gouwuche_1= actions_gouwuche[['user_id','type']]
        actions_gouwuche_1= actions_gouwuche_1.groupby('user_id').count().reset_index()
        actions_gouwuche_1.columns = ['user_id',str(n)+'gouwuche_add']

        actions_gouwuche_2= actions_gouwuche[actions_gouwuche['cate']==8][['user_id','type']]
        actions_gouwuche_2= actions_gouwuche_2.groupby('user_id').count().reset_index()
        actions_gouwuche_2.columns = ['user_id',str(n)+'gouwuche_add_cate_8']

        actions_guanzhu=actions[actions['type']==5]
        actions_guanzhu_1= actions_guanzhu[['user_id','type']]
        actions_guanzhu_1= actions_guanzhu_1.groupby('user_id').count().reset_index()
        actions_guanzhu_1.columns = ['user_id',str(n)+'guanzhu_add']

        actions_guanzhu_2= actions_guanzhu[actions_guanzhu['cate']==8][['user_id','type']]
        actions_guanzhu_2= actions_guanzhu_2.groupby('user_id').count().reset_index()
        actions_guanzhu_2.columns = ['user_id',str(n)+'guanzhu_add_cate_8']

        actions = pd.merge(actions_gouwuche_1,actions_gouwuche_2,on='user_id',how ='outer')
        actions = pd.merge(actions,actions_guanzhu_1,on='user_id',how ='outer')
        actions = pd.merge(actions,actions_guanzhu_2,on='user_id',how ='outer')
        actions=actions.fillna(0)
        
        user_id = actions[['user_id']]
        del actions['user_id']
        actions = actions.fillna(0)
        columns = actions.columns
        min_max_scale = preprocessing.MinMaxScaler()
        actions = min_max_scale.fit_transform(actions.values)
        actions = pd.concat([user_id, pd.DataFrame(actions,columns = columns)], axis=1)  
        actions.to_csv(dump_path, index=False)
    
    
    return actions

#top n 中 某人使用了多少天产生了该行为
def get_action_user_feat0515_2_4(start_date, end_date, n):
    dump_path = './cache/get_action_user_feat0515_2_4_%s_%s_%s.csv' % (start_date, end_date, n)
    if os.path.exists(dump_path):
        actions = pd.read_csv(dump_path)
    else:

        start_days = datetime.strptime(end_date, '%Y-%m-%d') - timedelta(days=n)
        start_days = datetime.strftime(start_days, '%Y-%m-%d')

        actions = get_actions(start_days,end_date)[['user_id','type','time']]
        actions['time'] = actions['time'].map(lambda x: (datetime.strptime(end_date,'%Y-%m-%d')-datetime.strptime(x, '%Y-%m-%d %H:%M:%S')).days)
        actions=actions.drop_duplicates(['user_id','type','time'])
        actions = actions.groupby(['user_id','type']).count()
        actions.columns = [str(n)+'day_nums']
        actions=actions.unstack()
        actions=actions.reset_index()
        actions.columns = ['user_id'] + ['get_action_user_feat0515_2_4_' + str(n) + '_' + str(i) for i in range(1, actions.shape[1])]
        actions=actions.fillna(0)
        
        user_id = actions[['user_id']]
        del actions['user_id']
        actions = actions.fillna(0)
        columns = actions.columns
        min_max_scale = preprocessing.MinMaxScaler()
        actions = min_max_scale.fit_transform(actions.values)
        actions = pd.concat([user_id, pd.DataFrame(actions,columns = columns)], axis=1)  
        actions.to_csv(dump_path, index=False)  
    return actions


# 用户总购买/加购/关注/点击/浏览品牌数
def get_action_user_feat5(start_date, end_date):
    dump_path = './cache/user_feat5_a_%s_%s.csv' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = pd.read_csv(dump_path)
    else:
        actions = get_actions(start_date, end_date)
        action=None
        for i in (1,2,4,5,6):
            df=actions[actions['type']==i][['user_id', 'sku_id']]
            df = df.drop_duplicates(['user_id', 'sku_id'], keep='first')
            df = df.groupby('user_id', as_index=False).count()
            df.columns = ['user_id', 'num_%s'%i]
            if i==1:
                action=df
            else:
                action=pd.merge(action,df,on='user_id',how='outer')
        actions=action.fillna(0)
        actions = actions.astype('float')
        user=actions[['user_id']]
        min_max_scaler = preprocessing.MinMaxScaler()
        actions = min_max_scaler.fit_transform(actions.drop(['user_id'],axis=1).values)
        actions = pd.DataFrame(actions)
        actions = pd.concat([user, actions], axis=1)
        actions.to_csv(dump_path, index=False)
    actions.columns = ['user_id'] + ['u_feat5_' + str(i) for i in range(1, actions.shape[1])]
    return actions

#top  k 用户总购买/加购/关注/点击/浏览品牌数
def get_action_u0515_feat5(start_date,end_date,k):
    dump_path = './cache/u0515_feat5_%s_%s_%s.csv' % (start_date, end_date,k)
    if os.path.exists(dump_path):
        actions = pd.read_csv(dump_path)
    else:
        start_days=pd.to_datetime(end_date)-timedelta(days=k)
        start_days=str(start_days).split(' ')[0]
        actions=get_action_user_feat5(start_days, end_date)
        actions.to_csv(dump_path,index=False)
    actions.columns=['user_id']+['u0515_feat5_'+str(k)+'_'+str(i) for i in range(1,actions.shape[1])]
    return actions


#最早交互时间
def get_action_u0524_feat1(start_date,end_date):
    dump_path = './cache/u0524_feat1_%s_%s.csv' % (start_date, end_date,)
    if os.path.exists(dump_path):
        actions = pd.read_csv(dump_path)
    else:
        #全集
        actions=get_actions(start_date,end_date)[['user_id','time']]
        actions=actions.groupby('user_id',as_index=False).first()
        actions['time_diff_early']=pd.to_datetime(end_date)-pd.to_datetime(actions['time'])
        actions['time_diff_early']=actions['time_diff_early'].dt.days*24+actions['time_diff_early'].dt.seconds//3600
        actions=actions[['user_id','time_diff_early']]
        #子集
        sub_actions=sub_get_actions(start_date,end_date)[['user_id','time']]
        sub_actions=sub_actions.groupby('user_id',as_index=False).first()
        sub_actions['sub_time_diff_early']=pd.to_datetime(end_date)-pd.to_datetime(sub_actions['time'])
        sub_actions['sub_time_diff_early']=sub_actions['sub_time_diff_early'].dt.days*24+sub_actions['sub_time_diff_early'].dt.seconds//3600
        sub_actions = sub_actions[['user_id', 'sub_time_diff_early']]

        actions=pd.merge(actions,sub_actions,on='user_id',how='left')
        actions=actions.fillna(0)
        min_max_scale = preprocessing.MinMaxScaler()
        action = min_max_scale.fit_transform(actions.drop(['user_id'], axis=1).values)
        actions = pd.concat([actions[['user_id']], pd.DataFrame(action)], axis=1)
        actions.to_csv(dump_path,index=False)
    actions.columns=['user_id']+['u0524_feat1_'+str(i)for i in range(1,actions.shape[1])]
    return actions

#最晚交互时间
def get_action_u0524_feat2(start_date,end_date):
    dump_path = './cache/u0524_feat2_%s_%s.csv' % (start_date, end_date,)
    if os.path.exists(dump_path):
        actions = pd.read_csv(dump_path)
    else:
        # 全集
        actions = get_actions(start_date, end_date)[['user_id', 'time']]
        actions = actions.groupby('user_id', as_index=False).last()
        actions['time_diff_recent'] = pd.to_datetime(end_date) - pd.to_datetime(actions['time'])
        actions['time_diff_recent'] = actions['time_diff_recent'].dt.days * 24 + actions['time_diff_recent'].dt.seconds // 3600
        actions = actions[['user_id', 'time_diff_recent']]
        # 子集
        sub_actions = sub_get_actions(start_date, end_date)[['user_id', 'time']]
        sub_actions = sub_actions.groupby('user_id', as_index=False).last()
        sub_actions['sub_time_diff_recent'] = pd.to_datetime(end_date) - pd.to_datetime(sub_actions['time'])
        sub_actions['sub_time_diff_recent'] = sub_actions['sub_time_diff_recent'].dt.days * 24 + sub_actions['sub_time_diff_recent'].dt.seconds // 3600
        sub_actions = sub_actions[['user_id', 'sub_time_diff_recent']]

        actions = pd.merge(actions, sub_actions, on='user_id', how='left')
        actions=actions.fillna(0)
        min_max_scale = preprocessing.MinMaxScaler()
        action = min_max_scale.fit_transform(actions.drop(['user_id'], axis=1).values)
        actions = pd.concat([actions[['user_id']], pd.DataFrame(action)], axis=1)
        actions.to_csv(dump_path,index=False)
    actions.columns = ['user_id'] + ['u0524_feat2_' + str(i) for i in range(1, actions.shape[1])]
    return actions


#活跃天数
def get_action_u0524_feat3(start_date,end_date):
    dump_path = './cache/u0524_feat3_%s_%s.csv' % (start_date, end_date,)
    if os.path.exists(dump_path):
        actions = pd.read_csv(dump_path)
    else:
        #全集
        actions=get_actions(start_date,end_date)
        actions['time']=pd.to_datetime(actions['time']).dt.date
        actions=actions.drop_duplicates(['user_id','time'])[['user_id','time']]
        actions=actions.groupby('user_id',as_index=False).count()
        #子集
        sub_actions=sub_get_actions(start_date,end_date)
        sub_actions['time']=pd.to_datetime(sub_actions['time']).dt.date
        sub_actions=sub_actions.drop_duplicates(['user_id','time'])[['user_id','time']]
        sub_actions=sub_actions.groupby('user_id',as_index=False).count()
        actions=pd.merge(actions,sub_actions,on='user_id',how='left')
        actions=actions.fillna(0)
        min_max_scale = preprocessing.MinMaxScaler()
        action = min_max_scale.fit_transform(actions.drop(['user_id'], axis=1).values)
        actions = pd.concat([actions[['user_id']], pd.DataFrame(action)], axis=1)
        actions.to_csv(dump_path,index=False)
    actions.columns=['user_id']+['u0524_feat3_'+str(i) for i in range(1,actions.shape[1])]
    return actions


#点击模块
def get_action_user_feat0509_1_31(start_date,end_date,n):
    dump_path='./cache/user_feat0509_1_31_%s_%s_%s.csv'%(start_date,end_date,n)
    if os.path.exists(dump_path):
        actions = pd.read_csv(dump_path)
    else:
        start_days=datetime.strptime(end_date,'%Y-%m-%d')-timedelta(days=n)
        start_days=datetime.strftime(start_days,'%Y-%m-%d')
        actions=get_actions(start_days,end_date)
        actions=actions[actions['type']==6][['user_id','model_id']]
        
#         actions = actions.drop('type',axis=1)
        
        actions_click_sum=actions[['user_id','model_id']].groupby('user_id').count().reset_index()
        actions_click_sum.columns = ['user_id',str(n)+'click_sum_all']
        actions[str(n)+'u_click14_history'] = actions['model_id'].map(lambda x: int(x == 14))
        actions[str(n)+'u_click21_history'] = actions['model_id'].map(lambda x: int(x == 21))
        actions[str(n)+'u_click28_history'] = actions['model_id'].map(lambda x: int(x == 28))
        actions[str(n)+'u_click110_history'] = actions['model_id'].map(lambda x: int(x == 110))
        actions[str(n)+'u_click210_history'] = actions['model_id'].map(lambda x: int(x == 210))
        actions = actions.groupby('user_id').sum().reset_index().drop('model_id', axis=1)
#         actions.to_csv(dump_path,index=False)
        actions = pd.merge(actions,actions_click_sum,how='left',on='user_id')
        
        actions[str(n)+'u_click14/click_sum_history'] = actions[str(n)+'u_click14_history']/actions[str(n)+'click_sum_all']
        actions[str(n)+'u_click21/click_sum_history'] = actions[str(n)+'u_click21_history']/actions[str(n)+'click_sum_all']
        actions[str(n)+'u_click28/click_sum_history'] = actions[str(n)+'u_click28_history']/actions[str(n)+'click_sum_all']
        actions[str(n)+'u_click110/click_sum_history'] = actions[str(n)+'u_click110_history']/actions[str(n)+'click_sum_all']
        actions[str(n)+'u_click210/click_sum_history'] = actions[str(n)+'u_click210_history']/actions[str(n)+'click_sum_all']
        
        user_id = actions[['user_id']]
        del actions['user_id']
        actions = actions.fillna(0)
        columns = actions.columns
        min_max_scale = preprocessing.MinMaxScaler()
        actions = min_max_scale.fit_transform(actions.values)
        actions = pd.concat([user_id, pd.DataFrame(actions,columns = columns)], axis=1)
        actions.to_csv(dump_path,index=False)
    return actions
#u模型cate=8的购买者和不是cate=8的购买者
def get_action_u0513_feat16(start_date,end_date):
    dump_path = './cache/u0513_feat16_%s_%s.csv' % (start_date, end_date)
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
        actions.to_csv(dump_path,index=False)
    return actions

#get_action_u0513_feat16('2016-02-01','2016-04-16')
# 用户层级特征
def get_action_user_feat_six_xingwei(start_date, end_date, n):
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
        df.columns = ['user_id'] + ['get_action_user_feat_six_xingwei_' + str(n) + '_' + str(i) for i in range(1, df.shape[1])]
        df.to_csv(dump_path, index=False)
        actions=df
        
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
    actions.columns = ['user_id'] + ['get_action_user_feat_six_xingwei_' + str(n) + '_' + str(i) for i in range(1, actions.shape[1])]
    return actions


def deal_user_six_deal(start_date, end_date, n):
    dump_path = './cache/deal_user_six_action_%s_%s_%s_int.csv' % (start_date, end_date, n)
    if os.path.exists(dump_path):
        actions = pd.read_csv(dump_path)
        actions.columns = ['user_id'] + ['u_featsix_deal_' + str(n) + '_' + str(i) for i in range(1, actions.shape[1])]
        return actions
    else:
        temp = get_action_user_feat_six_xingwei(start_date, end_date, n)  # 修改
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

# # get  user sku
# def get_user(start_date, end_date):
#     dump_path = './cache/user_sku_%s_%s.csv' % (start_date, end_date)
#     if os.path.exists(dump_path):
#         actions = pd.read_csv(dump_path)
#     else:
#         actions = get_actions(start_date, end_date)
#         actions = actions[(actions['type'] == 2) | (actions['type'] == 5) | (actions['type'] == 4)]
#         actions=actions[actions['cate']==8]
#         actions = actions[['user_id']]
#         actions = actions.drop_duplicates(['user_id'], keep='first')
#         actions.to_csv(dump_path, index=False)
#     return actions


#用户购买前的行为
def get_action_u0509_feat_28(start_date, end_date,k):
    dump_path = './cache/u0509_feat_28_%s_%s_%s.csv' % (start_date, end_date,k)
    if os.path.exists(dump_path):
        actions = pd.read_csv(dump_path)
    else:
        actions = get_actions(start_date, end_date)
        actions = actions[actions['type'] == 4]
        actions['time_buy'] = actions['time'].map(lambda x: datetime.strptime(x.split(' ')[0], '%Y-%m-%d'))
        actions = actions[['user_id', 'sku_id', 'time_buy']].reset_index(drop=True)
        actions['before_time_buy'] = actions['time_buy'] - timedelta(days=k)

        df = get_actions('2016-02-01','2016-04-16')[['user_id', 'sku_id', 'time', 'type']]
        df['time'] = df['time'].map(lambda x: datetime.strptime(x.split(' ')[0], '%Y-%m-%d'))
        df = pd.merge(df, actions, on=['user_id', 'sku_id'], how='left')
        df = df.dropna(axis=0, how='any')
        df['before_days'] = (df['time'] - df['before_time_buy']).dt.days
        df['days'] = (df['time'] - df['time_buy']).dt.days
        df = df[(df['before_days'] >= 0) & (df['days'] < 0)]
        df_dummy = pd.get_dummies(df['type'], prefix='type')

        df = pd.concat([df, df_dummy], axis=1)[
            ['user_id', 'sku_id', 'type_1', 'type_2', 'type_3', 'type_4', 'type_5', 'type_6']]

        df = df.groupby(['user_id', 'sku_id'], as_index=False).sum()
        del df['sku_id']
        df = df.groupby('user_id', as_index=False).agg(['min', 'max', 'mean'])
        df = df.reset_index()
        df.columns = ['user_id'] + ['u0509_feat28_' + str(k) + '_' + i for i in (
        'type_1_min', 'type_1_max', 'type_1_mean', 'type_2_min', 'type_2_max', 'type_2_mean',
        'type_3_min', 'type_3_max', 'type_3_mean', 'type_4_min', 'type_4_max', 'type_4_mean',
        'type_5_min', 'type_5_max', 'type_5_mean', 'type_6_min', 'type_6_max', 'type_6_mean')]
        min_max_scaler = preprocessing.MinMaxScaler()
        actions = min_max_scaler.fit_transform(df.drop('user_id', axis=1).values)
        actions = pd.DataFrame(actions)
        actions = pd.concat([df[['user_id']], actions], axis=1)
        actions.columns = ['user_id']+['u0509_feat_28_'+str(i) for i in range(1,actions.shape[1])]
        actions.to_csv(dump_path,index=False)
    actions.columns = ['user_id']+['u0509_feat_28_'+str(k)+"_"+str(i) for i in range(1,actions.shape[1])]
    return actions

#用户看了几个cate=8中的brand、用户看的cate=8的brand/用户看的brand
def get_action_u0509_feat_29(start_date,end_date):
    dump_path = './cache/u0509_feat_29_%s_%s.csv' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = pd.read_csv(dump_path)
    else:
        actions=get_actions(start_date,end_date)
        df1=actions[actions['cate']==8].drop_duplicates(['user_id','brand'])[['user_id','brand']]
        df1=df1.groupby(['user_id'],as_index=False).count()
        df1.columns=['user_id','brand_cate=8']
        df2=actions.drop_duplicates(['user_id','brand'])[['user_id','brand']]
        df2 = df2.groupby(['user_id'], as_index=False).count()
        df2.columns=['user_id','brand_cate_all']
        df=pd.merge(df1,df2,on='user_id',how='right')
        df['rate']=df['brand_cate=8']/df['brand_cate_all']
#         print df
        actions=df.fillna(0)
        actions.to_csv(dump_path,index=False)
    actions.columns=['user_id']+['u0509_feat_29'+str(i) for i in range(1,actions.shape[1])]
    return actions

def get_action_u0521_feat_31(start_date,end_date,k):
    dump_path = './cache/u0509_feat_31_%s_%s_%s.csv' % (start_date, end_date,k)
    if os.path.exists(dump_path):
        actions = pd.read_csv(dump_path)
    else:
        start_days=pd.to_datetime(end_date)-timedelta(days=k)
        start_days=datetime.strftime(start_days,'%H-%m-%d')
        actions=get_actions(start_days,end_date)
        df1=actions[actions['cate']==8].drop_duplicates(['user_id','cate'])[['user_id','cate']]
        df1=df1.groupby('user_id',as_index=False).count()
        df1.columns=['user_id','cate8']
        df2=actions.drop_duplicates(['user_id','cate'])[['user_id','cate']]
        df2=df2.groupby('user_id',as_index=False).count()
        actions=pd.merge(df1,df2,on='user_id',how='right')
        actions['cate8/cate']=actions['cate8']/actions['cate']
        actions=actions.fillna(0)
        min_max_scaler = preprocessing.MinMaxScaler()
        df = min_max_scaler.fit_transform(actions[['cate8','cate']].values)
        df = pd.DataFrame(df)
        actions = pd.concat([actions[['user_id','cate8/cate']], df], axis=1)
        actions.to_csv(dump_path,index=False)
    actions.columns=['user_id']+['u0509_feat_31_'+str(k)+'_'+str(i)for i in range(1,actions.shape[1])]
    return actions


def get_action_u0521_feat_32(start_date,end_date):
    dump_path = './cache/u0509_feat_32_%s_%s.csv' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = pd.read_csv(dump_path)
    else:
        actions=get_actions(start_date,end_date)
        actions=actions[actions['cate']==8][['user_id','brand']]
        df1=actions.drop_duplicates(['user_id','brand']).groupby('user_id',as_index=False).count()
        df1.columns=['user_id','brand_num']
        df2=actions.groupby('user_id',as_index=False).count()
        actions=pd.merge(df1,df2,on='user_id',how='left')
        actions['brand_num/brand']=actions['brand']/actions['brand_num']
        actions=actions.fillna(0)
        min_max_scaler = preprocessing.MinMaxScaler()
        df = min_max_scaler.fit_transform(actions.drop(['user_id'],axis=1).values)
        df = pd.DataFrame(df)
        actions = pd.concat([actions[['user_id']], df], axis=1)
        actions.to_csv(dump_path, index=False)
    actions.columns = ['user_id'] + ['u0509_feat_32_' + str(i) for i in range(1, actions.shape[1])]
    return actions

def get_action_user_feat7_0522_huachuang(start_date, end_date,n):
    dump_path = './cache/user_feat7_six_%s_%s_%s_0522.csv' % (start_date, end_date,n)
    if os.path.exists(dump_path):
        actions = pd.read_csv(dump_path)
    else:
        start_days = datetime.strptime(end_date, '%Y-%m-%d') - timedelta(days=n)
        start_days = datetime.strftime(start_days, '%Y-%m-%d')
        
        df = get_actions(start_days, end_date)[['user_id', 'type', 'time']]
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
    actions.columns = ['user_id'] + ['u_feat7_' +str(n)+"_"+ str(i) for i in range(1, actions.shape[1])]
    return actions

def get_user_labels(test_start_date,test_end_date):
    dump_path = './cache/user_labels_%s_%s_11.csv' % (test_start_date, test_end_date)
    if os.path.exists(dump_path):
        actions = pd.read_csv(dump_path)
    else:
        actions = get_actions(test_start_date, test_end_date)
        actions = actions[actions['cate']==8]
        actions = actions[actions['type'] == 4].drop_duplicates(['user_id'])[['user_id']]
        actions['label'] = 1

    return actions


print("U model 0 finish  part_1")
#########################################################################################################


# In[ ]:




# In[2]:

import os
from datetime import datetime
from datetime import timedelta

# -*- coding: utf-8 -*-
"""
Created on Sun May 14 10:27:41 2017
@author: 老虎趴趴走
"""
import pandas as pd
import numpy as np
import math

def user_features(user, ful_action, sub_action, end_date):
    dump_path='./cache/user_features_%s_0514_2.csv'%(end_date)
    if os.path.exists(dump_path):
        actions = pd.read_csv(dump_path)
    
    else:
        end_date=pd.to_datetime(end_date)
        day = timedelta(1, 0)
        print('=====> 提取特征...')
        sub_1 = sub_action[(sub_action['time']>=end_date-1*day) & (sub_action['time']<end_date)]
        sub_3 = sub_action[(sub_action['time']>=end_date-3*day) & (sub_action['time']<end_date)]
        sub_5 = sub_action[(sub_action['time']>=end_date-5*day) & (sub_action['time']<end_date)]
        sub_30 = sub_action[(sub_action['time']>=end_date-30*day) & (sub_action['time']<end_date)]
        sub_all = sub_action[sub_action['time']<end_date]

        ful_5 = ful_action[(ful_action['time']>=end_date-5*day) & (ful_action['time']<end_date)]
        ful_30 = ful_action[(ful_action['time']>=end_date-30*day) & (ful_action['time']<end_date)]
        ful_all = ful_action[ful_action['time']<end_date]
        # ========================================
        #    用户历史行为  
        # ========================================
        # 6种行为特征
        df = pd.get_dummies(sub_all['type'], prefix='type')
        df['type_0'] = df.sum(axis=1)
        df = pd.concat([sub_all[['user_id', 'sku_id']], df], axis=1)
#         u_feature_history = action_dummy[['user_id']]
        u_feature_all = df.drop('sku_id', axis=1).groupby('user_id').sum().reset_index()
        col = ['user_id','browse_all','add_all','del_all','buy_all','follow_all','click_all','action_all']
        u_feature_all.columns = col
        # 比值
        u_feature_all['buy/browse_all'] = u_feature_all['buy_all']/(u_feature_all['browse_all']+0.001)*100
        u_feature_all['buy/add_all'] = u_feature_all['buy_all']/(u_feature_all['add_all']+0.001)*100
        u_feature_all['buy/click_all'] = u_feature_all['buy_all']/(u_feature_all['click_all']+0.001)*100
        u_feature_all['buy/follow_all'] = u_feature_all['buy_all']/(u_feature_all['follow_all']+0.001)*100
        u_feature_all['del/add_all'] = u_feature_all['del_all']/(u_feature_all['add_all']+0.001)*100

        # 用户对商品行为特征
        us = df.groupby(['user_id', 'sku_id']).sum().reset_index()
        us = us.drop('sku_id', axis=1)
        us_avg = us.groupby('user_id').mean().reset_index()
        col = ['user_id','us_browse_all_avg','us_add_all_avg','us_del_all_avg','us_buy_all_avg','us_follow_all_avg','us_click_all_avg','us_action_all_avg']
        us_avg.columns = col
        us_max = us.groupby('user_id').max().reset_index()
        col = ['user_id','us_browse_all_max','us_add_all_max','us_del_all_max','us_buy_all_max','us_follow_all_max','us_click_all_max','us_action_all_max']
        us_max.columns = col
        us_max = us_max.drop(['us_buy_all_max', 'us_del_all_max'], axis=1)
        u_feature_all = pd.merge(u_feature_all, us_avg, on='user_id', how='left')
        u_feature_all = pd.merge(u_feature_all, us_max, on='user_id', how='left').fillna(0)

        # 活跃天数
        u_days = sub_all[['user_id', 'date']]
        u_days = u_days.drop_duplicates()
        u_days = u_days.groupby('user_id').count().reset_index()
        u_days.rename(columns={'date': 'u_days_all'}, inplace=True)
        u_feature_all = pd.merge(u_feature_all, u_days, on='user_id', how='left').fillna(0)

        # 时间特征
        u_days = sub_all[['user_id', 'time']]
        u_start = u_days.groupby('user_id').min().reset_index()
        u_start.rename(columns={'time': 'start'}, inplace=True)
        u_end = u_days.groupby('user_id').max().reset_index()
        u_end.rename(columns={'time': 'end'}, inplace=True)
        u_duration = pd.merge(u_start, u_end, on='user_id')
        u_duration['u_duration_all'] = u_duration['end'] - u_duration['start']
        u_duration['u_duration_all'] = u_duration['u_duration_all'].map(lambda x: x.days*24+x.seconds/3600)
        u_duration = u_duration[['user_id', 'u_duration_all']]
        u_feature_all = pd.merge(u_feature_all, u_duration, on='user_id', how='left').fillna(0)

        # 行为/时间
        u_feature_all['action_avg_all'] = u_feature_all['action_all']/(u_feature_all['u_duration_all']+0.001)
        u_feature_all['browse_avg_all'] = u_feature_all['browse_all']/(u_feature_all['u_duration_all']+0.001)
        u_feature_all['add_avg_all'] = u_feature_all['add_all']/(u_feature_all['u_duration_all']+0.001)
        u_feature_all['del_avg_all'] = u_feature_all['del_all']/(u_feature_all['u_duration_all']+0.001)
        u_feature_all['buy_avg_all'] = u_feature_all['buy_all']/(u_feature_all['u_duration_all']+0.001)
        u_feature_all['follow_avg_all'] = u_feature_all['follow_all']/(u_feature_all['u_duration_all']+0.001)
        u_feature_all['click_avg_all'] = u_feature_all['click_all']/(u_feature_all['u_duration_all']+0.001)

        # 商品特征
        sku = df.drop('user_id', axis=1).groupby('sku_id').sum().reset_index()
        sku = pd.merge(df[['user_id', 'sku_id']].drop_duplicates(), sku, on='sku_id', how='left')
        sku = sku.drop('sku_id', axis=1)
        sku_avg = sku.groupby('user_id').mean().reset_index()
        col = ['user_id','sku_browse_all_avg','sku_add_all_max','sku_del_all_max','sku_buy_all_max','sku_follow_all_max','sku_click_all_max','sku_action_all_max']
        sku_avg.columns = col        

        sku_min = sku.groupby('user_id').min().reset_index()
        col = ['user_id','sku_browse_all_min','sku_add_all_min','sku_del_all_min','sku_buy_all_min','sku_follow_all_min','sku_click_all_min','sku_action_all_min']
        sku_min.columns = col  
        
        u_feature_all = pd.merge(u_feature_all, sku_avg, on='user_id', how='left')
        u_feature_all = pd.merge(u_feature_all, sku_min, on='user_id', how='left').fillna(0)

        # 全集行为特征
        df = pd.get_dummies(ful_all['type'], prefix='type')
        df['type_0'] = df.sum(axis=1)
        df = pd.concat([ful_all[['user_id', 'sku_id']], df], axis=1)
        u_feature_ful_all = df.drop('sku_id', axis=1).groupby('user_id').sum().reset_index()
        col = ['user_id','browse_ful_all','add_ful_all','del_ful_all','buy_ful_all','follow_ful_all','click_ful_all','action_ful_all']
        u_feature_ful_all.columns = col     

        u_feature_all = pd.merge(u_feature_all, u_feature_ful_all, on='user_id', how='left')
 
        # 子集/全集
        u_feature_all['action/ful_all'] = u_feature_all['action_all']/(u_feature_all['action_ful_all']+0.001)*100
        u_feature_all['browse/ful_all'] = u_feature_all['browse_all']/(u_feature_all['browse_ful_all']+0.001)*100
        u_feature_all['add/ful_all'] = u_feature_all['add_all']/(u_feature_all['add_ful_all']+0.001)*100
        u_feature_all['del/ful_all'] = u_feature_all['del_all']/(u_feature_all['del_ful_all']+0.001)*100
        u_feature_all['buy/ful_all'] = u_feature_all['buy_all']/(u_feature_all['buy_ful_all']+0.001)*100
        u_feature_all['follow/ful_all'] = u_feature_all['follow_all']/(u_feature_all['follow_ful_all']+0.001)*100
        u_feature_all['click/ful_all'] = u_feature_all['click_all']/(u_feature_all['click_ful_all']+0.001)*100
        u_feature_all = u_feature_all.drop(['browse_ful_all', 'action_ful_all', 'add_ful_all', 'del_ful_all',
                                                    'buy_ful_all', 'follow_ful_all', 'click_ful_all'], axis=1)


        # =======================================
        #    用户30天行为特征
        # =======================================
        df = pd.get_dummies(sub_30['type'], prefix='type')
        df['type_0'] = df.sum(axis=1)
        df = pd.concat([sub_30[['user_id', 'sku_id']], df], axis=1)
        # 子集行为特征
        u_feature_30 = df.drop('sku_id', axis=1).groupby('user_id').sum().reset_index()
        col = ['user_id','browse_30','add_30','del_30','buy_30','follow_30','click_30','action_30']
        u_feature_30.columns = col

        # 全集行为特征
        df = pd.get_dummies(ful_30['type'], prefix='type')
        df['type_0'] = df.sum(axis=1)
        df = pd.concat([ful_30[['user_id', 'sku_id']], df], axis=1)
        u_feature_ful_30 = df.drop('sku_id', axis=1).groupby('user_id').sum().reset_index()
        col = ['user_id','browse_ful_30','add_ful_30','del_ful_30','buy_ful_30','follow_ful_30','click_ful_30','action_ful_30']
        u_feature_ful_30.columns = col        

        u_feature_30 = pd.merge(u_feature_30, u_feature_ful_30, on='user_id', how='left')

        # 子集/全集
        u_feature_30['action/ful_30'] = u_feature_30['action_30']/(u_feature_30['action_ful_30']+0.001)*100
        u_feature_30['browse/ful_30'] = u_feature_30['browse_30']/(u_feature_30['browse_ful_30']+0.001)*100
        u_feature_30['add/ful_30'] = u_feature_30['add_30']/(u_feature_30['add_ful_30']+0.001)*100
        u_feature_30['del/ful_30'] = u_feature_30['del_30']/(u_feature_30['del_ful_30']+0.001)*100
        u_feature_30['buy/ful_30'] = u_feature_30['buy_30']/(u_feature_30['buy_ful_30']+0.001)*100
        u_feature_30['follow/ful_30'] = u_feature_30['follow_30']/(u_feature_30['follow_ful_30']+0.001)*100
        u_feature_30['click/ful_30'] = u_feature_30['click_30']/(u_feature_30['click_ful_30']+0.001)*100

        # ========================================
        #     用户5天行为特征
        # ========================================
        df = pd.get_dummies(sub_5['type'], prefix='type')
        df['type_0'] = df.sum(axis=1)
        df = pd.concat([sub_5[['user_id', 'sku_id']], df], axis=1)
        # 子集行为特征
        u_feature_5 = df.drop('sku_id', axis=1).groupby('user_id').sum().reset_index()
        col = ['user_id','browse_5','add_5','del_5','buy_5','follow_5','click_5','action_5']
        u_feature_5.columns = col

        # 用户对商品行为特征
        us = df.groupby(['user_id', 'sku_id']).sum().reset_index()
        us = us.drop('sku_id', axis=1)
        us_avg = us.groupby('user_id').mean().reset_index()
        col = ['user_id','us_browse_5_avg','us_add_5_avg','us_del_5_avg','us_buy_5_avg','us_follow_5_avg','us_click_5_avg','us_action_5_avg']
        us_avg.columns = col
        us_max = us.groupby('user_id').max().reset_index()
        col = ['user_id','us_browse_5_max','us_add_5_max','us_del_5_max','us_buy_5_max','us_follow_5_max','us_click_5_max','us_action_5_max']
        us_max.columns = col
        u_feature_5 = pd.merge(u_feature_5, us_avg, on='user_id', how='left')
        u_feature_5 = pd.merge(u_feature_5, us_max, on='user_id', how='left').fillna(0)

        # 时间特征
        u_days = sub_5[['user_id', 'time']]
        u_start = u_days.groupby('user_id').min().reset_index()
        u_start.rename(columns={'time': 'start'}, inplace=True)
        u_end = u_days.groupby('user_id').max().reset_index()
        u_end.rename(columns={'time': 'end'}, inplace=True)
        u_duration = pd.merge(u_start, u_end, on='user_id')
        u_duration['u_duration_5'] = u_duration['end'] - u_duration['start']
        u_duration['u_duration_5'] = u_duration['u_duration_5'].map(lambda x: x.days*24+x.seconds/3600)
        u_duration['u_stop_5'] = end_date - u_duration['end']
        u_duration['u_stop_5']= u_duration['u_stop_5'].map(lambda x: x.days*24+x.seconds/3600)
        u_duration = u_duration[['user_id', 'u_duration_5', 'u_stop_5']]
        u_feature_5 = pd.merge(u_feature_5, u_duration, on='user_id', how='left').fillna(0)

        # 全集行为特征
        df = pd.get_dummies(ful_5['type'], prefix='type')
        df['type_0'] = df.sum(axis=1)
        df = pd.concat([ful_5[['user_id', 'sku_id']], df], axis=1)
        u_feature_ful_5 = df.drop('sku_id', axis=1).groupby('user_id').sum().reset_index()
        col = ['user_id','browse_ful_5','add_ful_5','del_ful_5','buy_ful_5','follow_ful_5','click_ful_5','action_ful_5']
        u_feature_ful_5.columns = col

        # 子集/全集
        u_feature_5 = pd.merge(u_feature_5, u_feature_ful_5, on='user_id', how='left')
        u_feature_5['browse/ful_5'] = u_feature_5['browse_5'] / (u_feature_5['browse_ful_5']+0.001)*100
        u_feature_5['add/ful_5'] = u_feature_5['add_5'] / (u_feature_5['add_ful_5']+0.001)*100
        u_feature_5['del/ful_5'] = u_feature_5['del_5'] / (u_feature_5['del_ful_5']+0.001)*100
        u_feature_5['click/ful_5'] = u_feature_5['click_5'] / (u_feature_5['click_ful_5']+0.001)*100
        #u_feature_5D = u_feature_5D.drop(['u_browse_num_ful_5D','u_add_num_ful_5D','u_del_num_ful_5D','u_buy_num_ful_5D','u_follow_num_ful_5D','u_click_num_ful_5D'], axis=1)

        # ========================================
        #     用户3天行为特征  
        # ========================================
        df = pd.get_dummies(sub_3['type'], prefix='type')
        df['type_0'] = df.sum(axis=1)
        df = pd.concat([sub_3[['user_id', 'sku_id']], df], axis=1)
        u_feature_3 = df.groupby('user_id')['type_0'].sum().reset_index()
        u_feature_3.rename(columns={'type_0': 'action_3'}, inplace=True)

        # ========================================
        #     用户1天行为特征  
        # ========================================
        df = pd.get_dummies(sub_1['type'], prefix='type')
        df['type_0'] = df.sum(axis=1)
        df = pd.concat([sub_1[['user_id', 'sku_id']], df], axis=1)

        u_feature_1 = df.drop('sku_id', axis=1).groupby('user_id').sum().reset_index()
        col = ['user_id','browse_1','add_1','del_1','buy_1','follow_1','click_1','action_1']
        u_feature_1.columns = col
        # ========================================
        #          特征融合
        # ========================================
        actions = pd.merge(user[['user_id', 'user_lv_cd', 'reg_duration', 'reg_duration_cate']], u_feature_all, on='user_id', how='left')
        actions['lv/reg_day'] = actions['user_lv_cd']/(actions['reg_duration']+0.001)*100
        actions['lv/reg_day_cate'] = actions['user_lv_cd']/(actions['reg_duration_cate']+0.001)
        actions = pd.merge(actions, u_feature_30, on='user_id', how='left')
        actions = pd.merge(actions, u_feature_5, on='user_id', how='left')
        actions['action_5D/all'] = actions['action_5']/(actions['action_all']+0.001)
        actions = pd.merge(actions, u_feature_3, on='user_id', how='left')
        actions = pd.merge(actions, u_feature_1, on='user_id', how='left').fillna(0)

        actions['action_diff1'] = actions['action_1']-actions['action_avg_all']
        actions['browse_diff1'] = actions['browse_1'] - actions['browse_avg_all']
        actions['add_diff1'] = actions['add_1'] - actions['add_avg_all']
        actions['del_diff1'] = actions['del_1'] - actions['del_avg_all']
        actions['buy_diff1'] = actions['buy_1'] - actions['buy_avg_all']
        actions['follow_diff1'] = actions['follow_1'] - actions['follow_avg_all']
        actions['click_diff1'] = actions['click_1'] - actions['click_avg_all']

        print('=====> 完成!')
        actions.to_csv(dump_path,index=False)
        
#     user_id = actions[['user_id']]
#     del actions['user_id']
#     actions = actions.fillna(0)
#     actions=actions.replace(np.inf,0)
#         print(actions.head())
#     columns = actions.columns

#     min_max_scale = preprocessing.MinMaxScaler()
#     actions=actions.replace(np.inf,0)
#     actions = min_max_scale.fit_transform(actions.values)
#     actions = pd.concat([user_id, pd.DataFrame(actions,columns = columns)], axis=1)
    return actions

import pandas as pd
ful_action = pd.read_csv('./data/JData_Action.csv', parse_dates=[2], infer_datetime_format=True)
sub_action = pd.read_csv('./data/JData_subset_action.csv', parse_dates=[2, 7], infer_datetime_format=True) 
user = pd.read_csv('./data/JData_modified_user.csv', parse_dates=[4])
# user_features(user,ful_action,sel_action,'2016-04-11')
print("U model 0 finish  part_2")
######################################################################################


# In[ ]:




# In[10]:


def make_test_set(train_start_date, train_end_date,user,ful_action,sub_action):
    dump_path = './cache/bu0525model_0_u_test_set_%s_%s.csv' % (train_start_date, train_end_date)
    if os.path.exists(dump_path):
        actions = pd.read_csv(dump_path)
    else:
        start_days=str(pd.to_datetime(train_end_date)-timedelta(days=30)).split(' ')[0]
        actions_1 = get_actions(start_days, train_end_date)
        actions=actions_1[actions_1['cate']==8][['user_id']].drop_duplicates(['user_id'])
#         buy_actions = actions_1[(actions_1['type']==4)&(actions_1['cate']==8)][['user_id']].drop_duplicates()
#         actions = actions[actions['user_id'].isin(buy_actions['user_id'])==False]
        
#         start_days=str(pd.to_datetime(train_end_date)-timedelta(days=30)).split(' ')[0]
#         actions_1 = get_actions(start_days, train_end_date)
#         actions_1 = actions_1[(actions_1['type']==2)|(actions_1['type']==4)|(actions_1['type']==5)]
#         actions_1=actions_1[actions_1['cate']==8][['user_id']].drop_duplicates(['user_id'])
        
        
#         actions = pd.concat([actions,actions_1]).drop_duplicates(['user_id'])


        print (actions.shape)
#         start_days = train_start_date
        start_days = "2016-02-01"
#         actions = pd.merge(actions,get_basic_user_feat() , how='left', on='user_id')
#         print(actions.shape)
#         
     
#         actions = pd.merge(actions, get_action_user_feat1(start_days, train_end_date), how='left', on='user_id')
#         print(actions.shape)
        actions = pd.merge(actions, get_action_user_feat2(start_days, train_end_date), how='left', on='user_id')
        print(actions.shape)
        actions = pd.merge(actions, get_action_user_feat5(start_days, train_end_date), how='left', on='user_id')
        print(actions.shape)
        actions = pd.merge(actions, get_action_user_feat6(start_days, train_end_date), how='left', on='user_id')
        print(actions.shape)
        actions = pd.merge(actions, get_action_user_feat6_six(start_days, train_end_date), how='left', on='user_id')
        print(actions.shape)
        actions = pd.merge(actions, get_action_user_feat7(start_days, train_end_date), how='left', on='user_id')
        print(actions.shape)
        actions = pd.merge(actions, get_action_user_feat8(start_days, train_end_date), how='left', on='user_id')
        print (actions.shape)
        actions = pd.merge(actions, get_action_user_feat8_2(start_days, train_end_date), how='left', on='user_id')
        print (actions.shape)
        actions = pd.merge(actions, get_action_user_feat9(start_days, train_end_date), how='left', on='user_id')
        print (actions.shape)
        actions = pd.merge(actions, get_action_user_feat10(start_days, train_end_date), how='left', on='user_id')
        print (actions.shape)
        actions = pd.merge(actions, get_action_user_feat12(train_start_date, train_end_date), how='left', on='user_id')
        print (actions.shape)
        actions = pd.merge(actions, get_action_user_feat14(train_start_date, train_end_date), how='left', on='user_id')
        print (actions.shape)
        actions = pd.merge(actions, get_action_user_feat15(start_days, train_end_date), how='left', on='user_id')
        print (actions.shape)
        actions = pd.merge(actions, get_action_user_feat16(start_days, train_end_date), how='left', on='user_id')
        print (actions.shape)
        actions = pd.merge(actions, get_action_u0513_feat16(start_days, train_end_date), how='left', on='user_id')
        print (actions.shape)
        actions = pd.merge(actions, user_features(user,ful_action,sub_action,train_end_date), how='left', on='user_id')
        print (actions.shape)
        actions = pd.merge(actions, get_action_user_feat0515_2_1(train_start_date, train_end_date), how='left', on='user_id')
        print (actions.shape)
        actions = pd.merge(actions, get_action_user_feat0515_2_2(train_start_date, train_end_date), how='left', on='user_id')
        print (actions.shape)
        
        #模型1   和 模型二
#         actions = pd.merge(actions, get_action_u0509_feat_29(train_start_date, train_end_date), how='left', on='user_id')
#         print (actions.shape)
        #模型 二
#         actions = pd.merge(actions, get_action_u0521_feat_32(train_start_date, train_end_date), how='left', on='user_id')
        
        
#         actions = pd.merge(actions, get_action_u0524_feat1(start_days, train_end_date), how='left', on='user_id')
#         print (actions.shape)
        
#         actions = pd.merge(actions, get_action_u0524_feat2(start_days, train_end_date), how='left', on='user_id')
#         print (actions.shape)
#         actions = pd.merge(actions, get_action_u0524_feat3(start_days, train_end_date), how='left', on='user_id')
#         print (actions.shape)
        
        for i in (1, 2, 3, 7, 14, 28):
            actions = pd.merge(actions, get_action_user_feat_six_xingwei(train_start_date, train_end_date, i), how='left',on='user_id')
            actions = pd.merge(actions, deal_user_six_deal(train_start_date, train_end_date, i), how='left',on='user_id')
            actions = pd.merge(actions, get_action_user_feat11(train_start_date, train_end_date, i), how='left',on='user_id')
            actions = pd.merge(actions, get_action_user_feat13(train_start_date, train_end_date, i), how='left',on='user_id')
            actions = pd.merge(actions, get_action_user_feat0509_1_30(train_start_date, train_end_date, i), how='left',on='user_id')
            actions = pd.merge(actions, get_action_user_feat0515_2_3(train_start_date, train_end_date, i), how='left',on='user_id')
            actions = pd.merge(actions, get_action_feat(train_start_date, train_end_date,i), how='left', on='user_id')
            actions = pd.merge(actions, get_action_user_feat0515_2_4(train_start_date, train_end_date,i), how='left', on='user_id')
            actions = pd.merge(actions, get_action_u0515_feat5(train_start_date, train_end_date,i), how='left', on='user_id')
            #模型1   和 模型二
#             actions = pd.merge(actions, get_action_u0509_feat_28(train_start_date, train_end_date,i), how='left', on='user_id')
            if(i<=10):
                actions = pd.merge(actions,get_action_user_feat0509_1_31(train_start_date, train_end_date,i), how='left', on='user_id')
            #模型 二
#             actions = pd.merge(actions, get_action_u0521_feat_31(train_start_date, train_end_date,i), how='left', on='user_id')
#             actions = pd.merge(actions, get_action_user_feat7_0522_huachuang(train_start_date, train_end_date,i), how='left', on='user_id')
        print(actions.shape)
        print(actions.shape)

        actions = actions.fillna(0)
#         user_id = actions[['user_id']]
#         del actions['user_id']
#         actions = actions.fillna(0)
#         actions=actions.replace(np.inf,0)
# #         print(actions.head())
#         columns = actions.columns

#         min_max_scale = preprocessing.MinMaxScaler()
#         actions=actions.replace(np.inf,0)
#         actions = min_max_scale.fit_transform(actions.values)
#         actions = pd.concat([user_id, pd.DataFrame(actions,columns = columns)], axis=1)
#         actions.to_csv(dump_path,index=False)
    return actions


# 训练集
def make_train_set(train_start_date, train_end_date, test_start_date, test_end_date,user,ful_action,sub_action):
    dump_path = './cache/bu0525model_0_u_train_set_%s_%s_%s_%s.csv' % (train_start_date, train_end_date, test_start_date, test_end_date)
    if os.path.exists(dump_path):
        actions = pd.read_csv(dump_path)
    else:

        start_days=str(pd.to_datetime(train_end_date)-timedelta(days=30)).split(' ')[0]
        actions_1 = get_actions(start_days, train_end_date)
        actions=actions_1[actions_1['cate']==8][['user_id']].drop_duplicates(['user_id'])
#         buy_actions = actions_1[(actions_1['type']==4)&(actions_1['cate']==8)][['user_id']].drop_duplicates()
#         actions = actions[actions['user_id'].isin(buy_actions['user_id'])==False]
        
        
        
#         print (actions.shape)
        
#         start_days=str(pd.to_datetime(train_end_date)-timedelta(days=30)).split(' ')[0]
#         actions_1 = get_actions(start_days, train_end_date)
#         actions_1 = actions_1[(actions_1['type']==2)|(actions_1['type']==4)|(actions_1['type']==5)]
#         actions_1=actions_1[actions_1['cate']==8][['user_id']].drop_duplicates(['user_id'])
#         actions = pd.concat([actions,actions_1]).drop_duplicates(['user_id'])
        print (actions.shape)
#         start_days = train_start_date
        start_days = "2016-02-01"
#         actions = pd.merge(actions,get_basic_user_feat() , how='left', on='user_id')
        print(actions.shape)
        
#         actions = pd.merge(actions, get_action_user_feat1(start_days, train_end_date), how='left', on='user_id')
#         print(actions.shape)
        actions = pd.merge(actions, get_action_user_feat2(start_days, train_end_date), how='left', on='user_id')
        print(actions.shape)
        actions = pd.merge(actions, get_action_user_feat5(start_days, train_end_date), how='left', on='user_id')
        print(actions.shape)
        actions = pd.merge(actions, get_action_user_feat6(start_days, train_end_date), how='left', on='user_id')
        print(actions.shape)
        actions = pd.merge(actions, get_action_user_feat6_six(start_days, train_end_date), how='left', on='user_id')
        print(actions.shape)
        actions = pd.merge(actions, get_action_user_feat7(start_days, train_end_date), how='left', on='user_id')
        print(actions.shape)
        actions = pd.merge(actions, get_action_user_feat8(start_days, train_end_date), how='left', on='user_id')
        print (actions.shape)
        actions = pd.merge(actions, get_action_user_feat8_2(start_days, train_end_date), how='left', on='user_id')
        print (actions.shape)
        actions = pd.merge(actions, get_action_user_feat9(start_days, train_end_date), how='left', on='user_id')
        print (actions.shape)
        actions = pd.merge(actions, get_action_user_feat10(start_days, train_end_date), how='left', on='user_id')
        print (actions.shape)
        actions = pd.merge(actions, get_action_user_feat12(train_start_date, train_end_date), how='left', on='user_id')
        print (actions.shape)
        actions = pd.merge(actions, get_action_user_feat14(train_start_date, train_end_date), how='left', on='user_id')
        print (actions.shape)
        actions = pd.merge(actions, get_action_user_feat15(start_days, train_end_date), how='left', on='user_id')
        print (actions.shape)
        actions = pd.merge(actions, get_action_user_feat16(start_days, train_end_date), how='left', on='user_id')
        print (actions.shape)
        actions = pd.merge(actions, get_action_u0513_feat16(start_days, train_end_date), how='left', on='user_id')
        print (actions.shape)
        actions = pd.merge(actions, user_features(user,ful_action,sub_action,train_end_date), how='left', on='user_id')
        print (actions.shape)
        
        actions = pd.merge(actions, get_action_user_feat0515_2_1(train_start_date, train_end_date), how='left', on='user_id')
        print (actions.shape)
        actions = pd.merge(actions, get_action_user_feat0515_2_2(train_start_date, train_end_date), how='left', on='user_id')
        print (actions.shape)
       
    #     actions = pd.merge(actions, get_action_u0509_feat_29(train_start_date, train_end_date), how='left', on='user_id')
#         actions = pd.merge(actions, get_action_u0521_feat_32(train_start_date, train_end_date), how='left', on='user_id')

#         actions = pd.merge(actions, get_action_u0524_feat1(start_days, train_end_date), how='left', on='user_id')
#         print (actions.shape)
        
#         actions = pd.merge(actions, get_action_u0524_feat2(start_days, train_end_date), how='left', on='user_id')
#         print (actions.shape)
#         actions = pd.merge(actions, get_action_u0524_feat3(start_days, train_end_date), how='left', on='user_id')
#         print (actions.shape)
        print (actions.shape)
        for i in (1, 2, 3,7, 14, 28):
            actions = pd.merge(actions, get_action_user_feat_six_xingwei(train_start_date, train_end_date, i), how='left',on='user_id')
            actions = pd.merge(actions, deal_user_six_deal(train_start_date, train_end_date, i), how='left',on='user_id')
            actions = pd.merge(actions, get_action_user_feat11(train_start_date, train_end_date, i), how='left',on='user_id')
            actions = pd.merge(actions, get_action_user_feat13(train_start_date, train_end_date, i), how='left',on='user_id')
            actions = pd.merge(actions, get_action_user_feat0509_1_30(train_start_date, train_end_date, i), how='left',on='user_id')
            actions = pd.merge(actions, get_action_user_feat0515_2_3(train_start_date, train_end_date, i), how='left',on='user_id')
            actions = pd.merge(actions, get_action_feat(train_start_date, train_end_date,i), how='left', on='user_id')
            actions = pd.merge(actions, get_action_user_feat0515_2_4(train_start_date, train_end_date,i), how='left', on='user_id')
            actions = pd.merge(actions, get_action_u0515_feat5(train_start_date, train_end_date,i), how='left', on='user_id')
#             actions = pd.merge(actions, get_action_u0509_feat_28(train_start_date, train_end_date,i), how='left', on='user_id')
            if(i<=10):
                actions = pd.merge(actions,get_action_user_feat0509_1_31(train_start_date, train_end_date,i), how='left', on='user_id')
#             actions = pd.merge(actions, get_action_u0521_feat_31(train_start_date, train_end_date,i), how='left', on='user_id')
        
#             actions = pd.merge(actions, get_action_user_feat7_0522_huachuang(train_start_date, train_end_date,i), how='left', on='user_id')
        print(actions.shape)
        actions = pd.merge(actions, get_user_labels(test_start_date, test_end_date), how='left', on='user_id')
        
        actions = actions.fillna(0)
        print(actions.shape)
#         user_id = actions[['user_id']]
#         del actions['user_id']
#         actions = actions.fillna(0)
#         actions=actions.replace(np.inf,0)
# #         print(actions.head())
#         columns = actions.columns

#         min_max_scale = preprocessing.MinMaxScaler()
#         actions=actions.replace(np.inf,0)
#         actions = min_max_scale.fit_transform(actions.values)
#         actions = pd.concat([user_id, pd.DataFrame(actions,columns = columns)], axis=1)
#         actions.to_csv(dump_path,index=False)
    return  actions

print("U model 0 finish  part_3")






###########################################################################################


# In[ ]:




# In[ ]:




# In[ ]:




# In[11]:

#!/usr/bin/python

import numpy as np
import xgboost as xgb
# from user_feat import *
from sklearn.model_selection import train_test_split


train_start_date = '2016-03-10'
train_end_date = '2016-04-11'
test_start_date = '2016-04-11'
test_end_date = '2016-04-16'


sub_start_date = '2016-03-15'
sub_end_date = '2016-04-16'

#训练数据集
actions = make_train_set(train_start_date, train_end_date, test_start_date, test_end_date,user,ful_action,sub_action)
# print(np.isinf(actions))
# print(np.isnan(actions))



feature_name = actions.columns.values
#排序特征
# for index in feature_name[1:-1]:
#     actions["r"+index]=actions[index].rank(method='max')/actions.shape[0]

print(actions.shape)
actions_pos = actions[actions['label']==1]
actions_neg =  actions[actions['label']==0]


# actions_pos= pd.concat([actions_pos,actions_pos])
# actions_pos= pd.concat([actions_pos,actions_pos])
# actions_pos= pd.concat([actions_pos,actions_pos])
# actions_pos= pd.concat([actions_pos,actions_pos])
# actions=pd.concat([actions_pos,actions_neg])
print("+++++++++++++++++++++++")

label_value = actions['label'].value_counts()
print(label_value)
print ('训练集正负样本数分别为：正样本数为'+str(label_value[1])+'负样本数为:'+str(label_value[0])+
       '正负样本比例为:'+str(1.0*label_value[1]/label_value[0]))

train,test=train_test_split(actions.values,test_size=0.2,random_state=0)
train=pd.DataFrame(train,columns=actions.columns)
test=pd.DataFrame(test,columns=actions.columns)

X_train=train.drop(['user_id','label'],axis=1)
X_test=test.drop(['user_id','label'],axis=1)
y_train=train[['label']]
y_test=test[['label']]
train_index=train[['user_id']].copy()
test_index=test[['user_id']].copy()



#测试数据集
sub_test_data = make_test_set(sub_start_date, sub_end_date,user,ful_action,sub_action)

feature_name = sub_test_data.columns.values


sub_trainning_data=sub_test_data.drop(['user_id'],axis=1)
sub_user_index=sub_test_data[['user_id']].copy()    
########################################################################
print("U model 0 finish  part_3")


# In[ ]:


print ('==========>>>train xgboost model ....')

dtrain = xgb.DMatrix(X_train,label=y_train)
dtest = xgb.DMatrix(X_test,label=y_test)
param = {'learning_rate' : 0.1,
        'n_estimators': 1000,
        'max_depth': 3,
        'min_child_weight': 5,
        'gamma': 0,
        'subsample': 1.0,
        'colsample_bytree': 0.8,
        'eta': 0.05,
        'silent': 1,
        'objective':
        'binary:logistic',
        'scale_pos_weight':1}


num_round =120
plst = list(param.items())
plst += [('eval_metric', 'logloss')]
evallist = [(dtest, 'eval'), (dtrain, 'train')]
bst=xgb.train(plst,dtrain,num_round,evallist,early_stopping_rounds=10)





# print ('==========>>>print feature importance')
# score=bst.get_fscore()
# f_id = pd.DataFrame(list(score.keys()))
# f_score=pd.DataFrame(list(score.values()))
# fscore=pd.concat([f_id,f_score],axis=1)
# fscore.columns=['f_id','f_score']
# fscore.sort_values(by=['f_score'],ascending=[0],inplace=True)
# fscore.to_csv('./sub/u_feat_importace_u_model_0.csv',index=False)

# ============================================>>>>
print ('==========>>>predict test data label')


sub_trainning_data_1 = xgb.DMatrix(sub_trainning_data)
y = bst.predict(sub_trainning_data_1)
pred = sub_user_index
sub_user_index['label'] = y

# print(sub_user_index.head())

pred=sub_user_index
#pred.sort_values(by=['user_id','label'],ascending=[0,0],inplace=True)
pred=pred.sort_values(by=['user_id','label'],ascending=[0,0])
pred = pred.groupby('user_id').first().reset_index()
result=pred.sort_values(by=['label'],ascending=[0])
result['user_id']=result['user_id'].astype('int')


name=str(datetime.now()).replace(':','-').split('.')[0]
result.to_csv('./sub/Umodel_0.csv',index=False,index_label=False )
print("U model 0 finish  part_4")


# In[ ]:



