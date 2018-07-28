
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np

path = './'

def concat_action():
    action1 = pd.read_csv(path+'/data/JData_Action_201602.csv')
    action2 = pd.read_csv(path+'/data/JData_Action_201603.csv')
    action3 = pd.read_csv(path+'/data/JData_Action_201604.csv')
    action = pd.concat([action1,action2,action3]).sort_values(by='time')
    action.to_csv('./data/JData_Action.csv', index=False)

def map_user_reg(x):
    if x<pd.to_datetime('2016/04/16'):
        d = pd.to_datetime('2016/04/16') - x
        d = d.days // 30
    else:
        d = -1
    return d
        
def cate_user_reg(d):
    if d <0:
        d = -1
    elif d>=0 and d<=3:
        d = 1
    elif d>3 and d<=6:
        d = 2
    elif d>6 and d<=12:
        d = 3
    elif d>12 and d<=24:
        d = 4
    elif d>24 and d<=48:
        d = 5
    else:
        d = 6
    return d
    
def user_process():
    user = pd.read_csv(path + '/data/JData_User.csv', encoding='gbk', parse_dates=[4])
    user = user.drop_duplicates('user_id')
    #user = user[user['user_reg_tm']<pd.to_datetime('2016/04/16')]

    user['reg_duration'] = user['user_reg_tm'].apply(map_user_reg)
    user['reg_duration_cate'] = user['reg_duration'].apply(cate_user_reg)
    user['age'] = np.where(user['age']==u'15岁以下', 0,
                           np.where(user['age']==u'16-25岁', 1,
                                    np.where(user['age']==u'26-35岁', 2,
                                             np.where(user['age']==u'36-45岁', 3,
                                                      np.where(user['age']==u'46-55岁', 4,
                                                               np.where(user['age']==u'56岁以上', 5, -1))))))
    user = user.sort_values('user_id')
    user.to_csv( './data1/JData_modified_user.csv', index=False)
user_process()    
def product_process():
    product = pd.read_csv(path + '/data/JData_Product.csv')
    product = product.drop_duplicates('sku_id')
    product.to_csv( './data/JData_Product.csv', index=False)
# product_process() 
def action_process():
    product = pd.read_csv( './data/JData_Product.csv')
    user = pd.read_csv('./data/JData_modified_user.csv')
    action = pd.read_csv( './data/JData_Action.csv', parse_dates=[2], infer_datetime_format=True)
    action['date'] = action['time'].map(lambda x: x.date())
    action = action[action['sku_id'].isin(product['sku_id'])]
    action = action[action['user_id'].isin(user['user_id'])]
    action.to_csv( './data/JData_subset_action.csv', index=False)


concat_action()
user_process()  
product_process()   
action_process()   
print('可以运行U模型')





# In[ ]:



