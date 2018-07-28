#!/usr/bin/python
#-*-coding:utf-8-*-
import pandas as pd
#3个model的结果取权值，取top700
def u_id():
    df1=pd.read_csv('./sub/Umodel_0.csv')
    df1.columns=['user_id','label1']

    df2=pd.read_csv('./sub/Umodel_1.csv')
    df2.columns=['user_id','label2']

    df3=pd.read_csv('./sub/Umodel_2.csv')
    df3.columns=['user_id','label3']

    df=pd.merge(df1,df2,on='user_id',how='outer')
    df=pd.merge(df,df3,on='user_id',how='outer')
    df['label']=0.3*df['label1']+0.3*df['label2']+0.4*df['label3']
    df.sort_values(by=['label'],ascending=[0],inplace=True)
    df=df[['user_id','label']].reset_index(drop=True)
    df=df[['user_id']]
    return  df[:700]
#usmodel的结果取top325
def us_id():
    df=pd.read_csv('./sub/USModel.csv')
    df=df[['user_id']]
    return df[:325]
#合并user top700 ,us中的user top 325，结果为802
def merge_u_us():
    u = u_id()
    us = us_id()
    df=pd.merge(u,us,on='user_id',how='outer')
    df=df.drop_duplicates('user_id')
    return df

#合并user802与us model['user_id','sku_id'],得到结果
def result():
    u = merge_u_us()
    us=pd.read_csv( './sub/USModel.csv')
    us=us[['user_id','sku_id']]
    us=us.astype('int')
    result=pd.merge(u,us,how='left',on='user_id')
    print ('===========>>>打印输出结果：')
    result=result.fillna(0)
    result=result.astype('int')
    
    result.to_csv('./sub/best_result.csv',index=False)
    return  result

print (result())
