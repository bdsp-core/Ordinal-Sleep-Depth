
from tqdm import tqdm
from glob import glob

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import h5py as h5

import os
import pickle 
from scipy.stats import *
import statsmodels.api as sm
import scipy
import pingouin as pg
import pickle 


########
#definitions
########


df_test_all = pd.read_csv('/media/erikjan/Expansion/OSD/python_code/development_RESULTS.csv')



# Fit ANCOVA model
#AGE 
print('OSD ~ age + sex')
df_test_age = df_test_all[['OSD_N3','OSD_N2','OSD_N1','OSD_R','OSD_NR','age','N1_len','N2_len','N3_len','sex']].copy().dropna()
df_test_age = df_test_age[df_test_age['age']>17]
print(pg.partial_corr(data=df_test_age, x='OSD_N3', y='age', covar=['sex'],method='spearman').round(3))
print(pg.partial_corr(data=df_test_age, x='OSD_N2', y='age', covar=['sex'],method='spearman').round(3))
print(pg.partial_corr(data=df_test_age, x='OSD_N1', y='age', covar=['sex'],method='spearman').round(3))
print(pg.partial_corr(data=df_test_age, x='OSD_R', y='age', covar=['sex'],method='spearman').round(3))
print(pg.partial_corr(data=df_test_age, x='OSD_NR', y='age', covar=['sex','N1_len','N2_len','N3_len'],method='spearman').round(3))


#SEX
print('OSD ~ sex + age')
df_test_age = df_test_all[['OSD_N3','OSD_N2','OSD_N1','OSD_R','OSD_NR','age','N1_len','N2_len','N3_len','sex']].copy().dropna()
df_test_age = df_test_age[df_test_age['age']>17]
print(pg.partial_corr(data=df_test_age, x='OSD_N3', y='sex', covar=['age'],method='spearman').round(3))
print(pg.partial_corr(data=df_test_age, x='OSD_N2', y='sex', covar=['age'],method='spearman').round(3))
print(pg.partial_corr(data=df_test_age, x='OSD_N1', y='sex', covar=['age'],method='spearman').round(3))
print(pg.partial_corr(data=df_test_age, x='OSD_R', y='sex', covar=['age'],method='spearman').round(3))
print(pg.partial_corr(data=df_test_age, x='OSD_NR', y='sex', covar=['age','N1_len','N2_len','N3_len'],method='spearman').round(3))


#AHI
print('OSD ~ ahi+ sex + age')
df_test_ahi = df_test_all[['OSD_N3','OSD_N2','OSD_N1','OSD_R','OSD_NR','age','ahi','N1_len','N2_len','N3_len','ahi_rem','ahi_nrem','sex']].copy().dropna()
df_test_ahi = df_test_ahi[df_test_ahi['age']>17]
df_test_ahi = df_test_ahi[df_test_ahi['ahi_nrem']<(2*df_test_ahi['ahi_rem'])]
df_test_ahi = df_test_ahi[(2*df_test_ahi['ahi_nrem'])>(df_test_ahi['ahi_rem'])]

print(pg.partial_corr(data=df_test_ahi, x='OSD_N3', y='ahi', covar=['age','sex'],method='spearman').round(3))
print(pg.partial_corr(data=df_test_ahi, x='OSD_N2', y='ahi', covar=['age','sex'],method='spearman').round(3))
print(pg.partial_corr(data=df_test_ahi, x='OSD_N1', y='ahi', covar=['age','sex'],method='spearman').round(3))
print(pg.partial_corr(data=df_test_ahi, x='OSD_R', y='ahi', covar=['age','sex'],method='spearman').round(3))
print(pg.partial_corr(data=df_test_ahi, x='OSD_NR', y='ahi', covar=['age','sex','N1_len','N2_len','N3_len'],method='spearman').round(3))

#AHI_nr
print('OSD ~ nr_ahi+ sex + age')
df_test_ahi = df_test_all[['OSD_N3','OSD_N2','OSD_N1','OSD_R','OSD_NR','age','ahi','N1_len','N2_len','N3_len','ahi_rem','ahi_nrem','sex']].copy().dropna()
df_test_ahi = df_test_ahi[df_test_ahi['age']>17]
df_test_ahi = df_test_ahi[df_test_ahi['ahi_nrem']>(2*df_test_ahi['ahi_rem'])]

print(pg.partial_corr(data=df_test_ahi, x='OSD_N3', y='ahi', covar=['age','sex'],method='spearman').round(3))
print(pg.partial_corr(data=df_test_ahi, x='OSD_N2', y='ahi', covar=['age','sex'],method='spearman').round(3))
print(pg.partial_corr(data=df_test_ahi, x='OSD_N1', y='ahi', covar=['age','sex'],method='spearman').round(3))
print(pg.partial_corr(data=df_test_ahi, x='OSD_R', y='ahi', covar=['age','sex'],method='spearman').round(3))
print(pg.partial_corr(data=df_test_ahi, x='OSD_NR', y='ahi', covar=['age','sex','N1_len','N2_len','N3_len'],method='spearman').round(3))

#AHI_r
print('OSD ~ r_ahi+ sex + age')
df_test_ahi = df_test_all[['OSD_N3','OSD_N2','OSD_N1','OSD_R','OSD_NR','age','ahi','N1_len','N2_len','N3_len','ahi_rem','ahi_nrem','sex']].copy().dropna()
df_test_ahi = df_test_ahi[df_test_ahi['age']>17]
df_test_ahi = df_test_ahi[(2*df_test_ahi['ahi_nrem'])<(df_test_ahi['ahi_rem'])]

print(pg.partial_corr(data=df_test_ahi, x='OSD_N3', y='ahi', covar=['age','sex'],method='spearman').round(3))
print(pg.partial_corr(data=df_test_ahi, x='OSD_N2', y='ahi', covar=['age','sex'],method='spearman').round(3))
print(pg.partial_corr(data=df_test_ahi, x='OSD_N1', y='ahi', covar=['age','sex'],method='spearman').round(3))
print(pg.partial_corr(data=df_test_ahi, x='OSD_R', y='ahi', covar=['age','sex'],method='spearman').round(3))
print(pg.partial_corr(data=df_test_ahi, x='OSD_NR', y='ahi', covar=['age','sex','N1_len','N2_len','N3_len'],method='spearman').round(3))


print('a')

