
from glob import glob
import pandas as pd
import numpy as np
from scipy.stats import *
import statsmodels.api as sm
import pingouin as pg
import statsmodels.formula.api as smf
import ray
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# ray.init()

########
#definitions
########

@ray.remote
def load_orp_osd(file_path):
    try:
        df = pd.read_csv(file_path)
        OSD = df['OSD'].to_numpy()
        ORP = df['ORP'].to_numpy()
        hypno = df['Label'].to_numpy()

        return hypno,OSD,ORP
    except:
        return [],[],[]


################
# Fill in
################
WORKING_DIRECTORY = '/user/your/path/to/OSD_repo'
df_test_all = pd.read_csv(f'{WORKING_DIRECTORY}/OSD_predictions/Test_RESULTS.csv')

#model and split
model_version ='OSD_FINAL'
evaluation_part = 'test'
#read in scaler
params = pd.read_pickle(f'{WORKING_DIRECTORY}/OSD/utils/scaling_osd/scaling_values.pkl')

#load files
pred_path = f'{WORKING_DIRECTORY}/OSD_predictions/{model_version}/'
files = glob(f'{pred_path}*.csv')

################################
# Mixed effect model Sleep depth
################################
futures = [load_orp_osd.remote(file) for file in files]
label_pred_list = ray.get(futures)
labels_list=[]
osd_list = []
orp_list = []
pat_id_list = []
for i,(hypno,osd,orp) in enumerate(label_pred_list):
    hypno_cropped = hypno[:(len(hypno) // 10) * 10]
    osd_cropped = osd[:(len(osd) // 10) * 10]
    orp_cropped = orp[:(len(orp) // 10) * 10]

    labels_list.append(np.mean(hypno_cropped.reshape(-1,10),axis=1))
    osd_list.append(np.mean(((osd_cropped-params['min'])/params['max']).reshape(-1,10),axis=1))
    orp_list.append(np.mean(orp_cropped.reshape(-1,10),axis=1))
    pat_id_list.append((np.mean(orp_cropped.reshape(-1,10),axis=1)*0)+i)
#append to long list
labels = [int(value) for patient_list in labels_list for value in patient_list]
osd = [value for patient_list in osd_list for value in patient_list]
orp = [value/2.5 for patient_list in orp_list for value in patient_list]
pad_id = [int(value) for patient_list in pat_id_list for value in patient_list]


df = pd.DataFrame({
    "sleep_stage": labels,  # Sleep stage labels
    "OSD": osd,  # OSD algorithm scores
    "ORP": orp,   # ORP algorithm scores
    "participant":pad_id
})

# Convert to long format for mixed modeling (retain participant info)
df_long = df.melt(id_vars=["sleep_stage", "participant"], 
                  var_name="algorithm", 
                  value_name="sleep_depth")
df_long = df_long.dropna()
df_long["sleep_stage"] = df_long["sleep_stage"].astype("category")

# Fit Mixed Effects Model
model = smf.mixedlm("sleep_depth ~ algorithm * sleep_stage", 
                     df_long, 
                     groups=df_long["participant"]).fit()

print(model.summary())


# Tukey’s HSD test
tukey = pairwise_tukeyhsd(endog=df_long["sleep_depth"], 
                           groups=df_long["algorithm"], 
                           alpha=0.05)
print(tukey)

################################
# Fit partial_corr model
################################
#AGE 
print('OSD ~ age + sex')
df_test_age = df_test_all[['OSD_N3','OSD_N2','OSD_N1','OSD_R','OSD_NR','age','N1_len','N2_len','N3_len','sex']].copy().dropna()
print(pg.partial_corr(data=df_test_age, x='OSD_N3', y='age', covar=['sex'],method='spearman').round(3))
print(pg.partial_corr(data=df_test_age, x='OSD_N2', y='age', covar=['sex'],method='spearman').round(3))
print(pg.partial_corr(data=df_test_age, x='OSD_N1', y='age', covar=['sex'],method='spearman').round(3))
print(pg.partial_corr(data=df_test_age, x='OSD_R', y='age', covar=['sex'],method='spearman').round(3))
print(pg.partial_corr(data=df_test_age, x='OSD_NR', y='age', covar=['sex','N1_len','N2_len','N3_len'],method='spearman').round(3))

print('ORP ~ age + sex')
df_test_age = df_test_all[['ORP_N3','ORP_N2','ORP_N1','ORP_R','ORP_NR','age','N1_len','N2_len','N3_len','sex']].copy().dropna()
print(pg.partial_corr(data=df_test_age, x='ORP_N3', y='age', covar=['sex'],method='spearman').round(3))
print(pg.partial_corr(data=df_test_age, x='ORP_N2', y='age', covar=['sex'],method='spearman').round(3))
print(pg.partial_corr(data=df_test_age, x='ORP_N1', y='age', covar=['sex'],method='spearman').round(3))
print(pg.partial_corr(data=df_test_age, x='ORP_R', y='age', covar=['sex'],method='spearman').round(3))
print(pg.partial_corr(data=df_test_age, x='ORP_NR', y='age', covar=['sex','N1_len','N2_len','N3_len'],method='spearman').round(3))


#SEX
print('OSD ~ sex + age')
df_test_age = df_test_all[['OSD_N3','OSD_N2','OSD_N1','OSD_R','OSD_NR','age','N1_len','N2_len','N3_len','sex']].copy().dropna()
print(pg.partial_corr(data=df_test_age, x='OSD_N3', y='sex', covar=['age'],method='spearman').round(3))
print(pg.partial_corr(data=df_test_age, x='OSD_N2', y='sex', covar=['age'],method='spearman').round(3))
print(pg.partial_corr(data=df_test_age, x='OSD_N1', y='sex', covar=['age'],method='spearman').round(3))
print(pg.partial_corr(data=df_test_age, x='OSD_R', y='sex', covar=['age'],method='spearman').round(3))
print(pg.partial_corr(data=df_test_age, x='OSD_NR', y='sex', covar=['age','N1_len','N2_len','N3_len'],method='spearman').round(3))

print('ORP ~ sex + age')
df_test_age = df_test_all[['ORP_N3','ORP_N2','ORP_N1','ORP_R','ORP_NR','age','N1_len','N2_len','N3_len','sex']].copy().dropna()
print(pg.partial_corr(data=df_test_age, x='ORP_N3', y='sex', covar=['age'],method='spearman').round(3))
print(pg.partial_corr(data=df_test_age, x='ORP_N2', y='sex', covar=['age'],method='spearman').round(3))
print(pg.partial_corr(data=df_test_age, x='ORP_N1', y='sex', covar=['age'],method='spearman').round(3))
print(pg.partial_corr(data=df_test_age, x='ORP_R', y='sex', covar=['age'],method='spearman').round(3))
print(pg.partial_corr(data=df_test_age, x='ORP_NR', y='sex', covar=['age','N1_len','N2_len','N3_len'],method='spearman').round(3))


#AHI
print('OSD ~ ahi+ sex + age')
df_test_ahi = df_test_all[['OSD_N3','OSD_N2','OSD_N1','OSD_R','OSD_NR','age','ahi','N1_len','N2_len','N3_len','ahi_rem','ahi_nrem','sex']].copy().dropna()
df_test_ahi = df_test_ahi[df_test_ahi['ahi_nrem']<(2*df_test_ahi['ahi_rem'])]
df_test_ahi = df_test_ahi[(2*df_test_ahi['ahi_nrem'])>(df_test_ahi['ahi_rem'])]

print(pg.partial_corr(data=df_test_ahi, x='OSD_N3', y='ahi', covar=['age','sex'],method='spearman').round(3))
print(pg.partial_corr(data=df_test_ahi, x='OSD_N2', y='ahi', covar=['age','sex'],method='spearman').round(3))
print(pg.partial_corr(data=df_test_ahi, x='OSD_N1', y='ahi', covar=['age','sex'],method='spearman').round(3))
print(pg.partial_corr(data=df_test_ahi, x='OSD_R', y='ahi', covar=['age','sex'],method='spearman').round(3))
print(pg.partial_corr(data=df_test_ahi, x='OSD_NR', y='ahi', covar=['age','sex','N1_len','N3_len'],method='spearman').round(3))

print('ORP ~ ahi+ sex + age')
df_test_ahi = df_test_all[['ORP_N3','ORP_N2','ORP_N1','ORP_R','ORP_NR','age','ahi','N1_len','N2_len','N3_len','ahi_rem','ahi_nrem','sex']].copy().dropna()
df_test_ahi = df_test_ahi[df_test_ahi['ahi_nrem']<(2*df_test_ahi['ahi_rem'])]
df_test_ahi = df_test_ahi[(2*df_test_ahi['ahi_nrem'])>(df_test_ahi['ahi_rem'])]

print(pg.partial_corr(data=df_test_ahi, x='ORP_N3', y='ahi', covar=['age','sex'],method='spearman').round(3))
print(pg.partial_corr(data=df_test_ahi, x='ORP_N2', y='ahi', covar=['age','sex'],method='spearman').round(3))
print(pg.partial_corr(data=df_test_ahi, x='ORP_N1', y='ahi', covar=['age','sex'],method='spearman').round(3))
print(pg.partial_corr(data=df_test_ahi, x='ORP_R', y='ahi', covar=['age','sex'],method='spearman').round(3))
print(pg.partial_corr(data=df_test_ahi, x='ORP_NR', y='ahi', covar=['age','sex','N1_len','N3_len'],method='spearman').round(3))

#AHI_nr
print('OSD ~ nr_ahi+ sex + age')
df_test_ahi = df_test_all[['OSD_N3','OSD_N2','OSD_N1','OSD_R','OSD_NR','age','ahi','N1_len','N2_len','N3_len','ahi_rem','ahi_nrem','sex']].copy().dropna()
df_test_ahi = df_test_ahi[df_test_ahi['ahi_nrem']>(2*df_test_ahi['ahi_rem'])]

print(pg.partial_corr(data=df_test_ahi, x='OSD_N3', y='ahi', covar=['age','sex'],method='spearman').round(3))
print(pg.partial_corr(data=df_test_ahi, x='OSD_N2', y='ahi', covar=['age','sex'],method='spearman').round(3))
print(pg.partial_corr(data=df_test_ahi, x='OSD_N1', y='ahi', covar=['age','sex'],method='spearman').round(3))
print(pg.partial_corr(data=df_test_ahi, x='OSD_R', y='ahi', covar=['age','sex'],method='spearman').round(3))
print(pg.partial_corr(data=df_test_ahi, x='OSD_NR', y='ahi', covar=['age','sex','N1_len','N3_len'],method='spearman').round(3))

print('ORP ~ nr_ahi+ sex + age')
df_test_ahi = df_test_all[['ORP_N3','ORP_N2','ORP_N1','ORP_R','ORP_NR','age','ahi','N1_len','N2_len','N3_len','ahi_rem','ahi_nrem','sex']].copy().dropna()
df_test_ahi = df_test_ahi[df_test_ahi['ahi_nrem']>(2*df_test_ahi['ahi_rem'])]

print(pg.partial_corr(data=df_test_ahi, x='ORP_N3', y='ahi', covar=['age','sex'],method='spearman').round(3))
print(pg.partial_corr(data=df_test_ahi, x='ORP_N2', y='ahi', covar=['age','sex'],method='spearman').round(3))
print(pg.partial_corr(data=df_test_ahi, x='ORP_N1', y='ahi', covar=['age','sex'],method='spearman').round(3))
print(pg.partial_corr(data=df_test_ahi, x='ORP_R', y='ahi', covar=['age','sex'],method='spearman').round(3))
print(pg.partial_corr(data=df_test_ahi, x='ORP_NR', y='ahi', covar=['age','sex','N1_len','N3_len'],method='spearman').round(3))

#AHI_r
print('OSD ~ r_ahi+ sex + age')
df_test_ahi = df_test_all[['OSD_N3','OSD_N2','OSD_N1','OSD_R','OSD_NR','age','ahi','N1_len','N2_len','N3_len','ahi_rem','ahi_nrem','sex']].copy().dropna()
df_test_ahi = df_test_ahi[(2*df_test_ahi['ahi_nrem'])<(df_test_ahi['ahi_rem'])]

print(pg.partial_corr(data=df_test_ahi, x='OSD_N3', y='ahi', covar=['age','sex'],method='spearman').round(3))
print(pg.partial_corr(data=df_test_ahi, x='OSD_N2', y='ahi', covar=['age','sex'],method='spearman').round(3))
print(pg.partial_corr(data=df_test_ahi, x='OSD_N1', y='ahi', covar=['age','sex'],method='spearman').round(3))
print(pg.partial_corr(data=df_test_ahi, x='OSD_R', y='ahi', covar=['age','sex'],method='spearman').round(3))
print(pg.partial_corr(data=df_test_ahi, x='OSD_NR', y='ahi', covar=['age','sex','N1_len','N3_len'],method='spearman').round(3))

print('ORP ~ r_ahi+ sex + age')
df_test_ahi = df_test_all[['ORP_N3','ORP_N2','ORP_N1','ORP_R','ORP_NR','age','ahi','N1_len','N2_len','N3_len','ahi_rem','ahi_nrem','sex']].copy().dropna()
df_test_ahi = df_test_ahi[(2*df_test_ahi['ahi_nrem'])<(df_test_ahi['ahi_rem'])]

print(pg.partial_corr(data=df_test_ahi, x='ORP_N3', y='ahi', covar=['age','sex'],method='spearman').round(3))
print(pg.partial_corr(data=df_test_ahi, x='ORP_N2', y='ahi', covar=['age','sex'],method='spearman').round(3))
print(pg.partial_corr(data=df_test_ahi, x='ORP_N1', y='ahi', covar=['age','sex'],method='spearman').round(3))
print(pg.partial_corr(data=df_test_ahi, x='ORP_R', y='ahi', covar=['age','sex'],method='spearman').round(3))
print(pg.partial_corr(data=df_test_ahi, x='ORP_NR', y='ahi', covar=['age','sex','N1_len','N3_len'],method='spearman').round(3))

################################
# OLS
################################
#DEM + AHI
print('OSD ~ DEM + sex + age')
df_test_dem = df_test_all[['ORP_N3','ORP_N2','ORP_N1','ORP_R','ORP_NR','OSD_N3','OSD_N2','OSD_N1','OSD_R','OSD_NR','age','ahi','DEM_cat','N1_len','N2_len','N3_len','ahi_rem','ahi_nrem','sex']].copy().dropna()
df_test_dem = df_test_dem[df_test_dem['ahi']<5]

model_n3 = sm.formula.ols('OSD_N3 ~  DEM_cat + age+ ahi+ sex', data=df_test_dem).fit()
print(model_n3.summary())
model_n2 = sm.formula.ols('OSD_N2 ~  DEM_cat + age+ ahi+ sex', data=df_test_dem).fit()
print(model_n2.summary())
model_n1 = sm.formula.ols('OSD_N1 ~ DEM_cat + age+ ahi + sex', data=df_test_dem).fit()
print(model_n1.summary())
model_r = sm.formula.ols('OSD_R ~ DEM_cat + age+ ahi+ sex', data=df_test_dem).fit()
print(model_r.summary())
model_nr = sm.formula.ols('OSD_NR ~ DEM_cat + age+ ahi + sex +  N1_len + N2_len+ N3_len', data=df_test_dem).fit()
print(model_nr.summary())

print('ORP ~ DEM + sex + age')
model_n3 = sm.formula.ols('ORP_N3 ~  DEM_cat + age+ ahi+ sex', data=df_test_dem).fit()
print(model_n3.summary())
model_n2 = sm.formula.ols('ORP_N2 ~  DEM_cat + age+ ahi+ sex', data=df_test_dem).fit()
print(model_n2.summary())
model_n1 = sm.formula.ols('ORP_N1 ~ DEM_cat + age+ ahi+ sex', data=df_test_dem).fit()
print(model_n1.summary())
model_r = sm.formula.ols('ORP_R ~ DEM_cat + age+ ahi+ sex', data=df_test_dem).fit()
print(model_r.summary())
model_nr = sm.formula.ols('ORP_NR ~ DEM_cat + age+ ahi+ sex +  N1_len + N2_len+ N3_len', data=df_test_dem).fit()
print(model_nr.summary())

print('a')
