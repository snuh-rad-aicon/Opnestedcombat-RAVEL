import pandas as pd
import neuroCombat as nC
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from scipy.stats import ranksums, ttest_ind, ttest_rel, ks_2samp
import os
import numpy as np
from itertools import permutations
import OPNestedComBat as nested
from sklearn.decomposition import PCA

def categorize_manufacturer(name):
    if "SIEMENS" in name.upper():
        return "SIEMENS"
    elif "PHILIPS" in name.upper():
        return "PHILIPS"
    elif "GE" in name.upper():
        return "GE"
    else:
        return "OTHER"

def categorize_scanner(name):
    name_upper = name.upper()
    if "SKYRA" in name_upper or "MAGNETOM" in name_upper:
        return "SKYRA"
    elif "INGENIA" in name_upper:
        return "INGENIA"
    elif "SIGNA" in name_upper:
        return "SIGNA"
    elif "ACHIEVA" in name_upper:
        return "ACHIEVA"
    elif "DISCOVERY" in name_upper or "BIOGRAPH" in name_upper:
        return "DISCOVERY/BIOGRAPH"
    elif "AVANTO" in name_upper or "ESSENZA" in name_upper:
        return "AVANTO/ESSENZA"
    else:
        return "OTHER"

# Loading in features
filepath = "/mnt/hdd1/shan/GBM/combat/mol_his/ce_modified"
filepath2 = '/mnt/hdd1/shan/GBM/combat/mol_his/ce_modified/ce_modified_adc_combat_m+h/'
if not os.path.exists(filepath2):
    os.makedirs(filepath2)

# Loading in batch effects
batch_df = pd.read_csv('/mnt/hdd1/shan/GBM/combat/mol_his/ce_modified/final_batch_effect_merged_mag_after2.csv')
batch_list = ['Manufacturer','Hospital_Group']
# Loading in clinical covariates
covars_df = pd.read_csv('/mnt/hdd1/shan/GBM/combat/mol_his/ce_modified/adc_category_with_days.csv')
categorical_cols = ['Sex', 'WHO_Grade']
continuous_cols = ['days']
# CAPTK data
data_df = pd.read_csv('/mnt/hdd1/shan/GBM/combat/mol_his/ce_modified/adc_merged_radiomics_features.csv')

selected_columns = ["Patient ID", "original_firstorder_10Percentile", 
                    "original_firstorder_90Percentile", "original_firstorder_Mean", 
                    "original_firstorder_Median"]
data_df = data_df[selected_columns]
data_df = data_df.dropna().reset_index(drop=True)  # 결측치 제거
batch_df["clc"] = batch_df["clc"].astype(str).str.strip()
covars_df['Patient ID'] = covars_df['Patient ID'].astype(str).str.strip()
data_df['Patient ID'] = data_df['Patient ID'].astype(str).str.strip()
# 데이터 병합 및 정리
data_df = data_df.rename(columns={"Patient ID": "Case"})
data_df = data_df.merge(batch_df['clc'], left_on='Case', right_on='clc')
# data_df.to_csv('/root/workstation/data_df.csv')

# data_df.to_csv('/mnt/hdd1/shan/GBM/combat/data_df_1212.csv')
dat = data_df.iloc[:, 1:-1]
dat.to_csv(filepath2+'adc_dat.csv')
dat = dat.T.apply(pd.to_numeric)
caseno = data_df['Case'].str.upper() 


batch_df = data_df[['Case']].merge(batch_df, left_on='Case', right_on='clc')
# batch_df.to_csv(filepath2+'adc_batch_df.csv')
covars_df = data_df[['Case']].merge(covars_df, left_on='Case', right_on='Patient ID')
# covars_df.to_csv(filepath2+'adc_covars_df.csv')
covars_string = pd.DataFrame()
covars_string[categorical_cols] = covars_df[categorical_cols].copy()
# print(covars_string)
covars_string[batch_list] = batch_df[batch_list].copy()
# print(covars_string)
covars_quant = covars_df[continuous_cols]

# covars_string.to_csv(filepath2+'adc_covars_string.csv')

# Encoding categorical variables
covars_cat = pd.DataFrame()
for col in covars_string:
    if col == "Manufacturer":
        covars_string[col] = covars_string[col].apply(categorize_manufacturer)
    elif col == "Scanner Model":
        covars_string[col] = covars_string[col].apply(categorize_scanner)
    le = LabelEncoder()
    covars_cat[col] = le.fit_transform(covars_string[col])

covars = pd.concat([covars_cat, covars_quant], axis=1)
# print(covars)
# 1️⃣ 변동이 없는(즉, 모든 값이 동일한) 컬럼 제거
dat = dat.loc[~(dat.nunique(axis=1) == 1)]
# covars = covars.loc[:, covars.nunique() > 1]

# 2️⃣ `covars` 내 batch_list와 categorical_cols 간 중복 변수 제거
batch_list = [col for col in batch_list if col not in categorical_cols]

# 3️⃣ `covars` 내 상관계수 0.95 이상인 컬럼 제거
corr_matrix = covars.corr().abs()
high_corr_cols = set()
for i in range(len(corr_matrix.columns)):
    for j in range(i):
        if corr_matrix.iloc[i, j] > 0.95:
            col_name = corr_matrix.columns[i]
            high_corr_cols.add(col_name)
covars = covars.drop(columns=high_corr_cols)

# 4️⃣ `dat`에서 모든 값이 0이거나 변동이 매우 낮은 피처 제거
low_variance_features = dat.var() < 1e-8
dat = dat.loc[:, ~low_variance_features]

# 5️⃣ GMM Split
gmm_df = nested.GMMSplit(dat, caseno, filepath2)
gmm_df_merge = covars_df.merge(gmm_df, right_on='Patient', left_on='Patient ID')
covars['GMM'] = gmm_df_merge['Grouping']

# 6️⃣ OPNestedComBat 실행
categorical_cols.append('GMM')
covars.to_csv(filepath2+'adc_covars.csv')
# dat.to_csv('/mnt/hdd1/shan/GBM/combat/amc_snuh_non_gbm/wt_modified/adc_dat_dat.csv')
# dat.to_csv('/mnt/hdd1/shan/GBM/combat/Our_NestedComBat_dat.csv')
# covars.to_csv('/mnt/hdd1/shan/GBM/combat/amc_snuh_non_gbm/wt_modified/adc_covars.csv')

output_df = nested.OPNestedComBat(
    dat, covars, batch_list, filepath2, categorical_cols=categorical_cols, continuous_cols=continuous_cols
)

# 결과 저장
write_df = pd.concat([caseno, output_df], axis=1)
write_df.to_csv(filepath2+'SNUH_AMC_sev_adc_ce_modified_combat_after.csv')

# AD test p-values 계산 및 시각화
nested.feature_ad(dat.T, output_df, covars, batch_list, filepath2)
nested.feature_histograms(dat.T, output_df, covars, batch_list, filepath2)
