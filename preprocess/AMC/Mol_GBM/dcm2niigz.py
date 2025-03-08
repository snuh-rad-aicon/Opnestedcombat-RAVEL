from multiprocessing import Pool, cpu_count
import os
from glob import glob
import pandas as pd
import shutil
from tqdm import tqdm
import pydicom
import json


def process_dcm_dir(patient_info):
    patient_id, study_date = patient_info.split('_')
    study_date = study_date.split(' ')[0]
    study_date = ''.join(study_date.split('-'))[:8]
    source_dir = f'{source_root}/{patient_id}'
    target_dir = f'{target_root}/{patient_id}/{study_date}'
    os.makedirs(target_dir, exist_ok=True)

    dicom_dirs = glob(f'{source_dir}/*')
    for dicom_dir in dicom_dirs:
        fname = dicom_dir.split('/')[-1]
        os.system(f'dcm2niix -m n -z y -f {fname} -o {target_dir} {dicom_dir}')
        

if __name__ == '__main__':
    source_root = '/data/jhlee/data/AMC/molGBM/dicom'
    target_root = '/data/jhlee/data/AMC/molGBM/nifti_raw'
    patient_dirs = glob(os.path.join(source_root, '*'))
    patient_list = [patient_dir.split('/')[-1] for patient_dir in patient_dirs]
    # patient_dirs = patient_dirs[:1]
    
    excel_path = '/data/jhlee/data/AMC/molGBM/info/ASAN_mol_GBM.xlsx'
    excel_df = pd.read_excel(excel_path)
    # remove 2 rows and re set column names
    excel_df = excel_df.iloc[1:]
    excel_df.columns = excel_df.iloc[0]
    excel_df = excel_df.iloc[2:]
    patient_list2 = excel_df['Anonymized institutional patient ID (e.g. Yonsei-1...)'].astype(str) + '_' + excel_df['Date of pre-op baseline MRI '].astype(str)
    patient_list2 = patient_list2.values.tolist()
    patient_list = [patient for patient in patient_list2 if patient.split('_')[0] in patient_list]

    print(len(patient_list), patient_list[0])
    with Pool(processes=48) as pool:
        with tqdm(total=len(patient_list)) as pbar:
            for _ in pool.imap_unordered(process_dcm_dir, patient_list):
                pbar.update()

