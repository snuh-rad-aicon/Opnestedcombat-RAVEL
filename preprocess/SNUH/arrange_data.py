from multiprocessing import Pool, cpu_count
import os
from glob import glob
import pandas as pd
import shutil
from tqdm import tqdm
import pydicom
import json


def process_dcm_dir(patient_dir):
    patient_id = patient_dir.split('/')[-1]
    study_date = excel[excel['Patient ID'] == patient_id]['1st MR Exam'].values[0]
    study_date = str(study_date).split('T')[0].replace('-', '')
    target_dir = f'{target_root}/{patient_id}/{study_date}'
    os.makedirs(target_dir, exist_ok=True)
    
    dce_nordic_files = ['K12.nii', 'K21.nii', 'TTP.nii', 'Ve.nii', 'Vp.nii']
    files = glob(f'{patient_dir}/*')
    for file in files:
        if os.path.isdir(file):
            if file.split('/')[-1] == 'output_DSC':
                data_dir = f'{target_dir}/dsc_nordic_1'
            else:
                data_dir = f'{target_dir}/{file.split("/")[-1]}'
            shutil.copytree(file, data_dir, dirs_exist_ok=True)
        elif file.split('/')[-1] in dce_nordic_files:
            dce_nordic_dir = f'{target_dir}/dce_nordic_1'
            os.makedirs(dce_nordic_dir, exist_ok=True)
            shutil.copy(file, dce_nordic_dir)
        else:
            shutil.copy(file, target_dir)


if __name__ == '__main__':
    source_root = '/data/aicon/data/SNUH/GBM/add'
    target_root = '/data/aicon/data/SNUH/GBM/raw/dicom'
    
    patient_dirs = glob(os.path.join(source_root, '*'))
    patient_dirs = [patient_dir for patient_dir in patient_dirs if len(patient_dir.split('/')[-1]) == 8 and patient_dir.split('/')[-1].isdigit()]
    # patient_dirs = patient_dirs[:1]
    
    excel_path = '/data/jhlee/data/SNUH/GBM/info/GBM_list20250103.xlsx'
    excel = pd.read_excel(excel_path)
    excel['Patient ID'] = excel['Patient ID'].astype(str).str.zfill(8)
    # print(patient_dirs)

    print(len(patient_dirs), patient_dirs[0])
    with Pool(processes=48) as pool:
        with tqdm(total=len(patient_dirs)) as pbar:
            for _ in pool.imap_unordered(process_dcm_dir, patient_dirs):
                pbar.update()
