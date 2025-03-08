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
    
    # check if dwi in target_dir
    nifti_files = glob(f'{target_dir}/*dwi*') + glob(f'{target_dir}/*DWI*')
    for nifti_file in nifti_files:
        os.remove(nifti_file)

    dicom_dirs = glob(f'{source_dir}/*')
    for dicom_dir in dicom_dirs:
        fname = dicom_dir.split('/')[-1]
        os.system(f'dcm2niix -m y -z y -f {fname} -o {target_dir} {dicom_dir}')

if __name__ == '__main__':
    # source_root = '/data/aicon/data/AMC/nonenhancingGlioma/original'
    # target_root = '/data/jhlee/data/AMC/nonenhancingGBM/dicom'
    
    # for patient_dir in tqdm(glob(os.path.join(source_root, '*'))):
    #     patient_id = patient_dir.split('/')[-1]
    #     shutil.copytree(patient_dir, os.path.join(target_root, patient_id))
    
    # source_root = '/data/jhlee/data/AMC/nonenhancingGBM/dicom'
    # target_root = '/data/jhlee/data/AMC/nonenhancingGBM/nifti_raw'
    # patient_dirs = glob(os.path.join(source_root, '*'))
    # patient_list = [patient_dir.split('/')[-1] for patient_dir in patient_dirs]
    # # rename patient_dirs AMC_ to AMC-
    # for i, patient in tqdm(enumerate(patient_list)):
    #     if 'AMC_' in patient:
    #         shutil.move(os.path.join(source_root, patient), os.path.join(source_root, patient.replace('AMC_', 'AMC-')))
    # patient_dirs = glob(os.path.join(source_root, '*'))
    # patient_list = [patient.replace('AMC_', 'AMC-') for patient in patient_list]
    # # patient_dirs = patient_dirs[:1]
    
    source_root = '/data/jhlee/data/AMC/nonenhancingGBM/dicom'
    target_root = '/data/jhlee/data/AMC/nonenhancingGBM/nifti_raw'
    patient_list = glob(os.path.join(source_root, '*'))
    patient_list = [patient.split('/')[-1] for patient in patient_list]
    excel_path = '/data/jhlee/data/AMC/nonenhancingGBM/info/ASAN_nonenhancing_GBM.xlsx'
    excel_df = pd.read_excel(excel_path)
    # remove 2 rows and re set column names
    excel_df = excel_df.iloc[1:]
    patient_list2 = excel_df['New ID'].astype(str) + '_' + excel_df['Exam Date'].astype(str)
    patient_list2 = patient_list2.values.tolist()
    patient_list2 = [f'AMC-{patient}' for patient in patient_list2]
    patient_list = [patient for patient in patient_list2 if patient.split('_')[0] in patient_list]
    
    print(len(patient_list), patient_list[0])
    with Pool(processes=48) as pool:
        with tqdm(total=len(patient_list)) as pbar:
            for _ in pool.imap_unordered(process_dcm_dir, patient_list):
                pbar.update()

