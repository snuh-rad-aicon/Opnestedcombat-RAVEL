from multiprocessing import Pool
import os
from glob import glob
import pandas as pd
import json
import numpy as np
import shutil
from tqdm import tqdm
import ants


bet_MRI_names = ['t1', 't1ce', 't2', 'flair', 'dsc', 'dce', 'dwi']

def bet_MRI(patient_dir):
    patient_id, study_date = patient_dir.split('/')[-2:]
    target_dir = f'{target_root}/{patient_id}/{study_date}'
    os.makedirs(target_dir, exist_ok=True)
    
    mri_paths = glob(os.path.join(patient_dir, '*.nii.gz'))
    for mri_path in mri_paths:
        mri_name = mri_path.split('/')[-1].split('.nii.gz')[0]
        
        shutil.copy(mri_path, os.path.join(target_dir, f'{mri_name}.nii.gz'))
        mri_path = os.path.join(target_dir, f'{mri_name}.nii.gz')
        if 'autoDelay Map' in mri_path:
            shutil.move(mri_path, mri_path.replace('autoDelay Map', 'autoDelay_Map'))
            # nifti_mri = ants.image_read(mri_path)
            # nifti_mri.to_file(mri_path.replace('autoDelay Map', 'autoDelay_Map'))
            # os.rename(mri_path, mri_path.replace('autoDelay Map', 'autoDelay_Map'))
        if mri_name not in bet_MRI_names:
            continue
        
        mri_bet_path = mri_path.replace('.nii.gz', '_bet.nii.gz')
        mri_bet_mask_path = mri_path.replace('.nii.gz', '_bet_mask.nii.gz')
        
        try:
            mri_type = mri_path.split('/')[-1].split('.nii.gz')[0]
            if mri_type in ['dsc', 'dce']:
                nifti_mri = ants.image_read(mri_path)
                mri_vol_path = mri_path.replace('.nii.gz', '_vol.nii.gz')
                nifti_mri_vol = ants.slice_image(nifti_mri, axis=3, idx=0)
                nifti_mri_vol.to_file(mri_vol_path)        
                os.system(f'mri_synthstrip -i {mri_vol_path} -o {mri_bet_path} -m {mri_bet_mask_path}')
            else:
                os.system(f'mri_synthstrip -i {mri_path} -o {mri_bet_path} -m {mri_bet_mask_path}')
        except Exception as e:
            print(f'Error: {mri_path} {e}')
            # return
        
    mri_dirs = glob(os.path.join(patient_dir, '*'))
    for mri_dir in mri_dirs:
        if os.path.isdir(mri_dir):
            shutil.copytree(mri_dir, os.path.join(target_dir, mri_dir.split('/')[-1]), dirs_exist_ok=True)


if __name__ == '__main__':
    source_root = '/data/aicon/data/SNUH/GBM/raw/nifti_raw'
    target_root = '/data/aicon/data/SNUH/GBM/raw/nifti'
    
    patient_dirs = glob(os.path.join(source_root, '*', '*'))
    patient_dirs = [patient_dir for patient_dir in patient_dirs if not os.path.exists(os.path.join(target_root, patient_dir.split('/')[-2], patient_dir.split('/')[-1], 't1_bet.nii.gz'))
                    and not os.path.exists(os.path.join(target_root, patient_dir.split('/')[-2], patient_dir.split('/')[-1], 't1ce_bet.nii.gz'))
                    and not os.path.exists(os.path.join(target_root, patient_dir.split('/')[-2], patient_dir.split('/')[-1], 't2_bet.nii.gz'))
                    and not os.path.exists(os.path.join(target_root, patient_dir.split('/')[-2], patient_dir.split('/')[-1], 'flair_bet.nii.gz'))]
    # patient_dirs = patient_dirs[:1]

    print(len(patient_dirs), patient_dirs[0])
    with Pool(processes=48) as pool:
        with tqdm(total=len(patient_dirs)) as pbar:
            for _ in pool.imap_unordered(bet_MRI, patient_dirs):
                pbar.update()
