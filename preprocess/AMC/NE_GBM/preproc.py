from multiprocessing import Pool
import os
from glob import glob
import pandas as pd
import json
import numpy as np
from tqdm import tqdm
import ants
import nibabel as nib
import matplotlib.pyplot as plt
import shutil


fname_mapping_dict = {
    'DSC': 'dsc',
    'DCE': 'dce',
    'DWI_1000': 'dwi',
    'DWI_3000': 'dwi_3000',
    'FLAIR': 'flair',
    'T2': 't2',
    'GD': 't1ce',
    't1': 't1',
}

except_patient_list = []


def copy_MRI(data_path):
    patient_id, study_date, fname = data_path.split('/')[-3:]
    target_dir = os.path.join(target_root, patient_id, study_date)
    os.makedirs(target_dir, exist_ok=True)

    new_fname = None
    for key in fname_mapping_dict.keys():
        key_list = key.split('_')
        if [key.lower() in fname.lower() for key in key_list] == [True]*len(key_list):
            new_fname = fname_mapping_dict[key]
            break
    if new_fname is None:
        return

    target_path = os.path.join(target_dir, f'{new_fname}.nii.gz')
    if new_fname in ['dwi', 'dwi_3000']:
        dwi = ants.image_read(data_path)
        if len(dwi.shape) == 4:
            num_time = dwi.shape[3]
            dwi_1 = ants.slice_image(dwi, axis=3, idx=0)
            dwi_2 = ants.slice_image(dwi, axis=3, idx=num_time-1)
            
            median_dwi_1 = np.mean(dwi_1.numpy())
            median_dwi_2 = np.mean(dwi_2.numpy())
            
            if median_dwi_1 > median_dwi_2:
                dwi_b0 = dwi_1
                dwi_b1000 = dwi_2
            else:
                dwi_b0 = dwi_2
                dwi_b1000 = dwi_1
            
            target_mri_b0_path = os.path.join(target_dir, f'{new_fname}_b0.nii.gz')

            dwi_b1000.to_file(target_path)
            os.system(f'fslreorient2std {target_path} {target_path}')
            
            dwi_b0.to_file(target_mri_b0_path)
            os.system(f'fslreorient2std {target_mri_b0_path} {target_mri_b0_path}')

            bval_path = data_path.replace('.nii.gz', '.bval')
            bvec_path = data_path.replace('.nii.gz', '.bvec')
            target_bval_path = target_path.replace('.nii.gz', '.bval')
            target_bvec_path = target_path.replace('.nii.gz', '.bvec')
            shutil.copy(bval_path, target_bval_path)
            shutil.copy(bvec_path, target_bvec_path)
    else:
        os.system(f'fslreorient2std {data_path} {target_path}')

    json_path = data_path.replace('.nii.gz', '.json')
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            json_data = json.load(f)
        with open(target_path.replace('.nii.gz', '.json'), 'w') as f:
            json.dump(json_data, f)


def copy_MRI_map(patient_dir):
    patient_id, study_date = patient_dir.split('/')[-2:]

    adc_path = os.path.join(map_root, patient_id.replace('AMC-', 'AMC_'), 'adc.nii.gz')
    if os.path.exists(adc_path):
        target_path = os.path.join(patient_dir, 'adc.nii.gz')
        os.system(f'fslreorient2std {adc_path} {target_path}')

    dsc_dir = os.path.join(map_root, patient_id.replace('AMC-', 'AMC_'), 'output_DSC')
    if os.path.exists(dsc_dir):
        target_dir = os.path.join(patient_dir, 'dsc_nordic_1')
        os.makedirs(target_dir, exist_ok=True)
        nifti_files = glob(f'{dsc_dir}/*.nii')
        if os.path.exists(os.path.join(target_dir, 'autoDelay Map.nii.gz')):
            os.remove(os.path.join(target_dir, 'autoDelay Map.nii.gz'))
        for nifti_file in nifti_files:
            if 'autoDelay' in nifti_file:
                target_path = os.path.join(target_dir, 'autoDelay_Map.nii.gz')
            elif 'rCBV' in nifti_file:
                target_path = os.path.join(target_dir, 'rCBV.nii.gz')
            elif 'rBF' in nifti_file:
                target_path = os.path.join(target_dir, 'rCBF.nii.gz')
            elif 'MTT' in nifti_file:
                target_path = os.path.join(target_dir, 'rMTT.nii.gz')
            else:
                continue
            nifti_map = ants.image_read(nifti_file, reorient='RPI')
            nifti_map.to_file(target_path)
            
    dce_files = glob(f'{map_root}/{patient_id.replace('AMC-', 'AMC_')}/*.nii')
    for dce_file in dce_files:
        target_dir = os.path.join(patient_dir, 'dce_nordic_1')
        os.makedirs(target_dir, exist_ok=True)
        
        fname = dce_file.split('/')[-1]
        target_path = os.path.join(target_dir, f'{fname.split(".")[0]}.nii.gz')
        nifti_map = ants.image_read(dce_file, reorient='RPI')
        nifti_map.to_file(target_path)


def bet_MRI(mri_path):
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
        return
    

if __name__ == '__main__':
    # source_root = '/data/jhlee/data/AMC/nonenhancingGBM/nifti_raw'
    # target_root = '/data/jhlee/data/AMC/nonenhancingGBM/nifti'
    # os.makedirs(target_root, exist_ok=True)

    # datasets = glob(f'{source_root}/*/*/*.nii.gz')
    
    # print(len(datasets), datasets[0])
    # with Pool(processes=48) as pool:
    #     list(tqdm(pool.imap(copy_MRI, datasets), total=len(datasets)))
    # print('Done: copy_MRI')


    # source_root = '/data/jhlee/data/AMC/nonenhancingGBM/nifti'
    # map_root = '/data/aicon/data/AMC/nonenhancingGlioma/maps_4Ddicoms'
    # datasets = glob(f'{source_root}/*/*')
    
    # print(len(datasets), datasets[0])
    # with Pool(processes=48) as pool:
    #     list(tqdm(pool.imap(copy_MRI_map, datasets), total=len(datasets)))
    # print('Done: copy_MRI_map')


    # target_root = '/data/jhlee/data/AMC/nonenhancingGBM/nifti'
    # datasets = glob(f'{target_root}/*/*/*.nii.gz')
    # datasets = [mri for mri in datasets if '_bet' not in mri.split('/')[-1]]
    # datasets = [mri for mri in datasets if mri.split('/')[-1].split('.nii.gz')[0] not in ['adc', 'dwi_b0', 'dwi_3000', 'dwi_3000_b0', 'swan_p', 'swi_p', 'pcasl', 'dce_moco', 'dsc_moco']]
    # datasets = [mri for mri in datasets if 'bet' not in mri.split('/')[-1] and 'vol' not in mri.split('/')[-1] and 'moco' not in mri.split('/')[-1]]
    # datasets = [mri for mri in datasets if not os.path.exists(mri.replace('.nii.gz', '_bet.nii.gz'))]
    
    # print(len(datasets), datasets[0])
    # with Pool(processes=16) as pool:
    #     with tqdm(total=len(datasets)) as pbar:
    #         for _ in pool.imap_unordered(bet_MRI, datasets):
    #             pbar.update()
    
    
    
    # source_root = '/data/jhlee/data/AMC/nonenhancingGBM/dicom'
    # fname_list = glob(f'{source_root}/*/*')
    # fname_list = [f.split('/')[-1] for f in fname_list]

    # fname_mapping_dict = {
    #     'DSC': 'dsc',
    #     'DCE': 'dce',
    #     'DWI_1000': 'dwi',
    #     'FLAIR': 'flair',
    #     'T2': 't2',
    #     'GD': 't1ce',
    #     't1': 't1',
    # }
    # new_fname_list = list()
    # for fname in fname_list:
    #     fname_list = fname.split('_')
    #     flag = False
    #     for key in fname_mapping_dict.keys():
    #         key_list = key.split('_')
    #         if [key.lower() in fname.lower() for key in key_list] == [True]*len(key_list):
    #             flag = True
    #             break
    #     if not flag:
    #         new_fname_list.append(fname)
    # print(new_fname_list)

    # for fname in new_fname_list:
    #     if 't1tfe' in fname.lower():
    #         print(fname.lower())