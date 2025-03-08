from multiprocessing import Pool
import os
from glob import glob
import pandas as pd
import json
from tqdm import tqdm
import ants
import nibabel as nib
import shutil



def moco_perfusion(data_path):
    if os.path.exists(data_path.replace('.nii.gz', '_moco.nii.gz')):
        return
    
    nifti_mri = ants.image_read(data_path)
    if len(nifti_mri.shape) != 4:
        return
    
    nifti_mri_fix = ants.slice_image(nifti_mri, axis=3, idx=0)
    nifti_mri_fix = nifti_mri_fix.reorient_image2('RPI')
    
    nifti_mri_merge = list()
    for i in range(nifti_mri.shape[-1]):
        nifti_mri_mov = ants.slice_image(nifti_mri, axis=3, idx=i)
        nifti_mri_mov = nifti_mri_mov.reorient_image2('RPI')
        nifti_mri_mov = ants.registration(fixed=nifti_mri_fix, moving=nifti_mri_mov, type_of_transform='Rigid')
        nifti_mri_merge.append(nifti_mri_mov['warpedmovout'])
        
    nifti_mri_merge = ants.list_to_ndimage(nifti_mri, nifti_mri_merge)
    nifti_mri_merge.to_file(data_path.replace('.nii.gz', '_moco.nii.gz'))

if __name__ == '__main__':
    target_root = '/data/jhlee/data/AMC/molGBM/nifti'

    patient_dir_list = glob(f'{target_root}/*/*/dsc.nii.gz') + glob(f'{target_root}/*/*/dce.nii.gz')
    # patient_dir_list = patient_dir_list[:1]

    print(len(patient_dir_list), patient_dir_list[0])
    with Pool(processes=1) as pool:
        list(tqdm(pool.imap(moco_perfusion, patient_dir_list), total=len(patient_dir_list)))
    print('Done: moco_perfusion')