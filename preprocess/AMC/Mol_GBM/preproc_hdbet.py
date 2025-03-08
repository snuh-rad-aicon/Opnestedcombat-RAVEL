from multiprocessing import Pool
import os
from glob import glob
import pandas as pd
import json
from tqdm import tqdm
import ants
import nibabel as nib
import shutil


target_resolution = (1.0, 1.0, 1.0)
target_shape = [192, 224, 192]

# set visual gpu
os.environ['CUDA_VISIBLE_DEVICES'] = '3'


def registation_mri(patient_dir):
    patient_id, study_date = patient_dir.split('/')[-2:]
    t1_path = os.path.join(patient_dir, f't1.nii.gz')
    t1ce_path = os.path.join(patient_dir, f't1ce.nii.gz')
    t2_path = os.path.join(patient_dir, f't2.nii.gz')
    flair_path = os.path.join(patient_dir, f'flair.nii.gz')
    
    t1_bet_path = os.path.join(patient_dir, f't1_bet.nii.gz')
    t1ce_bet_path = os.path.join(patient_dir, f't1ce_bet.nii.gz')
    t2_bet_path = os.path.join(patient_dir, f't2_bet.nii.gz')
    flair_bet_path = os.path.join(patient_dir, f'flair_bet.nii.gz')
    brain_mask_path = os.path.join(patient_dir, f't1_bet_mask.nii.gz')

    target_dir = os.path.join(target_root, patient_id, study_date)
    os.makedirs(target_dir, exist_ok=True)
    shutil.copy(t1_path, os.path.join(target_dir, 't1.nii.gz'))
    os.system(f'robustfov -i {os.path.join(target_dir, "t1.nii.gz")} -r {os.path.join(target_dir, "t1.nii.gz")}')
    
    nifti_t1 = ants.image_read(os.path.join(target_dir, 't1.nii.gz'), reorient='RPI')
    nifti_t1ce = ants.image_read(t1ce_path, reorient='RPI')
    nifti_t2 = ants.image_read(t2_path, reorient='RPI')
    nifti_flair = ants.image_read(flair_path, reorient='RPI')
    
    nifti_t1ce_bet = ants.image_read(t1ce_bet_path, reorient='RPI')
    nifti_t2_bet = ants.image_read(t2_bet_path, reorient='RPI')
    nifti_flair_bet = ants.image_read(flair_bet_path, reorient='RPI')
    nifti_brain_mask = ants.image_read(brain_mask_path, reorient='RPI')
    
    nifti_t1 = ants.resample_image(nifti_t1, target_resolution, use_voxels=False, interp_type=3)
    nifti_t1_shape = nifti_t1.shape
    slice_index = [[(nifti_t1_shape[i] - target_shape[i]) // 2, (nifti_t1_shape[i] + target_shape[i]) // 2] if nifti_t1_shape[i] > target_shape[i] else [0, nifti_t1_shape[i]] for i in range(3)]
    nifti_t1 = ants.crop_indices(nifti_t1, [slice_index[0][0], slice_index[1][0], slice_index[2][0]], [slice_index[0][1], slice_index[1][1], slice_index[2][1]])
    if nifti_t1.shape[0] < target_shape[0] or nifti_t1.shape[1] < target_shape[1] or nifti_t1.shape[2] < target_shape[2]:
        size_x = target_shape[0]
        size_y = target_shape[1]
        size_z = target_shape[2]
        nifti_t1 = ants.pad_image(nifti_t1, shape=(size_x, size_y, size_z), value=0)
    
    nifti_brain_mask = ants.resample_image_to_target(nifti_brain_mask, nifti_t1, interp_type=1)
    nifti_t1_bet = ants.mask_image(nifti_t1, nifti_brain_mask)
    
    nifti_t1ce_bet = ants.registration(fixed=nifti_t1_bet, moving=nifti_t1ce_bet, type_of_transform='Rigid')
    nifti_t1ce = ants.apply_transforms(fixed=nifti_t1ce_bet['warpedmovout'], moving=nifti_t1ce, transformlist=nifti_t1ce_bet['fwdtransforms'])
    nifti_t2_bet = ants.registration(fixed=nifti_t1_bet, moving=nifti_t2_bet, type_of_transform='Rigid')
    nifti_t2 = ants.apply_transforms(fixed=nifti_t2_bet['warpedmovout'], moving=nifti_t2, transformlist=nifti_t2_bet['fwdtransforms'])
    nifti_flair_bet = ants.registration(fixed=nifti_t1_bet, moving=nifti_flair_bet, type_of_transform='Rigid')
    nifti_flair = ants.apply_transforms(fixed=nifti_flair_bet['warpedmovout'], moving=nifti_flair, transformlist=nifti_flair_bet['fwdtransforms'])
    
    nifti_t1 = nifti_t1ce.new_image_like(nifti_t1.numpy())
    nifti_brain_mask = nifti_t1ce.new_image_like(nifti_brain_mask.numpy())
    nifti_t1_bet = ants.mask_image(nifti_t1, nifti_brain_mask)
    nifti_t1ce_bet = ants.mask_image(nifti_t1ce, nifti_brain_mask)
    nifti_t2_bet = ants.mask_image(nifti_t2, nifti_brain_mask)
    nifti_flair_bet = ants.mask_image(nifti_flair, nifti_brain_mask)
    
    nifti_t1.to_file(os.path.join(target_dir, 't1.nii.gz'))
    nifti_t1ce.to_file(os.path.join(target_dir, 't1ce.nii.gz'))
    nifti_t2.to_file(os.path.join(target_dir, 't2.nii.gz'))
    nifti_flair.to_file(os.path.join(target_dir, 'flair.nii.gz'))
    
    nifti_t1_bet.to_file(os.path.join(target_dir, 't1_bet.nii.gz'))
    nifti_t1ce_bet.to_file(os.path.join(target_dir, 't1ce_bet.nii.gz'))
    nifti_t2_bet.to_file(os.path.join(target_dir, 't2_bet.nii.gz'))
    nifti_flair_bet.to_file(os.path.join(target_dir, 'flair_bet.nii.gz'))
    nifti_brain_mask.to_file(os.path.join(target_dir, 'brain_mask.nii.gz'))


def segmentation_mri(patient_dir):
    os.system(f'hd_glio_predict -t1 {os.path.join(patient_dir, "t1_bet.nii.gz")} -t1c {os.path.join(patient_dir, "t1ce_bet.nii.gz")} -t2 {os.path.join(patient_dir, "t2_bet.nii.gz")} -flair {os.path.join(patient_dir, "flair_bet.nii.gz")} -o {os.path.join(patient_dir, "tumor_mask.nii.gz")}')
    os.system(f'fslmaths {os.path.join(patient_dir, "tumor_mask.nii.gz")} -mas {os.path.join(patient_dir, "brain_mask.nii.gz")} {os.path.join(patient_dir, "tumor_mask.nii.gz")}')
    os.system(f'fslmaths {os.path.join(patient_dir, "tumor_mask.nii.gz")} -thr 1 -uthr 1 -bin {os.path.join(patient_dir, "NE_tumor_mask.nii.gz")}')
    os.system(f'fslmaths {os.path.join(patient_dir, "tumor_mask.nii.gz")} -thr 2 -uthr 2 -bin {os.path.join(patient_dir, "CE_tumor_mask.nii.gz")}')
    os.system(f'fslmaths {os.path.join(patient_dir, "tumor_mask.nii.gz")} -bin {os.path.join(patient_dir, "WT_tumor_mask.nii.gz")}')
    

if __name__ == '__main__':
    source_root = '/data/jhlee/data/AMC/molGBM/nifti'
    target_root = '/data/aicon/data/AMC/molGBM/processed/nifti'
    

    # patient_dir_list = glob(f'{source_root}/*/*')
    # patient_dir_list = [p for p in patient_dir_list 
    #                     if os.path.exists(os.path.join(p, 't1_bet.nii.gz')) 
    #                     and os.path.exists(os.path.join(p, 't1ce_bet.nii.gz'))
    #                     and os.path.exists(os.path.join(p, 't2_bet.nii.gz'))
    #                     and os.path.exists(os.path.join(p, 'flair_bet.nii.gz'))]
    # # patient_dir_list = patient_dir_list[:1]

    # print(len(patient_dir_list), patient_dir_list[0])
    # with Pool(processes=4) as pool:
    #     with tqdm(total=len(patient_dir_list)) as pbar:
    #         for _ in pool.imap_unordered(registation_mri, patient_dir_list):
    #             pbar.update()
    
    
    patient_dir_list = glob(f'{target_root}/*/*')
    patient_dir_list = [p for p in patient_dir_list 
                        if os.path.exists(os.path.join(p, 't1_bet.nii.gz')) 
                        and os.path.exists(os.path.join(p, 't1ce_bet.nii.gz'))
                        and os.path.exists(os.path.join(p, 't2_bet.nii.gz'))
                        and os.path.exists(os.path.join(p, 'flair_bet.nii.gz'))]
    # patient_dir_list = [p for p in patient_dir_list if not os.path.exists(f'{p}/WT_tumor_mask.nii.gz')]
    # patient_dir_list = patient_dir_list[:1]
    
    print(len(patient_dir_list), patient_dir_list[0])
    with Pool(processes=1) as pool:
        with tqdm(total=len(patient_dir_list)) as pbar:
            for _ in pool.imap_unordered(segmentation_mri, patient_dir_list):
                pbar.update()
    

    # patient_dir_list = glob(f'{target_root}/*/*')

    # print(len(patient_dir_list), patient_dir_list[0])
    # with Pool(processes=4) as pool:
    #     with tqdm(total=len(patient_dir_list)) as pbar:
    #         for _ in pool.imap_unordered(temp_func, patient_dir_list):
    #             pbar.update()