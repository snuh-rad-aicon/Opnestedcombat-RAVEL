from multiprocessing import Pool
import os
from glob import glob
import pandas as pd
import json
from tqdm import tqdm
import ants
import nibabel as nib
import shutil
import subprocess


target_resolution = (1.0, 1.0, 1.0)
target_shape = [192, 224, 192]
# target_resolution = (2.0, 2.0, 4.0)
# target_mni_shape = [96, 112, 44]
# target_shape = [96, 112, 32]



def reorientation_parameter_map(patient_dir):
    patient_id, study_date = patient_dir.split('/')[-2:]
    
    dsc_nordic_1_dir = os.path.join(patient_dir, 'dsc_nordic_1')
    if os.path.exists(dsc_nordic_1_dir):
        nordic_files = glob(f'{dsc_nordic_1_dir}/*')
        for nordic_file in nordic_files:
            nifti_nordic = ants.image_read(nordic_file, reorient='RPI')
            nifti_nordic.to_file(f'{nordic_file}')
    
    dce_nordic_1_dir = os.path.join(patient_dir, 'dce_nordic_1')
    if os.path.exists(dce_nordic_1_dir):
        nordic_files = glob(f'{dce_nordic_1_dir}/*')
        for nordic_file in nordic_files:
            nifti_nordic = ants.image_read(nordic_file, reorient='RPI')
            nifti_nordic.to_file(f'{nordic_file}')
        
    dce_nordic_2_dir = os.path.join(patient_dir, 'dce_nordic_2')
    if os.path.exists(dce_nordic_2_dir):
        nordic_files = glob(f'{dce_nordic_2_dir}/*')
        for nordic_file in nordic_files:
            nifti_nordic = ants.image_read(nordic_file, reorient='RPI')
            nifti_nordic.to_file(f'{nordic_file}')
            
    # make done file
    with open(f'{patient_dir}/done_reorientation_parameter_map.txt', 'w') as f:
        f.write('done')


def registration_adc_mri(patient_dir):
    try:
        patient_id, study_date = patient_dir.split('/')[-2:]
        dwi_path = os.path.join(patient_dir, f'dwi.nii.gz')
        dwi_bet_path = os.path.join(patient_dir, f'dwi_bet.nii.gz')
        dwi_mask_path = os.path.join(patient_dir, f'dwi_bet_mask.nii.gz')
        adc_path = os.path.join(patient_dir, f'adc.nii.gz')
        dwi_adc_path = os.path.join(patient_dir, f'dwi_adc.nii.gz')
        os.system(f'fslmerge -t {dwi_adc_path} {dwi_path} {adc_path}')
        
        nifti_dwi_bet = ants.image_read(dwi_bet_path, reorient='RPI')
        nifti_dwi_mask = ants.image_read(dwi_mask_path, reorient='RPI')
        nifti_dwi_adc = ants.image_read(dwi_adc_path, reorient='RPI')

        target_dir = os.path.join(target_root, patient_id, study_date)
        t1_bet_path = os.path.join(target_dir, f't1_bet.nii.gz')
        nifti_t1_bet = ants.image_read(t1_bet_path, reorient='RPI')
        
        nifti_dwi_bet_reg = ants.registration(fixed=nifti_t1_bet, moving=nifti_dwi_bet, type_of_transform='Rigid')
        nifti_dwi_adc_reg = ants.apply_transforms(fixed=nifti_dwi_bet_reg['warpedmovout'], moving=nifti_dwi_adc, transformlist=nifti_dwi_bet_reg['fwdtransforms'], imagetype=3)
        # nifti_dwi_mask_reg = ants.apply_transforms(fixed=nifti_dwi_bet_reg['warpedmovout'], moving=nifti_dwi_mask, transformlist=nifti_dwi_bet_reg['fwdtransforms'], interpolator='nearestNeighbor')

        nifti_dwi_reg = ants.slice_image(nifti_dwi_adc_reg, axis=3, idx=0)
        nifti_adc_reg = ants.slice_image(nifti_dwi_adc_reg, axis=3, idx=1)
        
        nifti_dwi_reg.to_file(f'{target_dir}/dwi.nii.gz')
        nifti_adc_reg.to_file(f'{target_dir}/adc.nii.gz')
        os.remove(dwi_adc_path)
    except Exception as e:
        os.makedirs(f'{"/".join(target_root.split("/")[:-1])}/info/error_patients', exist_ok=True)
        with open(f'{"/".join(target_root.split("/")[:-1])}/info/error_patients/{patient_id}_{study_date}_dwi_adc_registration.txt', 'w') as f:
            f.write(str(e))


def registation_dsc_mri(patient_dir):
    try:
        patient_id, study_date = patient_dir.split('/')[-2:]
        dsc_path = os.path.join(patient_dir, 'dsc_moco.nii.gz')
        
        nifti_dsc = ants.image_read(dsc_path, reorient='RPI')
        if len(nifti_dsc.shape) != 4:
            return
        
        dsc_mask_path = os.path.join(patient_dir, 'dsc_bet_mask.nii.gz')
        brain_mask_dsc = ants.image_read(dsc_mask_path, reorient='RPI')
        dsc_vol = ants.slice_image(nifti_dsc, axis=3, idx=0)
        dsc_vol_bet = ants.mask_image(dsc_vol, brain_mask_dsc)

        target_dir = os.path.join(target_root, patient_id, study_date)
        t1_bet_path = os.path.join(target_dir, f't1_bet.nii.gz')
        nifti_t1_bet = ants.image_read(t1_bet_path, reorient='RPI')
        
        dsc_vol_bet = ants.registration(nifti_t1_bet, dsc_vol_bet, type_of_transform='Rigid')
        nifti_dsc = ants.apply_transforms(dsc_vol_bet['warpedmovout'], nifti_dsc, transformlist=dsc_vol_bet['fwdtransforms'], interpolator='hammingWindowedSinc', imagetype=3)
        nifti_dsc.to_file(f'{target_dir}/dsc.nii.gz')
        dsc_vol_bet['warpedmovout'].to_file(f'{target_dir}/dsc_bet.nii.gz')
        
        dsc_nordic_1_dir = os.path.join(patient_dir, 'dsc_nordic_1')
        target_nordic_1_dir = os.path.join(target_dir, 'dsc_nordic_1')
        if os.path.exists(dsc_nordic_1_dir):
            os.makedirs(target_nordic_1_dir, exist_ok=True)
            dsc_nordic_1_path = os.path.join(patient_dir, 'dsc_nordic_1.nii.gz')
            nordic_files = [f'{patient_dir}/dsc_bet.nii.gz'] + glob(f'{dsc_nordic_1_dir}/*')
            os.system(f'fslmerge -t {dsc_nordic_1_path} {" ".join(nordic_files)}')
            
            nifti_dsc_nordic_1 = ants.image_read(dsc_nordic_1_path, reorient='RPI')
            nifti_dsc_nordic_1 = ants.apply_transforms(dsc_vol_bet['warpedmovout'], nifti_dsc_nordic_1, transformlist=dsc_vol_bet['fwdtransforms'], interpolator='hammingWindowedSinc', imagetype=3)
            
            for idx, nordic_file in enumerate(nordic_files):
                if idx == 0:
                    continue
                nifti_nordic = ants.slice_image(nifti_dsc_nordic_1, axis=3, idx=idx)
                nifti_nordic.to_file(f'{target_nordic_1_dir}/{os.path.basename(nordic_file)}')

            os.remove(dsc_nordic_1_path)
    except Exception as e:
        os.makedirs(f'{"/".join(target_root.split("/")[:-1])}/info/error_patients', exist_ok=True)
        with open(f'{"/".join(target_root.split("/")[:-1])}/info/error_patients/{patient_id}_{study_date}_dsc_registration.txt', 'w') as f:
            f.write(str(e))


def registation_dce_mri(patient_dir):
    try:
        patient_id, study_date = patient_dir.split('/')[-2:]
        dce_path = os.path.join(patient_dir, 'dce_moco.nii.gz')
        
        nifti_dce = ants.image_read(dce_path, reorient='RPI')
        if len(nifti_dce.shape) != 4:
            return

        dce_mask_path = os.path.join(patient_dir, 'dce_bet_mask.nii.gz')
        brain_mask_dce = ants.image_read(dce_mask_path, reorient='RPI')
        dce_vol = ants.slice_image(nifti_dce, axis=3, idx=0)
        dce_vol_bet = ants.mask_image(dce_vol, brain_mask_dce)
        
        target_dir = os.path.join(target_root, patient_id, study_date)
        t1_bet_path = os.path.join(target_dir, f't1_bet.nii.gz')
        nifti_t1_bet = ants.image_read(t1_bet_path, reorient='RPI')
        
        dce_vol_bet = ants.registration(nifti_t1_bet, dce_vol_bet, type_of_transform='Rigid')
        nifti_dce = ants.apply_transforms(dce_vol_bet['warpedmovout'], nifti_dce, transformlist=dce_vol_bet['fwdtransforms'], interpolator='hammingWindowedSinc', imagetype=3)
        nifti_dce.to_file(f'{target_dir}/dce.nii.gz')
        dce_vol_bet['warpedmovout'].to_file(f'{target_dir}/dce_bet.nii.gz')
        
        dce_nordic_1_dir = os.path.join(patient_dir, 'dce_nordic_1')
        target_nordic_1_dir = os.path.join(target_dir, 'dce_nordic_1')
        if os.path.exists(dce_nordic_1_dir):
            os.makedirs(target_nordic_1_dir, exist_ok=True)
            dce_nordic_1_path = os.path.join(patient_dir, 'dce_nordic_1.nii.gz')
            nordic_files = [f'{patient_dir}/dce_bet.nii.gz'] + glob(f'{dce_nordic_1_dir}/*') 
            os.system(f'fslmerge -t {dce_nordic_1_path} {" ".join(nordic_files)}')
            
            nifti_dce_nordic_1 = ants.image_read(dce_nordic_1_path, reorient='RPI')
            nifti_dce_nordic_1 = ants.apply_transforms(dce_vol_bet['warpedmovout'], nifti_dce_nordic_1, transformlist=dce_vol_bet['fwdtransforms'], interpolator='hammingWindowedSinc', imagetype=3)
            
            for idx, nordic_file in enumerate(nordic_files):
                if idx == 0:
                    continue
                nifti_nordic = ants.slice_image(nifti_dce_nordic_1, axis=3, idx=idx)
                nifti_nordic.to_file(f'{target_nordic_1_dir}/{os.path.basename(nordic_file)}')

            os.remove(dce_nordic_1_path)
            
        dce_nordic_2_dir = os.path.join(patient_dir, 'dce_nordic_2')
        target_nordic_2_dir = os.path.join(target_dir, 'dce_nordic_2')
        if os.path.exists(dce_nordic_2_dir):
            os.makedirs(target_nordic_2_dir, exist_ok=True)
            dce_nordic_2_path = os.path.join(patient_dir, 'dce_nordic_2.nii.gz')
            nordic_files = [f'{patient_dir}/dce_bet.nii.gz'] + glob(f'{dce_nordic_2_dir}/*') 
            os.system(f'fslmerge -t {dce_nordic_2_path} {" ".join(nordic_files)}')
            
            nifti_dce_nordic_2 = ants.image_read(dce_nordic_2_path, reorient='RPI')
            nifti_dce_nordic_2 = ants.apply_transforms(dce_vol_bet['warpedmovout'], nifti_dce_nordic_2, transformlist=dce_vol_bet['fwdtransforms'], interpolator='hammingWindowedSinc', imagetype=3)
            
            for idx, nordic_file in enumerate(nordic_files):
                if idx == 0:
                    continue
                nifti_nordic = ants.slice_image(nifti_dce_nordic_2, axis=3, idx=idx)
                nifti_nordic.to_file(f'{target_nordic_2_dir}/{os.path.basename(nordic_file)}')

            os.remove(dce_nordic_2_path)
    except Exception as e:
        os.makedirs(f'{"/".join(target_root.split("/")[:-1])}/info/error_patients', exist_ok=True)
        with open(f'{"/".join(target_root.split("/")[:-1])}/info/error_patients/{patient_id}_{study_date}_dce_registration.txt', 'w') as f:
            f.write(str(e))

def copy2ssd(patient_dir):
    patient_id, study_date = patient_dir.split('/')[-2:]
    target_dir = os.path.join(target_root, patient_id, study_date)
    os.makedirs(target_dir, exist_ok=True)
    
    all_files = glob(f'{patient_dir}/*')
    for file in all_files:
        fname = file.split('/')[-1]
        if fname in ['dsc.nii.gz', 'dce.nii.gz']:
            continue
        
        # # check if the file is a directory
        # if os.path.exists(os.path.join(target_dir, fname)):
        #     continue
        
        if os.path.isdir(file):
            shutil.copytree(file, os.path.join(target_dir, fname))
        else:
            shutil.copy(file, target_dir)
            

ssh_port = 7777
ssh_user = "jhlee"
ssh_host = "147.46.97.169"

def copy2aiconhd(patient_dir):
    patient_id, study_date = patient_dir.split('/')[-2:]
    target_dir = os.path.join(target_root, patient_id, study_date)

    all_files = glob(f'{patient_dir}/*')
    for file in all_files:
        fname = file.split('/')[-1]
        if fname in ['dsc.nii.gz', 'dce.nii.gz']:
            continue
        
        # rsync
        # check if the file is a directory
        if os.path.isdir(file):
            rsync_command = (
                f'rsync --progress -r -e "ssh -p {ssh_port}" '
                f'{file} {ssh_user}@{ssh_host}:{target_dir}'
            )
        else:
            rsync_command = (
                f'rsync --progress -e "ssh -p {ssh_port}" '
                f'{file} {ssh_user}@{ssh_host}:{target_dir}'
            )
        os.system(rsync_command)
        
        # # scp
        # if os.path.isdir(file):
        #     scp_command = (
        #         f'scp -P {ssh_port} -c aes128-ctr -l 8192 -r '
        #         f'{file} {ssh_user}@{ssh_host}:{target_dir}'
        #     )
        # else:
        #     scp_command = (
        #         f'scp -P {ssh_port} -c aes128-ctr -l 8192 '
        #         f'{file} {ssh_user}@{ssh_host}:{target_dir}'
        #     )
        # os.system(scp_command)
        
    
if __name__ == '__main__':
    source_root = '/data/jhlee/data/AMC/nonenhancingGBM/nifti'
    target_root = '/data/aicon/data/AMC/nonenhancingGBM/processed/nifti'

    patient_dir_list = glob(f'{source_root}/*/*')
    patient_dir_list = [p for p in patient_dir_list if not os.path.exists(f'{p}/done_reorientation_parameter_map.txt')]
    # patient_dir_list = patient_dir_list[:1]
    print(f'Reorientation: {len(patient_dir_list)}')
    
    if len(patient_dir_list) != 0:
        with Pool(processes=4) as pool:
            with tqdm(total=len(patient_dir_list)) as pbar:
                for _ in pool.imap_unordered(reorientation_parameter_map, patient_dir_list):
                    pbar.update()


    patient_dir_list = glob(f'{source_root}/*/*')
    patient_dir_list = [p for p in patient_dir_list if os.path.exists(f'{target_root}/{p.split("/")[-2]}/{p.split("/")[-1]}/t1_bet.nii.gz')]
    patient_dir_list = [p for p in patient_dir_list if os.path.exists(os.path.join(p, 'dwi.nii.gz')) and os.path.exists(os.path.join(p, 'adc.nii.gz'))]
    patient_dir_list = [p for p in patient_dir_list if not os.path.exists(os.path.join(target_root, p.split('/')[-2], p.split('/')[-1], 'dwi.nii.gz'))]
    # patient_dir_list = patient_dir_list[:1]
    # patient_dir_list = ['/data/aicon/data/SNUH/GBM/raw/nifti/16841787/20230904']
    
    if len(patient_dir_list) != 0:
        print(len(patient_dir_list), patient_dir_list[0])
        with Pool(processes=4) as pool:
            with tqdm(total=len(patient_dir_list)) as pbar:
                for _ in pool.imap_unordered(registration_adc_mri, patient_dir_list):
                    pbar.update()
    
    patient_dir_list = glob(f'{source_root}/*/*')
    patient_dir_list = [p for p in patient_dir_list if os.path.exists(f'{target_root}/{p.split("/")[-2]}/{p.split("/")[-1]}/t1_bet.nii.gz')]
    patient_dir_list = [p for p in patient_dir_list if os.path.exists(os.path.join(p, 'dsc_nordic_1', 'rCBV.nii.gz'))]
    patient_dir_list = [p for p in patient_dir_list if not os.path.exists(os.path.join(target_root, p.split('/')[-2], p.split('/')[-1], 'dsc_nordic_1', 'rCBV.nii.gz'))]
    
    if len(patient_dir_list) != 0:
        print(len(patient_dir_list), patient_dir_list[0])
        with Pool(processes=1) as pool:
            with tqdm(total=len(patient_dir_list)) as pbar:
                for _ in pool.imap_unordered(registation_dsc_mri, patient_dir_list):
                    pbar.update()
    
    patient_dir_list = glob(f'{source_root}/*/*')
    patient_dir_list = [p for p in patient_dir_list if os.path.exists(f'{target_root}/{p.split("/")[-2]}/{p.split("/")[-1]}/t1_bet.nii.gz')]
    patient_dir_list = [p for p in patient_dir_list if os.path.exists(os.path.join(p, 'dce_nordic_1', 'K12.nii.gz'))]
    patient_dir_list = [p for p in patient_dir_list if not os.path.exists(os.path.join(target_root, p.split('/')[-2], p.split('/')[-1], 'dce_nordic_1', 'K12.nii.gz'))]
    
    if len(patient_dir_list) != 0:
        print(len(patient_dir_list), patient_dir_list[0])
        with Pool(processes=1) as pool:
            with tqdm(total=len(patient_dir_list)) as pbar:
                for _ in pool.imap_unordered(registation_dce_mri, patient_dir_list):
                    pbar.update()

    # target_root = '/data/aicon/data/AMC/nonenhancingGBM/processed'
    # if not os.path.exists(f'{target_root}/info/tumor_volumes.csv'):
    #     patient_dir_list = glob(f'{target_root}/*/*')
    #     tumor_volumes = pd.DataFrame(columns=['patient_id', 'study_date', 'NE', 'CE', 'WT'])
    #     for patient_dir in tqdm(patient_dir_list):
    #         try:
    #             ne = subprocess.run(["fslstats", f"{patient_dir}/NE_tumor_mask.nii.gz", "-V"], capture_output=True).stdout
    #             ne = float(ne.decode('utf-8').split()[1])/1000
    #         except:
    #             ne = 0
    #         try:
    #             ce = subprocess.run(["fslstats", f"{patient_dir}/CE_tumor_mask.nii.gz", "-V"], capture_output=True).stdout
    #             ce = float(ce.decode('utf-8').split()[1])/1000
    #         except:
    #             ce = 0
    #         try:
    #             wt = subprocess.run(["fslstats", f"{patient_dir}/WT_tumor_mask.nii.gz", "-V"], capture_output=True).stdout
    #             wt = float(wt.decode('utf-8').split()[1])/1000
    #         except:
    #             wt = 0

    #         tumor_volumes = pd.concat([tumor_volumes, pd.DataFrame([[patient_dir.split('/')[-2], patient_dir.split('/')[-1], ne, ce, wt]], columns=tumor_volumes.columns)], axis=0)
    #     os.makedirs(f'{"/".join(target_root.split("/")[:-1])}/info', exist_ok=True)
    #     tumor_volumes.to_csv(f'{"/".join(target_root.split("/")[:-1])}/info/tumor_volumes.csv', index=False)
    # target_root = '/data/aicon/data/AMC/nonenhancingGBM/processed/nifti'
    # os.system(f'ssh -p 7777 jhlee@147.46.97.169 "mkdir -p /media/data/aicon/data/AMC/nonenhancingGBM/processed"')
    # os.system(f'scp -P 7777 -r /data/aicon/data/AMC/nonenhancingGBM/processed/info jhlee@147.46.97.169:/media/data/aicon/data/AMC/nonenhancingGBM/processed')
    
    
    # # using rsync
    # source_root = '/data/aicon/data/AMC/nonenhancingGBM/processed/nifti'
    # target_root = '/media/data/aicon/data/AMC/nonenhancingGBM/processed/nifti'
    
    # patient_dir_list = glob(f'{source_root}/*/*')
    # # patient_dir_list = patient_dir_list[:1]
    
    # mkdir_list = [f'{target_root}/{p.split("/")[-2]}/{p.split("/")[-1]}' for p in patient_dir_list]
    # os.system(f'ssh -p {ssh_port} {ssh_user}@{ssh_host} "mkdir -p {" ".join(mkdir_list)}"')
    
    # if len(patient_dir_list) != 0:
    #     print(len(patient_dir_list), patient_dir_list[0])
    #     with Pool(processes=2) as pool:
    #         with tqdm(total=len(patient_dir_list)) as pbar:
    #             for _ in pool.imap_unordered(copy2aiconhd, patient_dir_list):
    #                 pbar.update()

    
    # # source_root = '/data/aicon/data/SNUH/GBM/processed/nifti'
    # # target_root = '/mnt/kschoi/processed/nifti'
    # source_root = '/media/hrson1/processed/nifti'
    # target_root = '/media/data/aicon/data/SNUH/GBM/processed/nifti'
    
    # patient_dir_list = glob(f'{source_root}/*/*')
    # # patient_dir_list = patient_dir_list[:1]
    
    # if len(patient_dir_list) != 0:
    #     print(len(patient_dir_list), patient_dir_list[0])
    #     with Pool(processes=64) as pool:
    #         with tqdm(total=len(patient_dir_list)) as pbar:
    #             for _ in pool.imap_unordered(copy2ssd, patient_dir_list):
    #                 pbar.update()