from multiprocessing import Pool, cpu_count
import os
from glob import glob
import pandas as pd
import shutil
from tqdm import tqdm
import pydicom
import ants

dsc_nordic_names = ['rBF', 'rCBV', 'MTT', 'autoDelay Map']
def process_dcm_dir(patient_dir):
    patient_id, study_date = patient_dir.split('/')[-2:]
    target_dir = f'{target_root}/{patient_id}/{study_date}'
    os.makedirs(target_dir, exist_ok=True)
    
    data_files_dirs = glob(f'{patient_dir}/*')
    for data_file_dir in data_files_dirs:
        if os.path.isdir(data_file_dir):
            if 'nordic' in data_file_dir.split('/')[-1]:
                data_dir = f'{target_dir}/{data_file_dir.split("/")[-1]}'
                shutil.copytree(data_file_dir, data_dir, dirs_exist_ok=True)
                nordic_files = glob(f'{data_dir}/*')
                for nordic_file in nordic_files:
                    if '.nii.gz' in nordic_file:
                        continue
                    else:
                        if any([name in nordic_file for name in dsc_nordic_names]):
                            idx = [name in nordic_file for name in dsc_nordic_names].index(True)
                            rename_name = dsc_nordic_names[idx]
                            shutil.move(nordic_file, os.path.join(data_dir, f'{rename_name}.nii'))
                            nordic_file = os.path.join(data_dir, f'{rename_name}.nii')
                        nifti_nordic = ants.image_read(nordic_file)
                        nifti_nordic.to_file(nordic_file.replace('.nii', '.nii.gz'))
                        os.remove(nordic_file)
            else:
                mri_type = data_file_dir.split('/')[-1].lower()
                os.system(f'dcm2niix -m y -z y -f {mri_type} -o {target_dir} {data_file_dir}')
        else:
            shutil.copy(data_file_dir, target_dir)


if __name__ == '__main__':
    source_root = '/data/aicon/data/SNUH/GBM/raw/dicom'
    target_root = '/data/aicon/data/SNUH/GBM/raw/nifti_raw'
    
    patient_dirs = glob(os.path.join(source_root, '*', '*'))
    patient_dirs = [patient_dir for patient_dir in patient_dirs if not os.path.exists(os.path.join(target_root, patient_dir.split('/')[-2], patient_dir.split('/')[-1], 't1.nii.gz'))
                    and not os.path.exists(os.path.join(target_root, patient_dir.split('/')[-2], patient_dir.split('/')[-1], 't1ce.nii.gz'))
                    and not os.path.exists(os.path.join(target_root, patient_dir.split('/')[-2], patient_dir.split('/')[-1], 't2.nii.gz'))
                    and not os.path.exists(os.path.join(target_root, patient_dir.split('/')[-2], patient_dir.split('/')[-1], 'flair.nii.gz'))]
    # patient_dirs = patient_dirs[:1]

    print(len(patient_dirs), patient_dirs[0])
    with Pool(processes=48) as pool:
        with tqdm(total=len(patient_dirs)) as pbar:
            for _ in pool.imap_unordered(process_dcm_dir, patient_dirs):
                pbar.update()
