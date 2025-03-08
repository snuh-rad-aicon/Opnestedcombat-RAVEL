# 마지막 사용 20250204

import os
import pandas as pd
import SimpleITK as sitk
from radiomics import featureextractor
from tqdm import tqdm
from glob import glob
from collections import OrderedDict
from multiprocessing import Pool
from functools import partial

# ==========================================
# File Paths and Directories
# ==========================================
classified_file = '/mnt/hdd1/shan/GBM/combat/SNUH_dataset.xlsx'
data_base_dir = '/mnt/hdd1/shan/GBM/Ravel_preprocessed/csf/SNUH'
output_dir = '/mnt/hdd1/shan/GBM/Ravel_preprocessed/features/AMC_SNUH/SNUH/csf'
# mask_dir = '/home/yschoi/VASARI_VS/tumor_seg/brats_mask'

os.makedirs(output_dir, exist_ok=True)

# ==========================================
# Load Patient IDs from Excel File
# ==========================================
classified_df = pd.read_excel(classified_file)

# ==========================================
# Collect MRI File Paths for Each Patient
# ==========================================
patient_image_paths = []

for _, row in classified_df.iterrows():
    patient_id = str(row['Patient.ID'])
    category = row['Category']
    patient_dir = os.path.join(data_base_dir, patient_id)

    if os.path.exists(patient_dir):
        # Find date folders within the patient directory
        date_dirs = [os.path.join(patient_dir, d) for d in os.listdir(patient_dir) if os.path.isdir(os.path.join(patient_dir, d))]

        for date_dir in date_dirs:
            # Prepare the image files dictionary for each date folder
            image_files = {
                'Patient ID': patient_id,
                'Category': category,
                'rCBV_rad': os.path.join(date_dir, 'rCBV_rad.nii.gz'),
                'rCBV_vox': os.path.join(date_dir, 'rCBV_vox.nii.gz'),
                'ADC_rad': os.path.join(date_dir, 'adc_rad.nii.gz'),
                'ADC_vox': os.path.join(date_dir, 'adc_vox.nii.gz'),
                'Ktrans_rad': os.path.join(date_dir, 'Ktrans_rad.nii.gz'),
                'Ktrans_vox': os.path.join(date_dir, 'Ktrans_vox.nii.gz'),
                'Mask': os.path.join(date_dir, 'NE_tumor_mask.nii.gz'),
            }

            # Check if all necessary files exist
            if all(os.path.exists(v) for k, v in image_files.items() if k not in ['Patient ID', 'Category']):
                patient_image_paths.append(image_files)
            else:
                # Print missing files for debugging
                missing_files = [k for k, v in image_files.items() if not os.path.exists(v)]
                print(f"Missing files for {patient_id} at {date_dir}: {missing_files}")

                # Save missing file log
                with open("missing_files_log.txt", "a") as log_file:
                    log_file.write(f"{patient_id} - Missing files: {missing_files}\n")


# ==========================================
# Radiomics Feature Extraction Settings
# ==========================================
# binwidth=25인 경우 4시간, binwidth=15인 경우 33시간 소요
extractor = featureextractor.RadiomicsFeatureExtractor(binWidth=25)
extractor.enableFeatureClassByName('firstorder')  # first order statistics
extractor.enableFeatureClassByName('glcm')
extractor.enableFeatureClassByName('glrlm')
extractor.enableFeatureClassByName('glszm')
extractor.enableFeatureClassByName('ngtdm')
extractor.enableFeatureClassByName('shape')  # 3D shape features
# shape2D는 활성화하지 않음
extractor.enableAllImageTypes()
extractor.settings['force2D'] = True

# Create output directory if not exists
os.makedirs(output_dir, exist_ok=True)

# ==========================================
# Process Each Patient for a Given Image Type
# ==========================================
def process_patient(image_files, image_type):
    subj = image_files['Patient ID']
    image_path = image_files[f'{image_type}_rad']
    mask_path = image_files['Mask']
    vox_ref_path = image_files[f'{image_type}_vox']

    if not (os.path.exists(image_path) and os.path.exists(mask_path) and os.path.exists(vox_ref_path)):
        print(f"Skipping {subj}: Missing one of {image_path}, {mask_path}, {vox_ref_path}.")
        return None

    try:
        image = sitk.ReadImage(image_path)
        mask = sitk.ReadImage(mask_path)
        vox_ref = sitk.ReadImage(vox_ref_path)

        # 마스크 유효성 검사
        if sitk.GetArrayFromImage(mask).sum() == 0:
            print(f"Skipping {subj}: No labels found in the mask.")
            return None

        # 마스크와 영상의 공간 정보, 사이즈 불일치 시 재샘플링
        if (mask.GetSize() != image.GetSize() or 
            mask.GetSpacing() != image.GetSpacing() or 
            mask.GetDirection() != image.GetDirection() or 
            mask.GetOrigin() != image.GetOrigin()):

            mask = sitk.Resample(
                mask,
                image,
                sitk.Transform(),
                sitk.sitkNearestNeighbor,
                0,  # defaultPixelValue
                mask.GetPixelID()
            )

        # vox_ref_array를 이용해 mask 영역 제한
        vox_ref_array = sitk.GetArrayFromImage(vox_ref) > 0
        mask_array = sitk.GetArrayFromImage(mask).astype(bool) & vox_ref_array
        mask_array_sitk = sitk.GetImageFromArray(mask_array.astype(int))
        # 여기서 영상의 공간 정보로 복사
        mask_array_sitk.CopyInformation(image)
        mask = mask_array_sitk
    except Exception as e:
        print(f"Error processing images for {subj}: {e}")
        return None

    try:
        features = extractor.execute(image, mask)
    except Exception as e:
        print(f"Error extracting features for {subj}: {e}")
        return None

    for key in list(features.keys()):
        if key.startswith('diagnostics_'):
            del features[key]

    features = OrderedDict({'subj': subj, 'Category': image_files['Category'], **features})
    return pd.DataFrame(features, index=[0])


# ==========================================
# Process Patients in Parallel for All Image Types
# ==========================================
def extract_features_for_image_type(image_type):
    partial_func = partial(process_patient, image_type=image_type)
    results = []

    with Pool(processes=4) as pool:
        for res in tqdm(pool.imap(partial_func, patient_image_paths), total=len(patient_image_paths)):
            if res is not None:
                results.append(res)

    # Save the Radiomics Results
    if len(results) == 0:
        print(f"No results found for {image_type}. No Excel file will be saved.")
    else:
        df_radiomics = pd.concat(results, ignore_index=True)
        
        # Change the 'subj' column name to 'Patient ID'
        df_radiomics.rename(columns={'subj': 'Patient ID'}, inplace=True)
        
        df_radiomics.set_index(keys=['Patient ID'], inplace=True, drop=True)
        output_file_path = os.path.join(output_dir, f'radiomics_features_{image_type}.xlsx')

        # Save to Excel file
        df_radiomics.to_excel(output_file_path, engine='openpyxl')
        print(f"Radiomics features saved for {image_type}: {output_file_path}")


# ==========================================
# Extract Features for Each Image Type
# ==========================================
for image_type in ['Ktrans', 'ADC', 'rCBV']:
    print(f"Processing {image_type}...")
    extract_features_for_image_type(image_type)
