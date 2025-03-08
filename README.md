# Opnestedcombat-RAVEL

## AMC-NE preprocess sequence

### dcm2niigz.py -> preproc.py -> preproc_hdbet.py -> preproc_moco.py -> preproc_registration_t1.py

## AMC-Mol preprocess sequence

### dcm2niigz.py -> preproc.py -> preproc_hdbet.py -> preproc_moco.py -> preproc_registration_t1.py

## SNUH preprocess sequence 

### arrange_data.py -> dcm2niigz.py -> preproc_skullstrip.py -> preproc_hdbet.py -> preproc_moco.py -> preproc_registration_t1.py


### util_modified.py -> rad = (array-mean) / np.std(array[array>0]) * 100 

### util_original.py -> rad = (array-mean) / np.std(normal_white_matter) * 100 

### util_csf.py -> RAVEL normalization(with csf)

### util_wm.py -> RAVEL normalization(with wm)

# All Sequence

## preprocess -> util_modified.py or util_original.py or util_csf.py or util_wm.py -> radiomics_total_indiv.py -> combat
