# Flores-Saiffe Farías Adolfo
# Universidad de Guadalajara
# adolfo.flores.saiffe@gmail.com
# saiffe.adolfo@alumnos.udg.mx
# ORCID:0000-0003-4504-7251
# Researchgate: https://www.researchgate.net/profile/Adolfo_Flores_Saiffe2
# Github: https://github.com/fitobawino
#
# This script Preprocess MRIs (freesurfer) and fMRIs (see pipeline below), 
# and parcellate the fMRIs with Glasser Atlas [1]. 
#
# [1] Glasser, M. F., et al., (2016). A multi-modal parcellation of human cerebral cortex. Nature, 536(7615), 171–178.

# 0.0 Create working directory

##############################################
# 1. Pre-processing and parcellation of MRIs #
##############################################
# 1.1 Motion and intensity correction and normalization: freesurfer's recon-all
# 1.2 Parcellation using Glasser (or AAL) atlas.
# 1.3 Brain extraction of structural images using FSL-BET (used during non-linear registration)

##############################
# 2. Pre-processing of fMRIs #
##############################
# 2.1 Motion correction with FSL
# 2.2 Slice timing correction with FSL
# 2.3 Brain extraction to fMRIs with FSL-BET
# 2.4 Non-linear co-registration
# 2.4.1 Obtain the mean-fMRI from 2.3
# 2.4.2 Obtain the inverse of pre-processed T1 from 1.3
# 2.4.3 Affine and non-linear registration from mean-fMRI (2.4.2) (moving) to inverse T1 (2.4.1) (fixed)
# 2.4.4 Apply transformation to 4D fMRIs (2.3)

############################
# 3. Parcellation of fMRIs #
############################
# 3.1 Affine registration from atlas voume (moving) to pre-processed MRI (1.2) (fixed)


###########################
# 4. Strucutural analysis #
###########################
# See glasser_structural_analysis.r

#############################################################################################

# 0.0 Create working directory
import os
import shutil
from os.path import join as opj

epi_path = '/...'
str_path = '/...'
out_path = '/.../Raw'
subj     = ['Group1','Group2']
str_sub  = 'T1_structural'
epi_sub  = 'fun_MRI'

# Re-arrange files in /.../Raw/<subject>/4D.nii (Optional)
for CorP in subj:
	nums = os.listdir(epi_path + CorP)
	for num in nums:
		epi_path = opj(epi_path, CorP, num, epi_sub, '4D.nii')
		struct_path = opj(str_path, CorP, num, str_sub)
		save_path = opj(out_path, num)
		os.mkdir(save_path)
#		os.remove(save_path + '4D.nii')
		shutil.copy(epi_path, save_path)
#		shutil.copy(struct_path + os.listdir(struct_path)[0], save_path + 'T1.nii')

#############################
# 1. Pre-processing of MRIs #
#############################
# 1.0 Import libraries and define working paths
from nipype.interfaces.freesurfer import ReconAll
from nipype.interfaces.fsl import BET
from os.path import join as opj
import os
from os import listdir
import pdb
import shutil as sh

working_dir       = '/.../'
subjs             = os.listdir(working_dir + 'Raw/')
autorecon_folder  = opj(working_dir, 's11_autorecon')
parcellation_path = opj(working_dir, 's12_parcell')
betMRI_folder     = opj(working_dir, 's13_bet')
glasser_bash_path = "/usr/local/freesurfer/subjects/Glasser"
if not os.path.exists(autorecon_folder): os.mkdir(autorecon_folder)
if not os.path.exists(parcellation_path): os.mkdir(parcellation_path)
if not os.path.exists(betMRI_folder): os.mkdir(betMRI_folder)


# 1.1 Motion and intensity correction and normalization: freesurfer's recon-all
reconall = ReconAll()
reconall.inputs.directive    = 'autorecon1'
reconall.inputs.subjects_dir = autorecon_folder
reconall.inputs.openmp       = 8
reconall.inputs.flags 		 = '-nuintensitycor-3T -rmneck'

def run_reconall(subj, working_dir):
	from os.path import join as opj
	reconall.inputs.subject_id = subj
	reconall.inputs.T1_files   = opj(working_dir,'Raw',subj,'T1.nii')
	reconall.run()

def mgz2nii(file):
	from nipype.interfaces.freesurfer import MRIConvert
	mc = MRIConvert()
	mc.inputs.in_file  = file
	mc.inputs.out_file = file[:-3] + 'nii'
	mc.inputs.out_type = 'nii'
	mc.run()

# 1.2 Parcellation using either AAL or Glasser atlases. (see: https://cjneurolab.org/2016/11/22/hcp-mmp1-0-volumetric-nifti-masks-in-native-structural-space/)
def glasser_parcellation(glasser_bash_path, subject):
	os.system("bash " + glasser_bash_path + "create_subj_volume_parcellation.sh " +
		"-L " + subject + " " +
		"-a HCP-MMP1 " +
		"-d " + subject + "/Glasser" +
		"-m YES -s YES")

# 1.3 Brain extraction of structural images using FSL-BET (used during non-linear registration)
def run_T1bet(subj, autorecon_folder, betMRI_folder):
	from os.path import join as opj
	mgz2nii(opj(autorecon_folder,subj,'mri','nu_noneck.mgz'))
	os.system('mkdir -p %s'%opj(betMRI_folder,subj))
	fslbet.inputs.in_file  = opj(autorecon_folder,subj,'mri','nu_noneck.nii')
	fslbet.inputs.out_file = opj(betMRI_folder,subj,'T1_brain.nii')
	fslbet.inputs.robust   = True
	fslbet.run()

# RUN!
for subj in subjs:
	run_reconall(subj, working_dir)
	glasser_parcellation(glasser_bash_path, subject=subj)
	run_T1bet(subj, autorecon_folder, betMRI_folder)

sh.move(glasser_bash_path, parcellation_path)


##############################
# 2. Pre-processing of fMRIs #
##############################
#2.0 Import libraries and create working directories
from nipype.interfaces import fsl as fsl
from nipype.interfaces.fsl import ImageStats
from os.path import join as opj
import os

working_dir    = '/.../'
raw_folder     = opj(working_dir, 'Raw')
motion_folder  = opj(working_dir, 's21_motion_correction')
slicet_folder  = opj(working_dir, 's22_slice_timing')
betfMRI_folder = opj(working_dir, 's23_betfMRI')
betMRI_folder  = opj(working_dir, 's13_bet')
regis_folder   = opj(working_dir, 's24_regis')
T1_inv_path    = betfMRI_folder
subjs          = os.listdir(raw_folder)


if not os.path.exists(motion_folder): os.mkdir(motion_folder)
if not os.path.exists(slicet_folder): os.mkdir(slicet_folder)
if not os.path.exists(betfMRI_folder): os.mkdir(betfMRI_folder)
if not os.path.exists(regis_folder): os.mkdir(regis_folder)


# 2.1 Motion correction with FSL
mcflirt = fsl.MCFLIRT()
def run_mcflirt(subj, raw_folder, motion_folder):
	print("\n    1.Motion correction")
	os.system('mkdir -p %s'%opj(motion_folder, subj))
	mcflirt.inputs.in_file    = opj(raw_folder, subj, '4D.nii')
	mcflirt.inputs.out_file   = opj(motion_folder, subj, '4D_static.nii.gz')
	mcflirt.inputs.mean_vol = True
	#mcflirt.inputs.stats_imgs = True
	#mcflirt.inputs.save_plots = True
	#mcflirt.inputs.stages     = 4
	print(mcflirt.cmdline)
	os.system(mcflirt.cmdline)

motion_out = fsl.MotionOutliers()
def run_motion_outliers(subj, motion_folder):
	print("\n    Motion outliers")
	motion_out.inputs.in_file          = opj(motion_folder, subj, '4D_static.nii.gz')
	motion_out.inputs.dummy            = 4
	motion_out.inputs.metric           = 'fd'
	motion_out.inputs.out_file         = opj(motion_folder, subj, 'outfile.txt')
	motion_out.inputs.output_type      = 'NIFTI_GZ'
	motion_out.inputs.out_metric_plot  = opj(motion_folder, subj, 'metricplot.png')
	motion_out.inputs.out_metric_values= opj(motion_folder, subj, 'metrics.txt')
	print(motion_out.cmdline)
	motion_out.run()

# 2.2 Slice timing correction with FSL
slice_timing = fsl.SliceTimer()
def run_slice_timing(subj, motion_folder, slicet_folder):
	print("\n    2.Slice timing correction")
	if not os.path.exists(opj(slicet_folder, subj)): os.mkdir(opj(slicet_folder, subj))
	slice_timing.inputs.in_file         = opj(motion_folder, subj, '4D_static.nii.gz')
	slice_timing.inputs.out_file        = opj(slicet_folder, subj, '4D_st.nii.gz')
	slice_timing.inputs.output_type     = 'NIFTI_GZ'
	slice_timing.inputs.time_repetition = 2
	print(slice_timing.cmdline)
	slice_timing.run()

# 2.3 Brain extraction to fMRIs with FSL-BET
fslbet = fsl.BET()
def run_fbet(subj, betfMRI_folder, slicet_folder):
	print("\n    3.Brain extraction of fMRI")
	if not os.path.exists(opj(betfMRI_folder, subj)): os.mkdir(opj(betfMRI_folder, subj))
	fslbet.inputs.in_file    = opj(slicet_folder, subj, '4D_st.nii.gz')
	fslbet.inputs.out_file   = opj(betfMRI_folder, subj, '4D_brain.nii.gz')
	fslbet.inputs.functional = True
	fslbet.meanvol           = True
	print(fslbet.cmdline)
	fslbet.run()

# 2.4 Non-linear co-registration
# 2.4.1 Obtain the mean-fMRI from 2.3
fslbet = fsl.BET()
def run_fbet_mean(subj, motion_folder, betfMRI_folder):
	print("\n    4.Obtain the mean-fMRI")
	fslbet.inputs.in_file    = opj(motion_folder, subj, '4D_static.nii.gz_mean_reg.nii.gz')
	fslbet.inputs.out_file   = opj(betfMRI_folder, subj, 'mean_brain.nii.gz')
	print(fslbet.cmdline)
	fslbet.run()

# 2.4.2 Obtain the inverse of pre-processed T1 from 1.3
def run_invert_T1(subj, T1_inv_path, betfMRI_folder, betMRI_folder):
	print("\n    5.Calculating inverse T1")
	stats = ImageStats(in_file = opj(betMRI_folder, subj, 'T1_brain.nii.gz'), op_string = '-R')
	T1minmax = stats.run()
	stats = ImageStats(in_file = opj(betfMRI_folder, subj, 'mean_brain.nii.gz'), op_string = '-R')
	EPIminmax = stats.run()
	factor = (T1minmax.outputs.out_stat[1] - T1minmax.outputs.out_stat[0])/(EPIminmax.outputs.out_stat[1] - EPIminmax.outputs.out_stat[0])
	os.system("fslmaths " +
		opj(betMRI_folder, subj, 'T1_brain.nii.gz') +
		" -mul -1" + 
		" -add " + str(T1minmax.outputs.out_stat[1]) + 
		" -mul " + str(factor) + " " + opj(betMRI_folder, subj, 'T1_brain_inv.nii.gz')
		)

# 2.4.3 Affine and non-linear registration from mean-fMRI (2.4.2) (moving) to inverse T1 (2.4.1) (fixed)
def run_nonlinear_regis(subj, betMRI_folder, betfMRI_folder, regis_folder):
	print("\n    6. Non-linear registration")
	if not os.path.exists(opj(regis_folder, subj)): os.mkdir(opj(regis_folder, subj))
	os.system('antsRegistrationSyN.sh' + 
		" -f " + opj(betMRI_folder, subj, 'T1_brain_inv.nii.gz') + 
		" -m " + opj(betfMRI_folder, subj, 'mean_brain.nii.gz') + 
		" -o " + opj(regis_folder, subj, "SyN_" + subj) + " -n 8 -r 2") #### default: -r 4
	#sh.move("SyN_" + subj + "Warped.nii.gz", )

# 2.4.4 Apply transformation to 4D fMRIs (2.3)
def run_apply_transforms(subj, betfMRI_folder, betMRI_folder, regis_folder):
	print("\n    7.Appy transformation to all fMRIs")
	if not os.path.exists(opj(betfMRI_folder, subj, '3D')): os.mkdir(opj(betfMRI_folder, subj, '3D'))
	if not os.path.exists(opj(regis_folder, subj, '3D')): os.mkdir(opj(regis_folder, subj, '3D'))
	os.chdir(opj(betfMRI_folder, subj, '3D'))
	os.system("fslsplit " + opj(betfMRI_folder, subj, "4D_brain.nii.gz"))
	vols = os.listdir(opj(betfMRI_folder, subj, "3D"))
	vols.sort()
	'''
	# Process 4D: Not enough memory
	os.system("antsApplyTransforms -d 4 -e 3" + 
		" -i " + opj(betfMRI_folder, subj, "4D_brain.nii.gz") +
		" -r " + opj(betMRI_folder, subj, "T1_brain_inv.nii.gz") + 
		" -o " + opj(regis_folder, subj, "trans_4D_brain.nii.gz") +
		" -t " + opj(regis_folder, subj, "SyN_" + subj + "1Warp.nii.gz") +
		" -t " + opj(regis_folder, subj, "SyN_" + subj + "0GenericAffine.mat"))
	'''
	
	# Separate 4D into 3D and register.
	for vol in vols:
		# Apply transforms to all fMRI volumes
		os.system("antsApplyTransforms -d 3 -e 0" + 
			" -i " + opj(betfMRI_folder, subj, "3D", vol) +
			" -r " + opj(betMRI_folder, subj, "T1_brain_inv.nii.gz") + 
			" -o " + opj(regis_folder, subj, "3D", vol) +
			" -t " + opj(regis_folder, subj, "SyN_" + subj + "1Warp.nii.gz") +
			" -t " + opj(regis_folder, subj, "SyN_" + subj + "0GenericAffine.mat") +
			" -v 1")
		# Return fMRI volumes to low-res
		os.system("antsApplyTransforms -d 3 -e 0" + 
			" -i " + opj(regis_folder, subj, "3D", vol) +
			" -r " + opj(betfMRI_folder, subj, 'mean_brain.nii.gz') +
			#opj(regis_folder, subj, "SyN_" + subj + "1InverseWarp.nii.gz") + 
			" -o " + opj(regis_folder, subj, "3D", "LR_" + vol) +
			" -n NearestNeighbor" +
			" -v 1")

	os.system("fslmerge -t " + 
		opj(regis_folder, subj, "LR_4D_fMRI.nii.gz") + 
		' ' + opj(' ' + regis_folder, subj, "3D/LR_") + opj(' ' + regis_folder, subj, "3D/LR_").join(vols))


for subj in subjs:
	print(" >>> PROCESSING SUBJECT: " + subj + " <<<")
	run_mcflirt(subj, raw_folder, motion_folder)
	#run_motion_outliers(subj, motion_folder)
	run_slice_timing(subj, motion_folder, slicet_folder)
	run_fbet(subj, betfMRI_folder, slicet_folder)
	run_fbet_mean(subj, motion_folder, betfMRI_folder)
	run_invert_T1(subj, T1_inv_path, betfMRI_folder, betMRI_folder)
	run_nonlinear_regis(subj, betMRI_folder, betfMRI_folder, regis_folder)
	run_apply_transforms(subj, betfMRI_folder, betMRI_folder, regis_folder)


############################
# 3. Parcellation of fMRIs #
############################
import os
from os.path import join as opj

working_dir       = '/media/fito/Databases/PROY_DIAB_Paty/Connectivity2/'
subjs             = os.listdir(working_dir + 'Raw/')
parcellation_path = opj(working_dir, 's12_parcell', 'Glasser')
betfMRI_folder    = opj(working_dir, 's23_betfMRI')

# 3.1 Affine registration from atlas voume (moving) to pre-processed MRI (1.2) (fixed)
def register_parcells(subj, parcellation_path, betfMRI_folder):
	os.system("antsApplyTransforms -d 3 -e 0" + 
		" -i " + opj(parcellation_path, subj, "HCP-MMP1.nii.gz") +
		" -r " + opj(betfMRI_folder, subj, 'mean_brain.nii.gz') +
		" -o " + opj(parcellation_path, subj, "LR_HCP-MMP1.nii.gz") +
		" -n NearestNeighbor")

for subj in subjs:
	register_parcells(subj, parcellation_path, betfMRI_folder)

