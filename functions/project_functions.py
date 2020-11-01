"""
Project functions are functions that are specifically written for this project and typically find no use within another project

Author : Shaheen Syed
Date created : 2020-08-28

"""
import os
import re
import logging
import socket

from functions.helper_functions import create_directory, read_directory

def get_environment():

	# local environment should be: Shaheens-MacBook-Pro-2.local
	# nofima GPU workstation should be: shaheensyed-gpu
	# UIT GPU workstation should be: shaheengpu
	return socket.gethostname()

def get_paths(env = None, create_folders = True):
	"""
	Get all project paths
	"""

	# if environement argument is not given then get hostname with socket package
	if env is None:
		env = get_environment()

	# empty dictionary to return
	paths = {}

	# name of the project
	paths['project_name'] = 'cod_supervised_classification'

	# path for local machine
	if env == 'Shaheens-MacBook-Pro-2.local' or env == 'shaheens-mbp-2.lan':
		# project base folder
		paths['base_path'] = os.path.join(os.sep, 'Users', 'shaheen.syed', 'data', 'projects', paths['project_name'])
	elif env == 'shaheensyed-gpu':
		# base folder on nofima GPU workstation
		paths['base_path'] = os.path.join(os.sep, 'home', 'shaheensyed', 'projects', paths['project_name'])
	elif env == 'shaheengpu':
		# base folder on UIT GPU workstation
		paths['base_path'] = os.path.join(os.sep, 'home', 'shaheen', 'projects', paths['project_name'])
	else:
		logging.error(f'Environment {env} not implemented.')
		exit(1)

	# folder contained original MRI data in Dicom format
	paths['mri_folder'] = os.path.join(paths['base_path'], 'data', 'mri')
	# folder for HDF5 files
	paths['hdf5_folder'] = os.path.join(paths['base_path'], 'data', 'hdf5')
	# folder for .dcm files with new patient name
	paths['dcm_folder'] = os.path.join(paths['base_path'], 'data', 'dcm')
	# folder location for segmentation labels
	paths['segmentation_folder'] = os.path.join(paths['base_path'], 'data', 'segmentations')
	# define folder for features
	paths['feature_folder'] = os.path.join(paths['base_path'], 'data', 'features')
	# define folder for data augmentation
	paths['augmentation_folder'] = None#os.path.join(paths['base_path'], 'data', 'augmentation')
	# folder for datasets
	paths['dataset_folder'] = os.path.join(paths['base_path'], 'data', 'datasets')
	# define the plot folder
	paths['plot_folder'] = os.path.join(paths['base_path'], 'plots')
	# define plot folder for paper ready plots
	paths['paper_plot_folder'] = os.path.join(paths['base_path'], 'plots', 'paper_plots')
	# define folder for tables
	paths['table_folder'] = os.path.join(paths['base_path'], 'data', 'tables')
	# folde for trained models
	paths['model_folder'] = os.path.join(paths['base_path'], 'models')
	

	# create all folders if not exist
	if create_folders:
		for folder in paths.values():
			if folder is not None:
				if folder != paths['project_name']:
					create_directory(folder)	
			
	return paths

def get_parameters():
	"""
	Project parameters
	"""

	# empty dictionary to hold parameters
	params = {}

	# HDF5 file name
	params['hdf5_file'] = 'FISH-MRI-V2.h5'
	# original MRI group name
	params['group_original_mri'] = 'ORIGINAL-MRI'
	# HDF5 group name with mri images without background
	params['group_no_bg'] = 'NO-BG-MRI'
	# HDF5 group with images where the background is removed and gray scales adjusted
	params['group_no_bg_adjusted_gray_mri'] = 'NO-BG-ADJUSTED-GRAY-MRI'
	# HDF5 group with segmented classification
	params['group_segmented_classification_mri'] = 'SEGMENTED-CLASSIFICATION-MRI-V2'

	# feature size for feature creation
	params['feature_size'] = (8,8)
	# rescale factor. Scale 12 bit images to 0 and 1
	params['rescale_factor'] = 1 / (2 ** 12)

	# define class labels
	params['class_labels'] = {0 : 'background', 1 : 'damaged',  2 : 'non-damaged', 3: 'damaged connected', 4 : 'non-damaged connected'}
	# set label to class index
	params['label_to_class_index'] = {value: key for key, value in params['class_labels'].items()}
	# set non-damaged connected tissue to non-damaged tissue
	params['set_non_damaged_connected_to_non_damaged'] = False
	# set damaged connected tissue to damaged tissue
	params['set_damaged_connected_to_damaged'] = False
	# skip first number of slices due to artifacts
	params['trim_start_slices'] = 10
	# skip last number of slices due to artifacts
	params['trim_end_slices'] = 5

	# CNN model to use for inference
	params['cnn_model'] = '20200924002852' # v6 latest model
	# step size for sliding window inference
	params['step_size'] = (4,4)


	return params


def create_patients(treatments = [], samples = [], states = []):
	"""
		Create list of patients to process
	"""

	# define treatments
	treatments = range(1,4) if not treatments else treatments
	# define samples
	samples = range(1,11) if not samples else samples
	# define states
	states = ['fersk', 'Tint'] if not states else states

	# empty list to store patients to
	patients = []

	# dynamically create patients
	for treatment in treatments:
		for sample in samples:
			for state in states:
				# create patient
				patient = f'Torsk {treatment}-{sample} {state}'
				# add patient to list
				patients.append(patient)

	return patients

def get_protocol_translation(protocolname):

	# translation of protocolname
	protocol_translation = {'FSE T2w (axial,n)' : 'axial', 'FSE T2w (cor,n)' : 'cor'}

	return protocol_translation.get(str(protocolname))

def create_class_label(patientname):
	"""
	Based on the patientname, create a class label
	"""

	# convert patientname to string
	patientname = str(patientname)

	# create class label based on patientname
	if 'fersk' in patientname:
		return 0
	elif re.search(r'Torsk 1-.*Tint', str(patientname)):
		return 1
	elif re.search(r'Torsk 2-.*Tint', str(patientname)):
		return 2
	elif re.search(r'Torsk 3-.*Tint', str(patientname)):
		return 3
	else:
		logging.error(f'Patientname cannot be converted to class: {patientname}')
		exit(1)

def check_validity_mri_scan(patientname, datetime):
	"""
	Some scans were not successful and need to be removed from the analays

	Torsk 1-1 fersk 20200817101346

	Torsk 1-1 Tint 20200824111933
	Torsk 1-2 Tint 20200824114015
	Torsk 1-10 Tint 20200824130944
	Torsk 2-1 Tint 20200824133947

	"""	

	# concatenate name and datetime
	patientname = f'{patientname} {datetime}'

	invalid_scans = ['Torsk 1-1 fersk 20200817101346',
					'Torsk 1-1 Tint 20200824111933',
					'Torsk 1-2 Tint 20200824114015',
					'Torsk 1-10 Tint 20200824130944',
					'Torsk 2-1 Tint 20200824133947']

	return False if patientname in invalid_scans else True

def get_datasets_paths(datasets_folder):
	"""
	Read datasetes from datafolder and return back as dictionary of file locations

	Parameters
	----------
	data_folder : os.path
		location of X_train, Y_train, X_val, Y_val, X_test, and Y_test

	Returns
	---------
	datasets : dict()
		Dictionary with for example, X_train as key, and the file location as value
	"""

	# empty dictionary to store data to
	datasets = {}

	# loop over each file in directory and add file name as key and file location as value
	for file in read_directory(datasets_folder):

		# extra dataset name from file
		dataset_name = file.split(os.sep)[-1][:-4]
		# add to dictionary
		datasets[dataset_name] = file

	return datasets

def check_mri_slice_validity(patient, mri_slice, total_num_slices):
	"""
	When making MRI scans some slices are not valid and should be taken out of the analysis. After a visual inspection, these slices
	should be filtered out

	Parameters
	----------
	patient : string
		patient ID, for example Torks 1-1 fersk
	mri_slice : int
		slice of MRI scan
	"""

	# get parameters
	params = get_parameters()

	# trim start slices
	trim_start_slices = params['trim_start_slices']
	# trim end slices
	trim_end_slices = params['trim_end_slices']
	# skip slices
	valid_range_slices = range(total_num_slices)[trim_start_slices:None if trim_end_slices == 0 else -trim_end_slices]

	# check if whole patient is invalid
	# invalid_patients = ['Torsk 1-4 fersk']
	# if patient in invalid_patients:
	# 	return False

	# check if slice is in valid_range_slices
	if mri_slice not in valid_range_slices:
		return False

	# check if there are other slices that are invalid
	invalid_slices = {
					'Torsk 1-3 Tint' : [10,11],
					'Torsk 1-9 fersk' : range(0,21),
					'Torsk 1-10 Tint' : [10,11,13],
					'Torsk 2-1 Tint' : [10],
					'Torsk 2-3 Tint' : [10,11],
					'Torsk 2-5 fersk' : [10,11],
					'Torsk 2-5 Tint' : range(0,22),
					'Torsk 2-7 Tint' : range(0,15),
					'Torsk 2-9 fersk' : [11],
					'Torsk 2-9 Tint' : range(0,16),
					'Torsk 3-5 Tint' : range(0,18),
					'Torsk 3-6 Tint' : range(0,20)}
	
	if patient in invalid_slices:
		if mri_slice in invalid_slices[patient]:
			return False

	# all other cases the slice is valid
	return True

def parse_patientname(patient_name):
	"""
	Parse out treatment, sample, and state from patient name
	"""

	# extract treatment, sample, and state from patient name
	treatment = int(re.search('.* ([0-9]){1}', patient_name).group(1))
	sample = int(re.search('.*-([0-9]{1,2})', patient_name).group(1))
	state = re.search('.*(fersk|Tint)', patient_name).group(1)

	return treatment, sample, state

def treatment_to_title(treatment):
	"""
	Convert treatment ID to human readable title
	"""

	# first check state, since treatment has not yet been done if state is fresh
	titles = {	1 : u'-5 °C Freezing',
				2 : u'-20 °C Freezing',
				3 : u'-40 °C Freezing',
	}
	
	return titles[treatment]

def state_to_title(state):
	"""
	Convert state to human readable title
	"""

	titles = {	'fersk' : 'Fresh',
				'Tint' : 'Frozen/thawed'}

	return titles[state]

def process_connected_tissue(images, params):
	"""
	There are two switches that can set damaged-connected tissue to damaged tissue, and also non-damaged connected tissue to non-damage tissue
	See:
	# set non-damaged connected tissue to non-damaged tissue
	params['set_non_damaged_connected_to_non_damaged'] = True
	# set damaged connected tissue to damaged tissue
	params['set_damaged_connected_to_damaged'] = True 

	"""

	# check if connected damaged tissue needs to be set to damaged tissue
	if params['set_damaged_connected_to_damaged']:
		# {0 : 'background', 1 : 'damaged',  2 : 'non-damaged', 3: 'damaged connected', 4 : 'non-damaged connected'}
		images[images == 3] = 1
	# check if non-damaged connected needs to be set to non-damaged tissue
	if params['set_non_damaged_connected_to_non_damaged']:
		# {0 : 'background', 1 : 'damaged',  2 : 'non-damaged', 3: 'damaged connected', 4 : 'non-damaged connected'}
		images[images == 4] = 2

	return images