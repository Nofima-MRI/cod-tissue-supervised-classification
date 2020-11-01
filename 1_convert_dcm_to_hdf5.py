# IMPORT PACKAGES
import os
import re
import pydicom
import dicom_numpy
import numpy as np
import matplotlib.pyplot as plt
import shutil

# IMPORT FUNCTIONS
from functions.helper_functions import set_start, set_end, read_directory, create_directory
from functions.project_functions import get_parameters, get_paths, get_protocol_translation, create_class_label, check_validity_mri_scan
from functions.hdf5_functions import save_data_to_group_hdf5, get_datasets_from_group, read_dataset_from_group, read_metadata_from_group_dataset, save_meta_data_to_group_dataset

def process_convert_dcm_to_hdf5(paths, params, copy_dcm_files = False):
	"""
	Read dcm files from file and extract image data and save as HDF5

	Parameters
	----------
	
	"""

	# read all dicom files
	F = [f for f in read_directory(paths['mri_folder']) if f[-4:] == '.dcm']

	# loop over each file, read dicom files, save data
	for f in F:

		# verbose
		logging.info(f'Processing file : {f}')

		# read dcm file
		dataset = pydicom.dcmread(f)

		# construct meta data
		meta_data = {	'SOPClassUID' : dataset.SOPClassUID,
						'SeriesInstanceUID' : dataset.SeriesInstanceUID,
						'PatientName' : dataset.PatientName,
						'PatientID' : dataset.PatientID,
						'SeriesNumber' : dataset.SeriesNumber,
						'Rows' : dataset.Rows,
						'Columns' : dataset.Columns,
						'AcquisitionDateTime' : dataset.AcquisitionDateTime,
						'ProtocolName' : dataset.ProtocolName,
						'SeriesDescription' : dataset.SeriesDescription
					}
						
		# convert all meta data to string
		meta_data = {key : str(value) for key, value in meta_data.items()}

		# get the MRI image				
		image = dataset.pixel_array

		# verbose
		logging.debug(f'Image shape : {image.shape}')
		logging.debug(f'Image datatype : {image.dtype}')

		# if image has 3 slices, then this is the scout image (the first image to get a quick scan of the sample), this we skip
		if image.shape[0] == 3:
			logging.debug('MRI image is scout, skipping...')
			continue


		# get treatment-sample combination
		treatment_sample = re.findall('[0-9]+-[0-9]+', meta_data['PatientName'])[0]

		# get state
		state = f.split(os.sep)[-6]

		# change patient name into specific format that will be used throughout all analysis
		# This is: Torsk [treatment]-[sample] [Tint|fersk]
		# for example: Tork 1-1 fersk
		patient_name = 'Torsk {} {}'.format(treatment_sample, state)

		# check if patient scan is valid
		if not check_validity_mri_scan(patientname = patient_name, datetime = meta_data['AcquisitionDateTime']):
			logging.debug('Scan was invalid, skipping...')
			continue

		if copy_dcm_files:
			# copy .DCM file to folder with new patient name
			destination_folder = os.path.join(paths['dcm_folder'], state, patient_name)
			# create state folder
			create_directory(destination_folder)
			# define source file
			source_file = f
			# define destination file
			destination_file = os.path.join(destination_folder, f'{patient_name}.dcm')
			# start copying
			try:
				shutil.copy(source_file, destination_file)
			except Exception as e:
				logging.error(f'Failed copying .DCM file to dcm folder: {e}')
		
		# add extra meta data
		meta_data['ClassLabel'] = create_class_label(patient_name)
		meta_data['ProtocolTranslation'] = get_protocol_translation(dataset.ProtocolName)
		meta_data['DrippLossPerformed'] = False 
						
		# save data to HDF5
		save_data_to_group_hdf5(group = params['group_original_mri'],
								data = image,
								data_name = patient_name, 
								hdf5_file = os.path.join(paths['hdf5_folder'], params['hdf5_file']),
								meta_data = meta_data, 
								overwrite = True)


def process_plot_mri_images(paths, params):
	"""
	Plot MRI images from HDF5 file
	"""

	# create full path of HDF5 file
	hdf5_file = os.path.join(paths['hdf5_folder'], params['hdf5_file'])

	# read datasets from HDF5 file
	D = get_datasets_from_group(group_name = params['group_original_mri'], hdf5_file = hdf5_file)

	# read data from each dataset and plot mri data
	for d in D:

		# read data from group	
		data = read_dataset_from_group(group_name = params['group_original_mri'], dataset = d, hdf5_file = hdf5_file)

		# image plot folder
		image_plot_folder = os.path.join(paths['plot_folder'], params['group_original_mri'], d.split()[-1], d)

		# create folder to store image to
		create_directory(image_plot_folder)

		# a single image for each image in dimensions[0]
		for i in range(data.shape[0]):

			# create figure and axes
			fig, ax = plt.subplots(1, 1, figsize = (10,10))
			
			# plot mri image
			ax.imshow(data[i], cmap = 'gray')

			# crop white space
			plt.gca().set_axis_off()
			plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
			plt.margins(0,0)
			plt.gca().xaxis.set_major_locator(plt.NullLocator())
			plt.gca().yaxis.set_major_locator(plt.NullLocator())
			
			# save the figure
			fig.savefig(os.path.join(image_plot_folder, f'{i}.png'), dpi = 300)
			
			# close the plot environment
			plt.close()
		
# SCRIPT STARTS HERE	
if __name__ == '__main__':

	tic, process, logging = set_start()

	# get all folder paths
	paths = get_paths(env = 'local')
	# get project parameters
	params = get_parameters()

	"""
		CALL FUNCTIONS
	"""
	# read dcm files, extract meta data, extract MRI image data, and save as HDF5
	process_convert_dcm_to_hdf5(paths = paths, params = params, copy_dcm_files = True)

	# plot mri images
	process_plot_mri_images(paths = paths, params = params)
	
	set_end(tic, process)
