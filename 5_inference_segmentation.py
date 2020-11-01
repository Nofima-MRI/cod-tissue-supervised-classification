import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib.patches as mpatches

# parallel processing
from multiprocessing import cpu_count
from joblib import Parallel
from joblib import delayed

from functions.helper_functions import set_start, create_directory
from functions.hdf5_functions import get_datasets_from_group, read_dataset_from_group, save_data_to_group_hdf5
from functions.project_functions import get_paths, get_parameters, check_mri_slice_validity
from functions.img_functions import sliding_window_view, mask_image

from tensorflow.keras.models import load_model

def perform_inference_segmentation(paths, params):

	# hdf5 file that contains the original images
	hdf5_file = os.path.join(paths['hdf5_folder'], params['hdf5_file'])

	# path to trained CNN model
	model_file = os.path.join(paths['model_folder'], params['cnn_model'], 'model.h5')

	# get all patient names from original MRI group
	patients = get_datasets_from_group(group_name = params['group_no_bg'], hdf5_file = hdf5_file)
	
	# loop over each patient, read data, perform inference
	for i, patient in enumerate(patients):

		logging.info(f'Processing patient: {patient} {i}/{len(patients)}')

		# read images
		images = read_dataset_from_group(dataset = patient, group_name = params['group_no_bg'], hdf5_file = hdf5_file)

		# rescale 12bit images to 0-1
		images = images * params['rescale_factor']

		# create empty array to save reconstructed images
		segmented_images = np.empty_like(images, dtype = 'uint8')

		# use parallel processing to speed up processing time
		executor = Parallel(n_jobs = cpu_count(), backend = 'multiprocessing')

		# create tasks so we can execute them in parallel
		tasks = (delayed(classify_img_feature)(img = images[img_slice], 
												slice_idx = img_slice, 
												feature_size = params['feature_size'], 
												step_size = params['step_size'],
												model_file = model_file,
												verbose = True) for img_slice in range(images.shape[0]))
		
		# execute tasks and process the return values
		for segmented_image, slice_idx in executor(tasks):

			# add each segmented image slice to the overall array that holds all the slices
			segmented_images[slice_idx] = segmented_image

		# save segmentated image to HDF5 file
		save_data_to_group_hdf5(group = params['group_segmented_classification_mri'],
								data = segmented_images,
								data_name = patient,
								hdf5_file = hdf5_file,
								overwrite = True)
	

def classify_img_feature(img, slice_idx, feature_size, step_size, model_file, apply_mask = True, verbose = False):
	"""
	Take an MRI image slice, then perform a sliding window in x, and y direction to create smaller image patches.
	These image patches are then classified as damaged, non-damaged, damaged-connected, and non-damaged-connected

	Parameters
	-------------
	img : np.ndarray()
		a single slice of the MRI image, typically (256,256)
	slice_idx : int
		the index of the slice. Important for multiprocessing so we know which data we return
	feature_size : tuple
		size of the feature size used when creating the cnn model, for example (8,8)
	step_size : tuple
		step size for sliding window, basically how many pixels to skip/overlap.
	model_file : os.path
		location of the trained CNN model
	apply_mask : bool (optional)
		apply background mask. Default true
	"""

	if verbose:
		logging.debug(f'Processing slice: {slice_idx}')

	"""
		CREATE PATCHES FOR INFERENCE
	"""

	# empty list to hold the patches
	X = []

	# loop over y and x coordinates
	for y in range(0, img.shape[0] - step_size[0], step_size[0]):
		for x in range(0, img.shape[1] - step_size[1], step_size[1]):
			# go from x, y coordinates to a matrix with the size of 'feature_matrix'
			x_left = x 
			x_right = x + (feature_size[0])
			y_top = y 
			y_bot = y + (feature_size[1])
		
			X.append([img[y_top:y_bot, x_left : x_right]])
	
	# convert X to numpy array
	X = np.vstack(X)

	#  add a channel (1 for grayscale)
	X = np.expand_dims(X, axis = 3)

	"""
		PERFORM INFERENCE
	"""
	# load cnn model
	model = load_model(model_file)
	# infer labels
	y_hat = model.predict_classes(X)
	# create empty array to store the segmented image (these are the class labels for each pixel)
	segmented_image = np.empty_like(img, dtype = 'uint8')

	# index so we know what label we should use
	y_hat_idx = 0
	for y in range(0, img.shape[0] - step_size[0], step_size[0]):
		for x in range(0, img.shape[1] - step_size[1], step_size[1]):
			# go from x, y coordinates to a matrix with the size of 'feature_matrix'
			x_left = x 
			x_right = x + (feature_size[0])
			y_top = y 
			y_bot = y + (feature_size[1])
		
			# add labels to all pixels from y and x 
			segmented_image[y_top:y_bot, x_left : x_right] = y_hat[y_hat_idx] + 1
			y_hat_idx += 1	

	"""
		POSTPROCESSING
	"""

	# set background to zero
	if apply_mask:
		segmented_image = mask_image(img = segmented_image, segmented_img = img, mask_value = img[0][0], fill_value = 0)

	return segmented_image, slice_idx

def _classify_img_feature(img, slice_idx, feature_size, step_size, model_file, verbose = False):
	"""
	For each slice of an MRI image, classify features of size 'feature_size' (e.g., 8x8) with a sliding window approach

	Parameters
	------------
	img_slice : np.array()
		numpy array with a single slice of an MRI image
	slice_idx : int
		the slice index of the MRI image
	feature_matrix : tuple()
		tuple with width x height of the feature matrix, for instance, 8 x 8
	step_size :  
	"""

	# change argument variable names
	img_slice = img
	feature_matrix = feature_size
	step_size = step_size[0]

	logging.debug(f'Processing slice {slice_idx}')

	# load cnn model
	model = load_model(model_file)

	# create empty array to store the segmented image (these are the class labels for each pixel)
	segmented_image = np.empty_like(img_slice, dtype = 'uint8')

	# perform sliding window approach
	for x, y, window in _sliding_window(img_slice, step_size = step_size, window_size = feature_matrix):
	
		# check if shape is 8,8
		if window.shape == feature_matrix:

			# default label is 0, which means background
			label = 0

			# if more than half of the pixels within a feature matrix are 0.0, then this is still considered background
			if (window == 0.0).sum() < (feature_matrix[0] * feature_matrix[1] / 2):

				# reshape window to have (num_sampled x width x height x channels)
				window = np.expand_dims(np.expand_dims(window, axis = 0), axis = 3)
				
				# infer label and add 1, since 0 is assigned to background
				label = model.predict_classes(window).squeeze() + 1

			# go from x, y coordinates to a matrix with the size of 'feature_matrix'
			x_left = x 
			x_right = x + (feature_matrix[0])
			y_top = y 
			y_bot = y + (feature_matrix[1])
		
			# update the window based on label
			segmented_image[y_top:y_bot, x_left : x_right] = label
	
	return segmented_image, slice_idx

def _sliding_window(image, step_size, window_size):
	# slide a window across the image
	for y in range(0, image.shape[0], step_size):
		for x in range(0, image.shape[1], step_size):
			# yield the current window
			yield (x, y, image[y:y + window_size[1], x:x + window_size[0]])


def plot_segmented_images(paths, params):
	"""
	Plot segmented images
	"""

	# create hdf5 file
	hdf5_file = os.path.join(paths['hdf5_folder'], params['hdf5_file'])

	# get list of patient names to plot
	patients = get_datasets_from_group(group_name = params['group_segmented_classification_mri'], hdf5_file = hdf5_file)

	# plot each patient
	for i, patient in enumerate(patients):

		logging.info(f'Processing patient: {patient} {i}/{len(patients)}')

		# read segmented images
		images = read_dataset_from_group(dataset = patient, group_name = params['group_segmented_classification_mri'], hdf5_file = hdf5_file)

		# set up plotting environment
		fig, axs = plt.subplots(6,9, figsize = (20,20))		
		axs = axs.ravel()

		# loop over each slice and print
		for mri_slice in range(images.shape[0]):

			logging.debug(f'Processing slice: {mri_slice}') 

			# check slice validity
			if check_mri_slice_validity(patient = patient, mri_slice = mri_slice, total_num_slices = images.shape[0]):

				# plot image
				im = axs[mri_slice].imshow(images[mri_slice], vmin = 0, vmax = 5, interpolation='none')
			axs[mri_slice].set_title(f'{mri_slice}')

		# get class labels
		class_labels = params['class_labels']
		# get class indexes from dictionary
		values = class_labels.keys()
		# get the colors of the values, according to the 
		# colormap used by imshow
		colors = [ im.cmap(im.norm(value)) for value in values]
		# create a patch (proxy artist) for every color 
		patches = [ mpatches.Patch(color = colors[i], label= class_labels[i]) for i in range(len(values)) ]
		# put those patched as legend-handles into the legend
		plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0. )
		
		# make adjustments to each subplot	
		for ax in axs:
			ax.axis('off')

		# create plotfolder subfolder
		plot_sub_folder = os.path.join(paths['plot_folder'], 'segmentation', params['cnn_model'])
		create_directory(plot_sub_folder)

		# crop white space
		fig.set_tight_layout(True)
		# save the figure
		fig.savefig(os.path.join(plot_sub_folder, f'{patient}.png'))

		# close the figure environment
		plt.close()
	
if __name__ == "__main__":
	
	tic, process, logging = set_start()

	# get all folder paths
	paths = get_paths()
	# get project parameters
	params = get_parameters()

	# 1) reconstruct tissue with inference
	perform_inference_segmentation(paths = paths, params = params)

	# 2) plot segmented images
	plot_segmented_images(paths = paths, params = params)