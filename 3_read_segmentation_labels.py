import nibabel as nib
import numpy as np
import os

from functions.helper_functions import set_start, set_end, read_directory, create_directory
from functions.project_functions import get_paths, get_parameters
from functions.hdf5_functions import read_dataset_from_group

import matplotlib.pyplot as plt

def process_convert_segmentation_to_features(paths, params, verbose = True):

	# read in all segmentation files
	F = [x for x in read_directory(paths['segmentation_folder']) if x[-4:] == '.nii' or x[-7:] == '.nii.gz']

	# get feature size from params
	feature_size = params['feature_size']

	# process each segmentation file
	for f_idx, file in enumerate(F):
		
		logging.info(f'Processing segmentation file : {file} {f_idx}/{len(F)}')

		# extract patient name from file
		patient = file.split(os.sep)[-1][:-7]
		
		# read patient original MRI image
		original_images = read_dataset_from_group(group_name = params['group_original_mri'], 
												dataset = patient, 
												hdf5_file = os.path.join(paths['hdf5_folder'], params['hdf5_file']))

		# check if original image can be found
		if original_images is None:
			logging.error(f'No original image found, please check patient name : {patient}')
			exit(1)
		
		# read in nifti file with segmenation data. shape 256,256,54
		images = nib.load(file)

		# empty lists to store X and Y features
		X = []
		Y = []

		# fig, axs = plt.subplots(6,4, figsize = (10,10))
		# axs = axs.ravel()
		# plt_idx = 0

		# process each slice
		for mri_slice in range(images.shape[2]):

			if verbose:
				logging.debug(f'Slice : {mri_slice}')
			
			# extract image slice
			img = images.dataobj[:,:,mri_slice]

			# test image for patchers
			# img_patches = np.zeros((img.shape))

			# check if there are any segmentations to be found
			if np.sum(img) == 0:
				if verbose:
					logging.debug('No segmentations found, skipping...')
				continue

			# we have to now flip and rotate the image to make them comparable with original dicom orientation when reading it into pyhon
			img = np.flip(img, 1)
			img = np.rot90(img)

			# unique segmentation classes
			seg_classes = np.unique(img)
			# remove zero class (this is the background)
			seg_classes = seg_classes[seg_classes != 0]
			
			# get features for each class
			for seg_class in seg_classes:

				if verbose:
					logging.debug(f'Processing segmentation class : {seg_class}')
			
				# check which rows have an annotation (we skip the rows that don't have the annotation)
				rows = np.argwhere(np.any(img[:] == seg_class, axis = 1))
				# check which colums have an annotation
				cols = np.argwhere(np.any(img[:] == seg_class, axis = 0))
				# get start and stop rows 
				min_rows, max_rows = rows[0][0], rows[-1][0]
				# get start and stop columns
				min_cols, max_cols = cols[0][0], cols[-1][0]

				logging.debug(f'Processing rows: {min_rows}-{max_rows}')
				logging.debug(f'Processing cols: {min_cols}-{max_cols}')
				
				# loop over rows and columns to extract patches of the image and check if there are annotations	
				for i in range(min_rows, max_rows - feature_size[0]):
					for j in range(min_cols, max_cols - feature_size[1]):

						# extract image patch with the dimensions of the feature
						img_patch = img[i:i + feature_size[0], j : j + feature_size[1]]

						# check if all cells have been annotated
						if np.all(img_patch == seg_class):

							# extract patch from original MRI image, these will contain the features.
							patch = original_images[mri_slice][i:i + feature_size[0], j : j + feature_size[1]]
							
							# add patch to X and segmentation class to Y
							X.append([patch])
							Y.append([seg_class])

		# 					img_patches[i:i + feature_size[0], j : j + feature_size[1]] = seg_class
			
		# 	axs[plt_idx].imshow(original_images[mri_slice], cmap = 'gray')
		# 	axs[plt_idx + 1].imshow(img_patches, vmin = 0, vmax = 3, interpolation = 'nearest')

		# 	plt_idx += 2

		# plt.show()
		# continue

		# convert X and Y to numpy arrays
		X = np.vstack(X)
		Y = np.vstack(Y)

		# create save folder location
		save_folder = os.path.join(paths['feature_folder'], patient)
		# create folder
		create_directory(save_folder)
		# save features to disk
		np.savez(file = os.path.join(save_folder, f'{patient}.npz'), X = X, Y = Y)


if __name__ == "__main__":
	
	tic, process, logging = set_start()

	# get all folder paths
	paths = get_paths()
	# get project parameters
	params = get_parameters()

	# 1) read segmentations and create features
	process_convert_segmentation_to_features(paths = paths, params = params)

	set_end(tic, process)