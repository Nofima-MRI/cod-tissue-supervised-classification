import os
import numpy as np
import matplotlib.pyplot as plt
from cv2 import cv2

from itertools import cycle
from scipy import ndimage

from functions.helper_functions import set_start, set_end, create_directory
from functions.hdf5_functions import get_datasets_from_group, read_dataset_from_group, read_metadata_from_group_dataset, save_data_to_group_hdf5
from functions.img_functions import perform_knn_segmentation, mask_image, change_img_contrast, calculate_midpoint
from functions.project_functions import get_parameters, get_paths

def remove_bg(paths, params):
	"""
	Remove background from MRI images

	Parameters
	--------------
	hdf5_file : os.path
		location of HDF5 that contains the raw MRI data, and where we want to save data to
	img_group_name : string
		name of HDF5 group that contains the raw MRI images
	save_group_name : string
		name of HDF5 group to store images with background removed
	"""

	# dynamically create hdf5 file
	hdf5_file = os.path.join(paths['hdf5_folder'], params['hdf5_file'])

	# read original MRI datasets from HDF5 file
	D = get_datasets_from_group(group_name = params['group_original_mri'], hdf5_file = hdf5_file)

	# read data from each dataset and plot mri data
	for d_idx, d in enumerate(D):

		logging.info(f'Processing dataset : {d} {d_idx}/{len(D)}')

		# read data from group	
		data = read_dataset_from_group(group_name = params['group_original_mri'], dataset = d, hdf5_file = hdf5_file)

		# read meta data
		meta_data = read_metadata_from_group_dataset(group_name = params['group_original_mri'], dataset = d, hdf5_file = hdf5_file)

		logging.info(f'Processing patient : {meta_data["PatientName"]}')

		# new numpy array to hold segmented data
		data_segmented = np.empty_like(data, dtype = 'int16')

		# process each slice
		for i in range(data.shape[0]):

			# ind_cycle = cycle(range(10))
			# fig, axs = plt.subplots(1,8, figsize = (20,5))
			# axs = axs.ravel()

			# original MRI
			img = data[i]
			# plt_index = next(ind_cycle)
			# axs[plt_index].imshow(img, cmap = 'gray')
			# axs[plt_index].set_title('Original MRI')

			# change grayscale
			img = change_img_contrast(img, phi = 10, theta = 1)
			# plt_index = next(ind_cycle)
			# axs[plt_index].imshow(img, cmap = 'gray')
			# axs[plt_index].set_title('Changed gray scale')


			# convert to 8 bit
			if d not in ['Torsk 1-4 fersk']:
				img = np.array(img, dtype = 'uint8')
				# plt_index = next(ind_cycle)
				# axs[plt_index].imshow(img, cmap = 'gray')
				# axs[plt_index].set_title('Convert to 8 bit')


			# inverted colors
			# img = (255) - img
			# plt_index = next(ind_cycle)
			# axs[plt_index].imshow(img, cmap = 'gray')
			# axs[plt_index].set_title('Inverted MRI')


			# max filter
			img = ndimage.maximum_filter(img, size = 7)
			# plt_index = next(ind_cycle)
			# axs[plt_index].imshow(img, cmap = 'gray')
			# axs[plt_index].set_title('Max filter')

			# erosion
			img = cv2.erode(img, None, iterations = 4)
			# plt_index = next(ind_cycle)
			# axs[plt_index].imshow(img, cmap = 'gray')
			# axs[plt_index].set_title('Erosion')


			# gaussian filter
			img = cv2.GaussianBlur(img, (11, 11), 0)
			# plt_index = next(ind_cycle)
			# axs[plt_index].imshow(img, cmap = 'gray')
			# axs[plt_index].set_title('Gaussian Blur')

			# knn bg remove
			segmented_img = perform_knn_segmentation(n_clusters = 2, img = img)
			img = mask_image(img = data[i], segmented_img = segmented_img, mask_value = segmented_img[0][0], fill_value = 0)
			# plt_index = next(ind_cycle)
			# axs[plt_index].imshow(img, cmap = 'gray')
			# axs[plt_index].set_title('KNN BG remove')

			# add masked image to data_segmented, where we store each slice
			data_segmented[i] = img


			# plt.show()

		# save data to HDF5
		save_data_to_group_hdf5(group = params['group_no_bg'],
								data = data_segmented,
								data_name = d,
								hdf5_file = hdf5_file,
								meta_data = meta_data, 
								overwrite = True)
		
def process_plot_mri_images(paths, params):
	"""
	Plot MRI images from HDF5 file
	"""

	# dynamically create hdf5 file
	hdf5_file = os.path.join(paths['hdf5_folder'], params['hdf5_file'])

	# read datasets from HDF5 file
	D = get_datasets_from_group(group_name = params['group_no_bg'], hdf5_file = hdf5_file)

	# read data from each dataset and plot mri data
	for i, d in enumerate(D):

		logging.info(f'Processing dataset : {d} {i}/{len(D)}')

		# read data from group	
		data = read_dataset_from_group(group_name = params['group_no_bg'], dataset = d, hdf5_file = hdf5_file)

		# image plot folder
		image_plot_folder = os.path.join(paths['plot_folder'], params['group_no_bg'], d.split()[-1], d)
		
		# create folder to store image to
		create_directory(image_plot_folder)

		# a single image for each image in dimensions[0]
		for i in range(data.shape[0]):

			# create figure and axes
			fig, ax = plt.subplots(1, 1, figsize = (10,10))
			
			# plot mri image
			ax.imshow(data[i], cmap = 'gray', vmax = 1000)

			# remove all white space and axes
			plt.gca().set_axis_off()
			plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
			plt.margins(0,0)
			plt.gca().xaxis.set_major_locator(plt.NullLocator())
			plt.gca().yaxis.set_major_locator(plt.NullLocator())
						
			# save the figure
			fig.savefig(os.path.join(image_plot_folder, f'{i}.png'), dpi = 300)
			
			# close the plot environment
			plt.close()
		

if __name__ == '__main__':

	tic, process, logging = set_start()

	# get all folder paths
	paths = get_paths()
	# get project parameters
	params = get_parameters()

	# segment mri images so as to remove the background
	remove_bg(paths = paths, params = params)

	# plot images
	process_plot_mri_images(paths = paths, params = params)
	
	set_end(tic, process)