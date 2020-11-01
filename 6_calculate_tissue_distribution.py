import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from functions.helper_functions import set_start, create_directory
from functions.hdf5_functions import get_datasets_from_group, read_dataset_from_group
from functions.project_functions import get_paths, get_parameters, check_mri_slice_validity, parse_patientname, treatment_to_title, state_to_title, process_connected_tissue

def perform_calculate_tissue_distributions(paths, params):

	# hdf5 file that contains the original images
	hdf5_file = os.path.join(paths['hdf5_folder'], params['hdf5_file'])
	
	# get all patient names from original MRI group
	patients = get_datasets_from_group(group_name = params['group_segmented_classification_mri'], hdf5_file = hdf5_file)

	# empty pandas dataframe to hold all data
	data = pd.DataFrame()

	# loop over each patient, read data, perform inference
	for i, patient in enumerate(patients):

		logging.info(f'Processing patient: {patient} {i + 1}/{len(patients)}')

		# parse out treatment, sample, and state from patient name
		treatment, sample, state = parse_patientname(patient_name = patient)

		# read images
		images = read_dataset_from_group(dataset = patient, group_name = params['group_segmented_classification_mri'], hdf5_file = hdf5_file)

		# handle connected tissue (for example, set connected damaged tissue to damaged tissue)
		images = process_connected_tissue(images = images, params = params)

		# reshape image to unroll pixels in last two dimensions. Go from (54, 256, 256) to (54, 65536)
		images = images.reshape(images.shape[0], -1)
		
		# count damaged tissue
		damaged_tissue = np.sum((images == 1), axis = 1)
		# count non_damaged tissue
		non_damaged_tissue = np.sum((images == 2), axis = 1)
		# relative damaged
		rel_damaged = damaged_tissue / (damaged_tissue + non_damaged_tissue) * 100
		# relative non-damaged
		rel_non_damaged = 100 - rel_damaged
		
		# process data for each slice
		for mri_slice in range(images.shape[0]):

			# check slice validity
			if check_mri_slice_validity(patient = patient, mri_slice = mri_slice, total_num_slices = images.shape[0]):

				# add data to dictionary
				mri_slice_data = {	'patient' : patient,
									'treatment' : treatment,
									'sample' : sample,
									'state' : state,
									'mri_slice' : mri_slice,
									'damaged_pixels' : damaged_tissue[mri_slice],
									'non_damaged_pixels' : non_damaged_tissue[mri_slice],
									'rel_damaged' : rel_damaged[mri_slice],
									'rel_non_damaged' : rel_non_damaged[mri_slice],
									}
				# create unique ID
				mri_slice_id = f'{treatment}_{sample}_{state}_{mri_slice}'
				
				# add to pandas dataframe
				data[mri_slice_id] = pd.Series(mri_slice_data)

	# transpose and save dataframe as CSV
	data.T.to_csv(os.path.join(paths['table_folder'], 'tissue_distributions.csv'))

def create_table_per_sample(paths, params):

	# read tissue data
	tissue_data = pd.read_csv(os.path.join(paths['table_folder'], 'tissue_distributions.csv'), index_col = 0)

	# empty dataframe
	data = pd.DataFrame()

	# get average damage per patient/sample
	for patient, group_data in tissue_data.groupby('patient'):

		data[patient] = pd.Series( {'damage' : group_data['rel_damaged'].mean()})
	
	# transpose and save dataframe as CSV
	data.T.to_csv(os.path.join(paths['table_folder'], 'damage_per_patient.csv'))

def plot_tissue_distribution(paths, params):

	# load data
	data = pd.read_csv(os.path.join(paths['table_folder'], 'tissue_distributions.csv'), index_col = 0)

	# setting up the plot environment
	fig, axs = plt.subplots(1, 3, figsize = (10, 3), sharex = False, sharey=True)
	axs = axs.ravel()

	# define colors
	state_to_color = {'Tint' : 'blue', 'fersk' : 'red'}
				

	# group data by treatment
	for grouping, data_grouped in data.groupby(['treatment', 'state']):

		sns.distplot(data_grouped['rel_damaged'], hist = True, rug = False, ax = axs[grouping[0] - 1], axlabel = False, label = state_to_title(grouping[1]), color = state_to_color[grouping[1]])
		# set title
		axs[grouping[0] - 1].set_title(treatment_to_title(grouping[0]))
		# show legend
		axs[grouping[0] - 1].legend(loc = 'upper center')		


	# make adjustments to each subplot	
	for i, ax in enumerate(axs):

		# set x ticks 0.1 0.1 - 1.0
		ax.set_xticks(range(0,101, 10))
		# limit x axis
		ax.set_xlim(0,100)

		# limit y axis 
		ax.set_ylim(0,0.15)

		# only labels on bottom
		ax.set_xlabel('percentage')
		# only labels on side
		if i == 0:
			ax.set_ylabel('distribution')
	
	
	# crop white space
	fig.set_tight_layout(True)
	# save the figure
	fig.savefig(os.path.join(paths['paper_plot_folder'], 'damaged_tissue_comparison_by_treatment.pdf'))
	# close the figure environment
	plt.close()


if __name__ == "__main__":
	
	tic, process, logging = set_start()

	# get all folder paths
	paths = get_paths()
	# get project parameters
	params = get_parameters()

	# 1) Calculate the tissue distributions
	# perform_calculate_tissue_distributions(paths = paths, params = params)

	# 2) distributions per sample
	create_table_per_sample(paths, params)

	# 2) plot tissue distributions per treatment
	# plot_tissue_distribution(paths = paths, params = params)