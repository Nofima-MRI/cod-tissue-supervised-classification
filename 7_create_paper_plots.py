import os
import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib.patches as mpatches
import seaborn as sns
from math import sqrt
from itertools import cycle
from scipy.stats import pearsonr


from functions.helper_functions import set_start, create_directory
from functions.hdf5_functions import get_datasets_from_group, read_dataset_from_group
from functions.project_functions import get_paths, get_parameters, check_mri_slice_validity, parse_patientname, treatment_to_title, state_to_title, process_connected_tissue
from functions.data_functions import mean_confidence_interval

def process_plot_mri_with_damaged(paths, params):
	"""
	Plot original MRI on left and MRI image with damaged overlayed on the right
	"""


	# hdf5 file that contains the original images
	hdf5_file = os.path.join(paths['hdf5_folder'], params['hdf5_file'])
	
	# get all patient names from original MRI group
	patients = get_datasets_from_group(group_name = params['group_original_mri'], hdf5_file = hdf5_file)

	# get list of patients without state
	patients = set([re.search('(.*) (fersk|Tint)', x).group(1) for x in patients])
	
	# loop over each patient, read data, perform inference
	for i, patient in enumerate(patients):

		logging.info(f'Processing patient: {patient} {i + 1}/{len(patients)}')

		# parse out treatment, sample, and state from patient name
		treatment, _, _ = parse_patientname(patient_name = f'{patient} fersk')

		"""
		Get fresh state
		"""
		# read original images
		fresh_original_images = read_dataset_from_group(dataset = f'{patient} fersk', group_name = params['group_original_mri'], hdf5_file = hdf5_file)
		# read reconstructed images
		fresh_reconstructed_images = read_dataset_from_group(dataset = f'{patient} fersk', group_name = params['group_segmented_classification_mri'], hdf5_file = hdf5_file)
		# only take damaged tissue and set connected tissue
		fresh_reconstructed_damaged_images = (process_connected_tissue(images = fresh_reconstructed_images.copy(), params = params) == 1)

		"""
		Get frozen/thawed
		"""
		# read original images
		frozen_original_images = read_dataset_from_group(dataset = f'{patient} Tint', group_name = params['group_original_mri'], hdf5_file = hdf5_file)
		# read reconstructed images
		frozen_reconstructed_images = read_dataset_from_group(dataset = f'{patient} Tint', group_name = params['group_segmented_classification_mri'], hdf5_file = hdf5_file)
		# only take damaged tissue and set connected tissue
		frozen_reconstructed_damaged_images = (process_connected_tissue(images = frozen_reconstructed_images.copy(), params = params) == 1)

		# get total number of slices to process
		total_num_slices = fresh_original_images.shape[0]
		# loop over each slice
		for mri_slice in range(total_num_slices):

			# check slice validity of fresh patient
			if check_mri_slice_validity(patient = f'{patient} fersk', mri_slice = mri_slice, total_num_slices = total_num_slices):

				if check_mri_slice_validity(patient = f'{patient} Tint', mri_slice = mri_slice, total_num_slices = total_num_slices):
	
					# setting up the plot environment
					fig, axs = plt.subplots(2, 2, figsize = (8, 8))
					axs = axs.ravel()

					# define the colors we want
					plot_colors = ['#250463', '#e34a33']
					# create a custom listed colormap (so we can overwrite the colors of predefined cmaps)
					cmap = colors.ListedColormap(plot_colors)
					# subfigure label for example, a, b, c, d etc
					sf = cycle(['a', 'b', 'c', 'd', 'e', 'f', 'g'])

					"""
					Plot fresh state
					"""
					# obtain vmax score so image grayscales are normalized better
					vmax_percentile = 99.9
					vmax = np.percentile(fresh_original_images[mri_slice], vmax_percentile)
					
					# plot fresh original MRI image
					axs[0].imshow(fresh_original_images[mri_slice], cmap = 'gray', vmin = 0, vmax = vmax)
					axs[0].set_title(rf'$\bf({next(sf)})$ Fresh - Original MRI')
					
					# plot fresh reconstucted image overlayed on top of the original image
					# axs[1].imshow(fresh_original_images[mri_slice], cmap = 'gray', vmin = 0, vmax = vmax)
					# im = axs[1].imshow(fresh_reconstructed_images[mri_slice],alpha = 0.7, interpolation = 'none')
					# axs[1].set_title(rf'$\bf({next(sf)})$ Fresh - Reconstructed')
					

					# plot fresh reconstucted image overlayed on top of the original image
					axs[1].imshow(fresh_original_images[mri_slice], cmap = 'gray', vmin = 0, vmax = vmax)
					axs[1].imshow(fresh_reconstructed_damaged_images[mri_slice], cmap = cmap, alpha = .5, interpolation = 'none')
					axs[1].set_title(rf'$\bf({next(sf)})$ Fresh - Reconstructed')
					
					"""
					Plot frozen/thawed state
					"""
					# plot frozen/thawed original MRI image
					# obtain vmax score so image grayscales are normalized better
					vmax = np.percentile(frozen_original_images[mri_slice], vmax_percentile)
					axs[2].imshow(frozen_original_images[mri_slice], cmap = 'gray', vmin = 0, vmax = vmax)
					axs[2].set_title(rf'$\bf({next(sf)})$ {treatment_to_title(treatment)} - Original MRI')

					# plot frozen reconstucted all classes
					# axs[4].imshow(frozen_original_images[mri_slice], cmap = 'gray', vmin = 0, vmax = vmax)
					# im = axs[4].imshow(frozen_reconstructed_images[mri_slice], alpha = 0.7, interpolation = 'none')
					# axs[4].set_title(rf'$\bf({next(sf)})$ {treatment_to_title(treatment)} - Reconstructed')
					
					# # plot frozen/thawed reconstucted image overlayed on top of the original image
					axs[3].imshow(frozen_original_images[mri_slice], cmap = 'gray', vmin = 0, vmax = vmax)
					axs[3].imshow(frozen_reconstructed_damaged_images[mri_slice], cmap = cmap, alpha = .5, interpolation = 'none')
					axs[3].set_title(rf'$\bf({next(sf)})$ {treatment_to_title(treatment)} - Reconstructed')

					"""
					Create custom legend
					"""
					# add custom legend				
					class_labels = {0 : 'background', 1 : 'damaged tissue'}
					class_values = list(class_labels.keys())
					# create a patch 
					patches = [ mpatches.Patch(color = plot_colors[i], label= class_labels[i]) for i in range(len(class_values)) ]
					axs[1].legend(handles = patches)#, bbox_to_anchor=(1.05, 1), loc = 2, borderaxespad=0. )
				
					# legend for fully reconstructed image
					# get class labels
					# class_labels = params['class_labels']
					# # get class indexes from dictionary
					# values = class_labels.keys()
					# # get the colors of the values, according to the 
					# # colormap used by imshow
					# plt_colors = [ im.cmap(im.norm(value)) for value in values]
					# # create a patch (proxy artist) for every color 
					# patches = [ mpatches.Patch(color = plt_colors[i], label= class_labels[i]) for i in range(len(values)) ]
					# # put those patched as legend-handles into the legend
					# axs[1].legend(handles = patches)#, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0. )
					
					"""
					Adjust figures
					"""
					# remove axis of all subplots
					[ax.axis('off') for ax in axs]
					# define plot subfolder
					subfolder = os.path.join(paths['paper_plot_folder'], 'original_vs_reconstructed', patient)
					# create subfolder
					create_directory(subfolder)
					# crop white space
					fig.set_tight_layout(True)
					# save the figure
					fig.savefig(os.path.join(subfolder, f'slice_{mri_slice}.pdf'))
					
					# close the figure environment
					plt.close()
					

def process_plot_correlation(paths, params):
	"""
	Correlation plot between avg. damaged tissue and drip loss
	"""

	# read tissue data
	tissue_data = pd.read_csv(os.path.join(paths['table_folder'], 'tissue_distributions.csv'), index_col = 0)

	# dictionary to hold avg damage
	patient_data = {}

	# get average damage per patient/sample
	for patient, group_data in tissue_data.groupby('patient'):

		# only frozen/thawed
		if 'Tint' in patient:
			# add patient data to dictionary
			patient_data[patient] = {}
			patient_data[patient]['damaged'] = group_data['rel_damaged'].mean()
	
	# read drip loss
	drip_loss = pd.read_csv(os.path.join(paths['table_folder'], 'drip_loss.csv'), delimiter = ';', decimal = ',')
	
	# loop over each row of dataframe
	for _, row in drip_loss.iterrows():

		# create patient dynamically
		patient = f"Torsk {int(row['treatment'])}-{int(row['sample'])} Tint"
		# add drip loss to dictionary
		patient_data[patient]['drip_loss'] = row['thawing_loss_percentage']
		patient_data[patient]['treatment'] = str(int(row['treatment']))
		patient_data[patient]['treatment_readable'] = treatment_to_title(int(row['treatment']))

	# convert to pandas dataframe
	data = pd.DataFrame(patient_data).T
	
	data['damaged'] = data['damaged'].astype('float')
	data['drip_loss'] = data['drip_loss'].astype('float')

	# calculate correlation
	correlation, _ = pearsonr(x = data['damaged'], y = data['drip_loss'])

	# set color
	colors = {'1' : 'red', '2' : 'green', '3' : 'blue'}
	titles = {	0 : u'-40 °C Freezing',
				1 : u'-20 °C Freezing',
				2 : u'-5 °C Freezing',
	}
	data['color'] = data.apply(lambda col: colors[col.treatment], axis = 1)

	# create figure environement
	fig, ax = plt.subplots(1,1, figsize= (5,5))
	
	# plot scatterplot with fitted line
	ax = sns.regplot(data = data, x = 'damaged', y = 'drip_loss', fit_reg = True, scatter_kws = {'s': 50, 'edgecolors' : 'face', 'facecolors': data['color'], 'zorder' : 1})
	
	# Make a legend
	# groupby and plot points of one color
	ind = 0
	for i, grp in data.groupby(['color']):
		grp.plot(kind = 'scatter', x = 'damaged', y = 'drip_loss', c = i, ax = ax, label = titles[ind], zorder = 0)
		ind += 1       
		
	ax.legend(loc=2)

	ax.set(xlabel='damaged tissue (%)', ylabel = 'liquid loss (%)')
	ax.set_title(fr'$r^{2}$ = {round(correlation * correlation,2)}')
	
	# crop white space
	fig.set_tight_layout(True)
	# save the figure
	fig.savefig(os.path.join(paths['paper_plot_folder'], 'correlation_plot.pdf'))
	
	# close the figure environment
	plt.close()
	

def process_plot_training_results_per_epoch(paths, params):


	
	# get training history file 
	training_history = os.path.join(paths['model_folder'], params['cnn_model'], 'history_training.csv')
	# get test history file
	test_history = os.path.join(paths['model_folder'], params['cnn_model'], 'history_test.csv')

	# load training data as dataframe
	data = pd.read_csv(training_history, index_col = 0)
	# load test data as dataframe
	test_data = pd.read_csv(test_history, index_col = 0)

	test_accuracy = round(test_data.loc['accuracy'].iloc[0],3)
	test_loss = round(test_data.loc['loss'].iloc[0],3)
	
	# get data from final epoch from early stopping
	final_epoch = data.sort_values('val_loss').iloc[0]


	# set up the plot environment	
	fig, axs = plt.subplots(1,2, figsize = (8,4), sharex = True)
	axs = axs.ravel()

	x = range(1,len(data) + 1)

	# plot the following
	plot_cols = ['loss', 'accuracy']
	plot_titles = ['Loss (crossentropy)', 'Accuracy']

	for i, col in enumerate(plot_cols):

		# take every nth sample
		n = 1
		# plot loss
		axs[i].plot(x[::n], data[f'{col}'][::n], label = 'training (80%)', zorder = 10)
		axs[i].set_title(plot_titles[i])
		axs[i].set_xlabel('number of epochs')
		axs[i].plot(x[::n], data[f'val_{col}'][::n], label = 'validation (10%)')
		
	# customize plot
	for ax in axs:

		ax.legend(loc='best', prop={'size': 10})
		ax.set_xlim(0,len(data))
		ax.axvline(int(final_epoch.name) + 1, linestyle = '--', color = 'gray')


	# annotate loss
	# training loss
	axs[0].annotate(round(final_epoch['loss'], 3), xy=(int(final_epoch.name) - 17 , final_epoch['loss'] + 0.005), textcoords='offset points', ha='left', va='bottom', color='#1f77b4')
	# validation loss
	axs[0].annotate(round(final_epoch['val_loss'], 3), xy=(int(final_epoch.name) - 17 , final_epoch['val_loss'] + 0.015),textcoords='offset points', ha='left', va='bottom',color='#ff7f0e')
	# test loss
	axs[0].annotate(f'test loss = {test_loss}', xy=(3 , 0.05),textcoords='offset points', ha='left', va='bottom', color='black')

	# annotate accuracy
	# training accuracy
	axs[1].annotate(round(final_epoch['accuracy'], 3), xy=(int(final_epoch.name) - 9 , final_epoch['accuracy']),textcoords='offset points', ha='left', va='bottom', color='#1f77b4')
	# validation accuracy
	axs[1].annotate(round(final_epoch['val_accuracy'], 3), xy=(int(final_epoch.name) - 9 , final_epoch['val_accuracy']),textcoords='offset points', ha='left', va='bottom',color='#ff7f0e')
	# test accuracy
	axs[1].annotate(f'test accuracy = {test_accuracy}', xy=(3 , 0.975),textcoords='offset points', ha='left', va='bottom', color='black')

	# crop white space
	fig.set_tight_layout(True)
	# save the figure
	fig.savefig(os.path.join(paths['paper_plot_folder'], f"training_results_{params['cnn_model']}.pdf"))
	# close the figure environemtn
	plt.close()

def process_nmr_distribution(paths, file_name = 'nmrdata.csv'):


	# read in data
	data = pd.read_csv(os.path.join(paths['table_folder'], file_name), index_col = 0)

	# create figure environment
	fig, ax = plt.subplots(1,1, figsize = (5,5))

	# # define x-axis
	x = data['Axis']
	
	# define linewidth
	lw = 0.7

	# plot fresh nmr signal
	ax.plot(x, data['Fresh'], label = 'Fresh', c = 'black', linewidth = lw)
	# plot -5
	ax.plot(x, data['-5'], label = treatment_to_title(1), c = 'red', linewidth = lw)
	# plot -20
	ax.plot(x, data['-20'], label = treatment_to_title(2), c = 'green', linewidth = lw)
	# plot -40
	ax.plot(x, data['-40'], label = treatment_to_title(3), c = 'blue', linewidth = lw)

	# add legend
	plt.legend()
	# set x-axis to log scale
	plt.xscale('log')
	# set x label
	plt.xlabel('T2 (ms)')
	# set y label
	plt.ylabel('Intensity')

	# crop white space
	fig.set_tight_layout(True)
	# save the figure
	fig.savefig(os.path.join(paths['paper_plot_folder'], 't2_distributions.pdf'))
	# close the figure environemtn
	plt.close()

def process_liquid_loss(paths, file_name = 'drip_loss.csv'):

	# read data
	data = pd.read_csv(os.path.join(paths['table_folder'], file_name), delimiter = ';', decimal = ',')

	# create figure environment
	fig, ax = plt.subplots(1,1, figsize = (5,5))

	# define colors for bars
	colors = {1 : 'red', 2 : 'green', 3 : 'blue'}

	# group data by treatment
	for group, grouped_data in data.groupby('treatment'):

		# calculate mean score of treatment
		mean = grouped_data['thawing_loss_percentage'].mean()
		# calculate 95% confidence interval
		error = mean_confidence_interval(data = grouped_data['thawing_loss_percentage'].values)
		# create bar chart
		ax.bar(group, mean, width = 0.5, yerr = error[3], label = treatment_to_title(group), color = colors[group], alpha = .7)

	# adjust x axis
	plt.xticks([1,2,3], ['Group 1', 'Group 2', 'Group 3'])
	# set y label
	plt.ylabel('Liquid Loss (%)')
	# show legend
	plt.legend()	
	# crop white space
	fig.set_tight_layout(True)
	# save the figure
	fig.savefig(os.path.join(paths['paper_plot_folder'], 'liquid_loss.pdf'))
	# close the figure environemtn
	plt.close()


if __name__ == "__main__":
	
	tic, process, logging = set_start()

	# get all folder paths
	paths = get_paths()
	# get project parameters
	params = get_parameters()

	# 1) original MRI next to damaged tissue
	process_plot_mri_with_damaged(paths, params)

	# 2) create correlation plot avg damaged vs drip loss
	# process_plot_correlation(paths, params)

	# 3) process plot showing training results per epoch for training and validation set
	# process_plot_training_results_per_epoch(paths, params)

	# 4) NMR T2 distribution of various tissue types
	# process_nmr_distribution(paths)

	# 5) Liquid loss bar chart
	# process_liquid_loss(paths)