import os
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix

from functions.hdf5_functions import get_datasets_from_group
from functions.helper_functions import set_start, set_end, read_directory, create_directory, get_current_timestamp, save_pickle
from functions.project_functions import get_paths, get_parameters, create_patients, get_datasets_paths
from functions.data_functions import read_features, upsample_arrays, get_train_val_test_datasets, create_image_data_generator, shuffle_arrays
from functions.dl_functions import get_cnn_model

from tensorflow import distribute
from tensorflow.keras.models import load_model
from tensorflow import saved_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

def create_train_val_test_datasets(script_params, paths, params, start_y_from_zero = True):
	"""
	Read in X and Y features and create a training, development, and test set

	Parameters
	-----------
	script_params : dict()
		train_split = training size split of the dataset 
		val_split  = validation size split of the dataset (remainder is used for test)
		limit_treatments = in case a dataset needs to be created for only a subset of treatments 
		limit_samples = in case a dataset needs to be created for only a subset of samples
		limit_states = in case a dataset needs to be created for only a subset of states
		upsample = upsample under-represented classes to create a class balanced dataset
	paths : dict()
		folder locations to read/write data to (see project_functions)
	params : dict()
		dictionary holding project wide parameter values
	start_y_from_zero : bool (optional)
		if set to True, then class labels will start from zero. For example, class labels 1,2,3 will be converted to 0,1,2. Default is true
	"""

	# create list of patient names for filtering
	allow_patients = create_patients(treatments = script_params['limit_treatments'], samples = script_params['limit_samples'], states = script_params['limit_states'])

	# read features from file
	X, Y = read_features(paths = paths, allow_patients = allow_patients)

	# verbose original feature set
	logging.info('Number of features in original dataset')
	for y in np.unique(Y):
		logging.info(f'Found {np.sum(Y[Y == y])} features with class label {y}')
	
	# upsample data if set to True
	if script_params['upsample']:

		# upsample underrepresented samples to create a class balanced dataset
		X, Y = upsample_arrays(X, Y, perform_shuffle = True)

	# let y labels start from zero instead of some other class ID
	if start_y_from_zero:
		# calulate how much y need to shift to start from zero
		min_y = np.min(Y)
		# let Y start from zero
		Y = Y - min_y
		
	# create datasets
	datasets = get_train_val_test_datasets(X, Y, train_split = script_params['train_split'], val_split = script_params['val_split'])

	# save each dataset to disk
	for label, dataset in datasets.items():

		logging.debug(f'{label} shape {dataset.shape}')

		# save as numpy array to file
		np.save(os.path.join(paths['dataset_folder'], label), dataset)

def train_cnn_classifier(paths, params):
	"""
	Train CNN classifier

	Parameters
	-----------


	"""

	# grid search variables
	cnn_architectures = ['v6']

	for cnn_architecture in cnn_architectures:

		# read datasets from file
		datasets = get_datasets_paths(paths['dataset_folder'])
		# # type of architecture to use
		# cnn_architecture = 'v3'
		# read one dataset and extract number of classes
		num_classes = len(np.unique(np.load(datasets['Y_train'])))
		# read input shape
		input_shape = np.load(datasets['X_train']).shape
		# model checkpoint and final model save folder
		model_save_folder = os.path.join(paths['model_folder'], get_current_timestamp())
		# create folder
		create_directory(model_save_folder)
		

		"""
			DEFINE LEARNING PARAMETERS
		"""
		params.update({'ARCHITECTURE' : cnn_architecture,
					'NUM_CLASSES' : num_classes,
					'LR' : .05,
					'OPTIMIZER' : 'sgd',
					'TRAIN_SHAPE' : input_shape,
					'INPUT_SHAPE' : input_shape[1:],
					'BATCH_SIZE' : 32,
					'EPOCHS' : 100,
					'ES' : True,
					'ES_PATIENCE' : 20,
					'ES_RESTORE_WEIGHTS' : True,
					'SAVE_CHECKPOINTS' : True,
					'RESCALE' : params['rescale_factor'],
					'ROTATION_RANGE' : None,
					'WIDTH_SHIFT_RANGE' : None,
					'HEIGHT_SHIFT_RANGE' : None,
					'SHEAR_RANGE' : None,
					'ZOOM_RANGE' : None,
					'HORIZONTAL_FLIP' : False,
					'VERTICAL_FLIP' : False,
					'BRIGHTNESS_RANGE' : None,
					})

		"""
			DATAGENERATORS
		"""

		# generator for training data
		train_generator = create_image_data_generator(x = datasets['X_train'], y = datasets['Y_train'], batch_size = params['BATCH_SIZE'], rescale = params['RESCALE'],
													rotation_range = params['ROTATION_RANGE'], width_shift_range = params['WIDTH_SHIFT_RANGE'],
													height_shift_range = params['HEIGHT_SHIFT_RANGE'], shear_range = params['SHEAR_RANGE'], 
													zoom_range = params['ZOOM_RANGE'], horizontal_flip = params['HORIZONTAL_FLIP'],
													vertical_flip = params['VERTICAL_FLIP'], brightness_range = params['BRIGHTNESS_RANGE'],
													save_to_dir = None if paths['augmentation_folder'] is None else paths['augmentation_folder'])

		# generator for validation data
		val_generator = create_image_data_generator(x = datasets['X_val'], y = datasets['Y_val'], batch_size = params['BATCH_SIZE'], rescale = params['RESCALE'])	
		
		# generator for test data
		test_generator = create_image_data_generator(x = datasets['X_test'], y = datasets['Y_test'], batch_size = params['BATCH_SIZE'], rescale = params['RESCALE'])	

		"""
			CALLBACKS
		"""

		# empty list to hold callbacks
		callback_list = []

		# early stopping callback
		if params['ES']:
			callback_list.append(EarlyStopping(monitor = 'val_loss', min_delta = 0, patience = params['ES_PATIENCE'], restore_best_weights = params['ES_RESTORE_WEIGHTS'], verbose = 1, mode = 'auto'))

		# save checkpoints model
		if params['SAVE_CHECKPOINTS']:
			# create checkpoint subfolder
			create_directory(os.path.join(model_save_folder, 'checkpoints'))
			callback_list.append(ModelCheckpoint(filepath = os.path.join(model_save_folder, 'checkpoints', 'checkpoint_model.{epoch:02d}_{val_loss:.3f}_{val_accuracy:.3f}.h5'), save_weights_only = False, monitor = 'val_loss', mode = 'auto', save_best_only = True))

		"""
			TRAIN CNN MODEL
		"""

		# use multi GPUs
		mirrored_strategy = distribute.MirroredStrategy()
		
		# context manager for multi-gpu
		with mirrored_strategy.scope():

			# get cnn model architecture
			model = get_cnn_model(cnn_type = params['ARCHITECTURE'], input_shape = params['INPUT_SHAPE'], num_classes = params['NUM_CLASSES'], learning_rate = params['LR'], optimizer_name = params['OPTIMIZER'])
			
			history = model.fit(train_generator,
				epochs = params['EPOCHS'], 
				steps_per_epoch = len(train_generator),
				validation_data = val_generator,
				validation_steps = len(val_generator),
				callbacks = callback_list)

			# evaluate on test set
			history_test = model.evaluate(test_generator)

			# save the whole model
			model.save(os.path.join(model_save_folder, 'model.h5'))
			
			# save history of training
			pd.DataFrame(history.history).to_csv(os.path.join(model_save_folder, 'history_training.csv'))
			
			# save test results
			pd.DataFrame(history_test, index = ['loss', 'accuracy']).to_csv(os.path.join(model_save_folder, 'history_test.csv'))

			# save model hyperparameters
			pd.DataFrame(pd.Series(params)).to_csv(os.path.join(model_save_folder, 'params.csv'))

def calculate_classification_performance(paths, params, limit = None):
	"""
	Calculate classification performance on the labeled datasets
	"""

	# path to trained CNN model
	model_file = os.path.join(paths['model_folder'], params['cnn_model'], 'model.h5')
	checkpoint_model_file = os.path.join(paths['model_folder'], params['cnn_model'], 'checkpoint_model.h5')

	# load cnn models
	model = load_model(model_file)
	checkpoint_model = load_model(checkpoint_model_file)

	# read dataset paths
	datasets = get_datasets_paths(paths['dataset_folder'])

	# empty dictionary to hold results
	results = {}
	# empty dictionary to hold confusion matrix
	confusion_results = {}

	# calculate accuracy for each dataset
	for dataset in ['train', 'val', 'test']:
		
		logging.info(f'Calculating accuracy for {dataset} dataset')
		
		# read X
		X = np.load(datasets[f'X_{dataset}'])
		# read Y
		Y = np.load(datasets[f'Y_{dataset}'])
		
		# rescale X
		X = X * params['rescale_factor']
		
		# performance inference of cnn model
		y_hat = model.predict_classes(X[:limit])#.reshape(-1,1)
		# inference of checkpoint model
		checkpoint_y_hat = checkpoint_model.predict_classes(X[:limit])#.reshape(-1,1)


		"""
			Accuracy
		"""

		# calculate accuracy cnn model
		accuracy = accuracy_score(y_true = Y[:limit], y_pred = y_hat)
		# calculate accuracy of checkpoint model
		checkpoint_accuracy = accuracy_score(y_true = Y[:limit], y_pred = checkpoint_y_hat)

		logging.info(f'Accuracy CNN model: {accuracy}')
		logging.info(f'Accuracy checkpoint CNN model: {checkpoint_accuracy}')

		# add to dictionary
		results[f'{dataset}'] = accuracy
		results[f'{dataset}_checkpoint'] = checkpoint_accuracy

		"""
			Confusion matrix
		"""
		confusion = confusion_matrix(y_true = Y[:limit], y_pred = y_hat)
		checkpoint_confusion = confusion_matrix(y_true = Y[:limit], y_pred = y_hat)

		confusion_results[f'{dataset}'] = confusion
		confusion_results[f'{dataset}_checkpoint'] = checkpoint_confusion
	
	# save results to file
	results = pd.DataFrame(pd.Series(results))
	# save results to disk
	results.to_csv(path_or_buf = os.path.join(paths['model_folder'], params['cnn_model'], 'accuracy.csv'))

	# save confusion as pickle
	save_pickle(obj = confusion_results, file_name = 'confusion_results', folder = os.path.join(paths['model_folder'], params['cnn_model']))

if __name__ == "__main__":

	tic, process, logging = set_start()

	# get all folder paths
	paths = get_paths()
	# get project parameters
	params = get_parameters()

	# define dataset parameters
	dataset_params = {	'train_split' : 0.8, 
						'val_split' : 0.1,
						'limit_treatments' : [], 
						'limit_samples' : [], 
						'limit_states' : [],  
						'upsample' : True,
					}
	
	# 1) create featured dataset
	# create_train_val_test_datasets(script_params = dataset_params, paths = paths, params = params)

	# 2) Train CNN model
	# train_cnn_classifier(paths = paths, params = params)

	# 3) calculate classification accuracy on the labeled features
	calculate_classification_performance(paths = paths, params = params)