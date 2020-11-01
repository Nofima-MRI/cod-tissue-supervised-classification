import logging

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam, SGD

# from tensorflow.compat.v1 import ConfigProto
# from tensorflow.compat.v1 import InteractiveSession
# config = ConfigProto()
# config.gpu_options.allow_growth = True
# session = InteractiveSession(config = config)

# import tensorflow as tf
# config = tf.compat.v1.ConfigProto()
# config.gpu_options.allow_growth = True
# session = tf.compat.v1.InteractiveSession(config=config)

def get_cnn_model(cnn_type, input_shape, num_classes, learning_rate, loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'], optimizer_name = 'adam'):

	# create sequential model
	model = Sequential()
		
	if cnn_type == 'v1':

		model.add(Conv2D(input_shape = input_shape, filters = 64, kernel_size = (3,3), padding = "same", activation = "relu"))
		model.add(Conv2D(filters = 64, kernel_size = (3,3), padding = "same", activation = "relu"))
		model.add(MaxPooling2D(pool_size = (2,2), strides = (2,2)))
		
		model.add(Conv2D(filters = 128, kernel_size = (3,3), padding = "same", activation = "relu"))
		model.add(Conv2D(filters = 128, kernel_size = (3,3), padding = "same", activation = "relu"))
		model.add(MaxPooling2D(pool_size = (2,2), strides = (2,2)))
		
		model.add(Flatten())
		model.add(Dense(units = 4096, activation = "relu"))
		model.add(Dense(units = 4096, activation = "relu"))

	elif cnn_type == 'v2':

		model.add(Conv2D(input_shape = input_shape, filters = 64, kernel_size = (6,6), padding = "same", activation = "relu"))
		model.add(Conv2D(filters = 64, kernel_size = (6,6), padding = "same", activation = "relu"))
		model.add(MaxPooling2D(pool_size = (2,2), strides = (2,2)))
		
		model.add(Conv2D(filters = 128, kernel_size = (3,3), padding = "same", activation = "relu"))
		model.add(Conv2D(filters = 128, kernel_size = (3,3), padding = "same", activation = "relu"))

		model.add(Flatten())
		model.add(Dense(units = 1024, activation = "relu"))
		model.add(Dropout(0.5))
		model.add(Dense(units = 1024, activation = "relu"))
		model.add(Dropout(0.5))

	elif cnn_type == 'v3':

		model.add(Conv2D(input_shape = input_shape, filters = 128, kernel_size = (3,3), padding = "same", activation = "relu"))
		model.add(Conv2D(filters = 128, kernel_size = (3,3), padding = "same", activation = "relu"))
		model.add(MaxPooling2D(pool_size = (2,2), strides = (2,2)))

		model.add(Conv2D(input_shape = input_shape, filters = 128, kernel_size = (3,3), padding = "same", activation = "relu"))
		model.add(Conv2D(filters = 128, kernel_size = (3,3), padding = "same", activation = "relu"))
		model.add(MaxPooling2D(pool_size = (2,2), strides = (2,2)))

		model.add(Flatten())


	elif cnn_type == 'v4':

		model.add(Conv2D(input_shape = input_shape, filters = 128, kernel_size = (3,3), padding = "same", activation = "relu"))
		model.add(Conv2D(filters = 128, kernel_size = (3,3), padding = "same", activation = "relu"))
		model.add(MaxPooling2D(pool_size = (2,2), strides = (2,2)))

		model.add(Conv2D(input_shape = input_shape, filters = 128, kernel_size = (3,3), padding = "same", activation = "relu"))
		model.add(Conv2D(filters = 128, kernel_size = (3,3), padding = "same", activation = "relu"))
		model.add(MaxPooling2D(pool_size = (2,2), strides = (2,2)))

		model.add(Conv2D(input_shape = input_shape, filters = 128, kernel_size = (3,3), padding = "same", activation = "relu"))
		model.add(Conv2D(filters = 128, kernel_size = (3,3), padding = "same", activation = "relu"))
		model.add(MaxPooling2D(pool_size = (2,2), strides = (2,2)))

		model.add(Flatten())
	
	elif cnn_type == 'v5':

		model.add(Conv2D(input_shape = input_shape, filters = 128, kernel_size = (3,3), padding = "same", activation = "relu"))
		model.add(Conv2D(filters = 128, kernel_size = (3,3), padding = "same", activation = "relu"))
		model.add(MaxPooling2D(pool_size = (2,2), strides = (2,2)))

		model.add(Conv2D(input_shape = input_shape, filters = 128, kernel_size = (3,3), padding = "same", activation = "relu"))
		model.add(Conv2D(filters = 128, kernel_size = (3,3), padding = "same", activation = "relu"))
		model.add(MaxPooling2D(pool_size = (2,2), strides = (2,2)))

		model.add(Flatten())
		model.add(Dense(units = 1024, activation = "relu"))
		model.add(Dropout(0.5))
		model.add(Dense(units = 1024, activation = "relu"))
		model.add(Dropout(0.5))

	elif cnn_type == 'v6':

		model.add(Conv2D(input_shape = input_shape, filters = 256, kernel_size = (3,3), padding = "same", activation = "relu"))
		model.add(Conv2D(filters = 256, kernel_size = (3,3), padding = "same", activation = "relu"))
		model.add(MaxPooling2D(pool_size = (2,2), strides = (2,2)))

		model.add(Conv2D(input_shape = input_shape, filters = 256, kernel_size = (3,3), padding = "same", activation = "relu"))
		model.add(Conv2D(filters = 256, kernel_size = (3,3), padding = "same", activation = "relu"))
		model.add(MaxPooling2D(pool_size = (2,2), strides = (2,2)))

		model.add(Flatten())
		model.add(Dense(units = 2048, activation = "relu"))
		model.add(Dropout(0.5))
		model.add(Dense(units = 2048, activation = "relu"))
		model.add(Dropout(0.5))


	elif cnn_type == 'test':

		model.add(Conv2D(input_shape = input_shape, filters = 10, kernel_size = (3,3), padding = "same", activation = "relu"))

		model.add(Flatten())

	model.add(Dense(units = num_classes, activation = "softmax"))

	if optimizer_name == 'adam':
		optimizer = Adam(learning_rate)
	elif optimizer_name == 'sgd':
		optimizer = SGD(learning_rate)
	else:
		logging.error(f'Optimizer {optimizer_name} not yet implemented.')
	
	model.compile(optimizer = optimizer, loss = loss, metrics = metrics)

	return model
