import tensorflow as tf
from tensorflow import keras

from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input, Flatten, Reshape,Conv1D, MaxPooling1D, AveragePooling1D, Add, Multiply,Concatenate
from keras.optimizers import Adam

from keras import backend as K
from keras.callbacks import *
from keras.utils import *
from keras.regularizers import l1,l2,l1_l2

from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard
from sklearn.metrics import confusion_matrix

import os
import time
import random
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "6"
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

class Evaluate():
	def __init__(self, population=None, X_train=None, Y_train=None, X_valid=None, Y_valid=None, num_classes=5, N_samples=128,epochs=10, batch_size=256,max_complexity=100000):
		self.population=population
		self.X_train=X_train
		self.X_valid=X_valid
		self.Y_train=Y_train
		self.Y_valid=Y_valid
		self.epochs=epochs
		self.batch_size=batch_size
		self.num_classes=num_classes
		self.N_samples=N_samples
		self.max_complexity=max_complexity

		#print("evaluate pop ",self.population)
		#print("n samples ",self.N_samples)

	def evaluate_population(self, gen_no):
		for i in range(self.population.get_pop_size()):
			indi = self.population.get_individual_at(i)
			indi.complexity=100000
			indi.accuracy=0.0
			accuracy=0.0
			try:

				complexity, accuracy = self.evaluate_individual(indi,gen_no)
				indi.complexity = complexity
				indi.accuracy = accuracy
			except:
				indi.complexity=100000
				indi.accuracy=0.0
			gama=0.3
			indi.c_prob=gama*indi.c_prob + (1-gama)*accuracy
			#indi.c_prob=1.0

	def get_activation_type(self,activation):
		#print("activation is ",activation)
		activation_type='relu'

		if activation>=0.25 and activation<0.5:
			activation_type='selu'

		if activation>=0.5 and activation<0.75:
			activation_type='tanh'

		if activation>=0.75:
			activation_type='linear'

		return activation_type

	def get_kernel_regularizer(self,kernel_reg):
		
		kernel_regularizer_type=None

		if kernel_reg>=0.25 and kernel_reg<0.5:
			kernel_regularizer_type=keras.regularizers.l1

		if kernel_reg>=0.5 and kernel_reg<0.75:
			kernel_regularizer_type=keras.regularizers.l2

		if kernel_reg>=0.75:
			kernel_regularizer_type=keras.regularizers.l1_l2

		return kernel_regularizer_type

	def build_block(self,block_layer,block_in):

		#print("building a block width ", block_layer.width)

		id_x=block_in
		identity_branch=block_layer.identity_branch

		id_x=keras.layers.Conv1D(filters=identity_branch.conv_layer.filters,
				kernel_size=identity_branch.conv_layer.kernel_size,
				activation=self.get_activation_type(identity_branch.conv_layer.activation))(id_x)


		if identity_branch.before_merge == True:
			block_in=id_x
			
		
		#print("id shape ",id_x.get_shape())
		conv_xs=""

		for i in range(block_layer.width):
			conv_x=block_in
			conv_branch=block_layer.conv_branch
			#print("width ", i)
			#j=0
			for conv in conv_branch:
				#print("depth ", j)
				#j=j+1

				conv_x=keras.layers.Conv1D(filters=conv.filters,
				kernel_size=conv.kernel_size,
				activation=self.get_activation_type(conv.activation))(conv_x)
			
			if i==0:
				conv_xs=conv_x
			else:
				conv_xs=keras.layers.Concatenate(axis=2)([conv_xs,conv_x])
				#print("conv xs shape ",conv_xs.get_shape())
			
			#print("conv x shape ",conv_x.get_shape())

		#print(" final conv xs shape ",conv_xs.get_shape())
		#print("id shape ",id_x.get_shape())

		zero_padding=int(id_x.get_shape()[1])-int(conv_xs.get_shape()[1])
		#print("zero padd ", zero_padding)

		if zero_padding>0:
			conv_xs=keras.layers.ZeroPadding1D(padding=(zero_padding,0))(conv_xs)

		if zero_padding<0:
			id_x=keras.layers.ZeroPadding1D(padding=(np.abs(zero_padding),0))(id_x)


		#print(" after padding final conv xs shape ",conv_xs.get_shape())
		#print("id shape ",id_x.get_shape())

		merge_function=block_layer.merge_layer.merge_function
		#0-0.33 Concat, 0.3-0.66 Multiply, 0.66-1 Add
		#block_out=conv_xs+id_x
		#merge_function=0.8
		

		if merge_function<0.33:
			#print("concat")
			block_out=keras.layers.Concatenate(axis=1)([conv_xs,id_x])

		if merge_function>=0.33 and merge_function<0.66:
			#print("mult")
			block_out=keras.layers.Multiply()([conv_xs,id_x])

		if merge_function>=0.66:
			block_out=keras.layers.Add()([conv_xs,id_x])
			#print("add")

		if block_layer.pooling_layer!=None:
			if block_layer.pooling_layer.kernel_type<0.5:
				block_out=keras.layers.MaxPooling1D(pool_size=block_layer.pooling_layer.kernel_size)(block_out)
			else:
				block_out=keras.layers.AveragePooling1D(pool_size=block_layer.pooling_layer.kernel_size)(block_out)


		#print("out shape ",block_out.get_shape())


		return block_out

	def build_model(self,indi):
		#print(indi)

		inp_seq = keras.layers.Input(shape=(self.N_samples,2))
		x=inp_seq

		for i in range(indi.get_layer_size()):
			layer=indi.get_layer_at(i)
			#print(layer)
			
			if layer.type==0:
				x=self.build_block(layer,x)

			if layer.type==1:
				x=keras.layers.Conv1D(filters=layer.filters,
				kernel_size=layer.kernel_size,
				activation=self.get_activation_type(layer.activation))(x)

			if layer.type==2:
				if layer.kernel_type<0.5:
					x=keras.layers.MaxPooling1D(pool_size=layer.kernel_size)(x)
				else:
					x=keras.layers.AveragePooling1D(pool_size=layer.kernel_size)(x)

			if layer.type==6:
				x=keras.layers.Flatten()(x)

			if layer.type==7:
				x=keras.layers.GlobalMaxPooling1D()(x)

			if layer.type==8:
				x=keras.layers.GlobalAveragePooling1D()(x)

			if layer.type==3:
				activation_type=self.get_activation_type(layer.activation)
				x=keras.layers.Dense(units=layer.units,activation=activation_type)(x)


		out=keras.layers.Dense(self.num_classes,activation='softmax')(x)

		model= keras.Model(inputs=inp_seq, outputs=out)

		model.summary()

		return model


	def evaluate_best(self,indi, X_test, Y_test, snr_mod_pairs_test, snrs, mods,gen_no=100):
		print("Evaluating the best individual....")
		print("snr mod pairs", snr_mod_pairs_test.shape)

		random.seed(33)
		os.environ['PYTHONHASHSEED'] = str(33)
		#print(tf.__version__)
		session_conf = tf.compat.v1.ConfigProto(
			intra_op_parallelism_threads=1, 
			inter_op_parallelism_threads=1)

		sess = tf.compat.v1.Session(
			graph=tf.compat.v1.get_default_graph(), 
			config=session_conf)
		tf.compat.v1.keras.backend.set_session(sess)

		es = EarlyStopping(patience=5, verbose=1, min_delta=0.001, monitor='val_acc', mode='auto')

		model=self.build_model(indi)
		#print("indi learning rate is ", indi.learning_rate)

		if indi.learning_rate<0.5:
			opt=keras.optimizers.Adam(learning_rate=0.001)
		elif indi.learning_rate>=0.5 and indi.learning_rate<0.75:
			opt=keras.optimizers.Adam(learning_rate=0.01)
		else:
			opt=keras.optimizers.Adam(learning_rate=0.0001)

		
		model.compile(loss='categorical_crossentropy',optimizer=opt,metrics=['accuracy'])
		
		trainable_count = np.sum([K.count_params(w) for w in model.trainable_weights])
		non_trainable_count = np.sum([K.count_params(w) for w in model.non_trainable_weights])
		print('Total params: {:,}'.format(trainable_count + non_trainable_count))
		print('Trainable params: {:,}'.format(trainable_count))
		print('Non-trainable params: {:,}'.format(non_trainable_count))
		complexity=trainable_count+non_trainable_count

		
		cp_save_path='/home/eperenda/multiple_signals/amc_franco/best_model_weights_gen'
		tb_log_dir = '/home/eperenda/multiple_signals/amc_franco/l_log' 
		checkpoint = ModelCheckpoint(cp_save_path, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max', save_weights_only=True)


		model.fit(self.X_train, self.Y_train, validation_data=(self.X_valid,self.Y_valid), batch_size=self.batch_size,epochs=self.epochs,verbose=1, 
			callbacks=[checkpoint, TensorBoard(log_dir=tb_log_dir)])


		
		#evaluate on the best saved model
		model.load_weights(cp_save_path)

		
		acc={}
		times_s={}


		for snr in snrs:
			print("Predicting for SNR of ",snr," \n"*2)
			indices=[]
			i=0
			j=0

			for snr_mod in snr_mod_pairs_test:
				
				if (snr_mod[1] == str(snr)):
					indices.append(i)

				i=i+1

			print ("Total number test data is ", len(indices))
			if len(indices) == 0:
				print("continue")
				continue

			X_test_1=X_test[indices]
			Y_test_1=Y_test[indices]
			start = time.time()

			y_pred=model.predict(X_test_1)
			end = time.time()
			period=end-start
			period=float(period)/len(indices)
			times_s[snr]=float(period)

			y_el=[]
			y_pred_el=[]

			for i in range(1,len(y_pred)):
				y_pred_el.append(y_pred[i-1].argmax())

			for i in range(1,len(y_pred)):
				y_el.append(Y_test_1[i-1].argmax())


			cnf_matrix=confusion_matrix(y_el, y_pred_el)
			cor=np.trace(cnf_matrix)
			cor_new=np.sum(np.diag(cnf_matrix))
			sum_all=np.sum(cnf_matrix)
			acc[snr]=float(cor)/float(sum_all)

		print("\noverall accuracy is ",acc, " \n\n")
		print("inference time is ", times_s," \n\n")

		f = open("best_ind_acc.txt", "a")
		f.write("Best individual ")
		f.write("Acc is  {} ".format(acc))
		f.close()
		


	def evaluate_individual(self,indi,gen_no=1,init=False):
		complexity=-1
		accuracy=0

		#complexity=np.random.randint(1000, 100000)
		#accuracy=np.random.rand()

		#return complexity, accuracy

		random.seed(33)

		os.environ['PYTHONHASHSEED'] = str(33)
		#print(tf.__version__)
		session_conf = tf.compat.v1.ConfigProto(
			intra_op_parallelism_threads=1, 
			inter_op_parallelism_threads=1)

		sess = tf.compat.v1.Session(
			graph=tf.compat.v1.get_default_graph(), 
			config=session_conf)
		tf.compat.v1.keras.backend.set_session(sess)

		es = EarlyStopping(patience=5, verbose=1, min_delta=0.001, monitor='val_acc', mode='auto')

		model=self.build_model(indi)
		#print("indi learning rate is ", indi.learning_rate)

		if indi.learning_rate<0.5:
			opt=keras.optimizers.Adam(learning_rate=0.001)
		elif indi.learning_rate>=0.5 and indi.learning_rate<0.75:
			opt=keras.optimizers.Adam(learning_rate=0.01)
		else:
			opt=keras.optimizers.Adam(learning_rate=0.0001)


		model.compile(loss='categorical_crossentropy',optimizer=opt,metrics=['accuracy'])
		
		trainable_count = np.sum([K.count_params(w) for w in model.trainable_weights])
		non_trainable_count = np.sum([K.count_params(w) for w in model.non_trainable_weights])
		print('Total params: {:,}'.format(trainable_count + non_trainable_count))
		print('Trainable params: {:,}'.format(trainable_count))
		print('Non-trainable params: {:,}'.format(non_trainable_count))
		complexity=trainable_count+non_trainable_count
		

		if complexity>self.max_complexity or init==True: #don't waste time on super big networks
			accuracy=0
			return complexity,accuracy



		cp_save_path='/home/eperenda/multiple_signals/amc_franco/model_weights'
		tb_log_dir = '/home/eperenda/multiple_signals/amc_franco/l_log' 
		checkpoint = ModelCheckpoint(cp_save_path, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max', save_weights_only=True)

		model.fit(self.X_train, self.Y_train, validation_data=(self.X_valid,self.Y_valid), batch_size=self.batch_size,epochs=self.epochs,verbose=1, 
			callbacks=[checkpoint, TensorBoard(log_dir=tb_log_dir)])
		
		#evaluate on the best saved model
		model.load_weights(cp_save_path)
		score = model.evaluate(self.X_valid, self.Y_valid, verbose=0)

		accuracy=score[1]
		


		return complexity,accuracy


   