import matplotlib as mpl
mpl.use('Agg')

import numpy as np
import random
import time
import math
from math import sqrt, ceil
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from population import Population
from evaluate import Evaluate
from dataPreparation import get_data
from random import shuffle
from pandas.plotting import register_matplotlib_converters
from scipy.spatial.distance import directed_hausdorff

import copy
import utils

register_matplotlib_converters()


""" Optimization of CNN-based network architecture """
class r0739975:

	""" Initialize the evolutionary algorithm solver. """
	def __init__(self, X_train, Y_train, X_valid, Y_valid,X_test,Y_test,snr_mod_pairs_test, snrs,mods,num_classes,N_samples):
		self.m_prob = 0.05     		# Mutation probability
		self.rec_prob = 1			# Recombination probability
		self.flip_prob = 0.6		# flip layers between parents
		self.lambdaa = 6   			# Population size
		self.mu = self.lambdaa		# Offspring size
		self.k = 2            		# Tournament selection
		self.numIters = 10			# Maximum number of iterations
		self.batch_size = 256 		# Batch Size
		self.epochs = 10			# Training Epochs Number
		self.max_epochs = 80 		# Training Epochs Number for the best individual
		self.X_train = X_train		# Train IQ samples
		self.Y_train = Y_train		# Train one-hot encoded trained labels
		self. X_valid = X_valid		# Validation IQ samples
		self.Y_valid = Y_valid		# Validation one-hot encoded trained labels
		self.X_test = X_test		# Test IQ samples
		self.Y_test = Y_test		# Test one-hot encoded test labels
		self.snr_mod_pairs_test = snr_mod_pairs_test # snr mod pairs for test phase
		self.snrs = snrs			# snrs range
		self.mods = mods 			# mods considered
		self.max_block_width = 32 	# Block width
		self.max_block_depth = 3 	# Block depth
		self.max_nbr_blocks	= 8 	# Max Number of blocks
		self.prob_dense=0.4			# Probability to add a dense layer
		self.max_nbr_fcl = 4		# Max Number of Dense Layers
		self.num_classes=num_classes # Number of classes for AMC 
		self.N_samples=N_samples 	#Number of IQ samples
		self.population=None
		self.intMax=100000
		self.minComplexity=10000  #complexity for stopping of EA
		self.maxAcc=0.90           #accuracy for stopping of EA
		self.max_complexity=100000 #max allowed complexity for NN
		self.flip_global_pool_prob = 0.5 #flip global pooling layer in crossover
		self.pareto_sets = np.zeros((self.numIters+1,),dtype=object) # for Hausdorff distance
		self.epsilon = 10**(-4)  # difference between two pareto sets to stop EA



	""" Initialize population """
	def initialize_popualtion(self):
		print("Population Initialization size of {}...".format(self.lambdaa))
		self.population = Population(self.lambdaa, self.max_complexity,self.num_classes,self.N_samples,
			self.max_block_depth,self.max_block_width,self.max_nbr_blocks,self.max_nbr_fcl,self.prob_dense)

	"""Perform training for each individual in population"""

	def evaluate_fitness(self, gen_no):
		print("Fitness Evaluation of generation {} ...".format(gen_no))
		evaluate = Evaluate(self.population, self.X_train, self.Y_train, self.X_valid, self.Y_valid, self.num_classes,self.N_samples,self.epochs, self.batch_size,self.max_complexity)
		evaluate.evaluate_population(gen_no)

		
		print(self.population)

	def dominatedFitnessWrapper(self, X):
		pop_size=self.population.get_pop_size()
		x_size=len(X)
		dominatedCounts=np.zeros(x_size)

		for i in range(x_size):
			x_ind=X[i]
			#print("ind at ",i)
			for j in range(pop_size):
				#print("pop at ",j)
				y_ind=self.population.get_individual_at(j)
				if (y_ind.accuracy>x_ind.accuracy and y_ind.complexity<=x_ind.complexity) or \
					(y_ind.accuracy>=x_ind.accuracy and y_ind.complexity<x_ind.complexity) :
					dominatedCounts[i]+=1

					#print("dominated counts ", dominatedCounts)


		#print("dominated counts ", dominatedCounts)
		return dominatedCounts

	def checkStopCriterium(self):
		for i in range(self.population.get_pop_size()):
			ind_x=self.population.get_individual_at(i)
			if ind_x.accuracy>self.maxAcc and ind_x.complexity<self.minComplexity:
				return True

		return False

	#averaged Hausdorff controls outliers penality by factor p, if p is inf than it is equal to direct Hausdorff	

	def checkImprovementStep_avg(self,gen_no):
		if gen_no<3:
			return True

		f = open("generations_improvment.txt", "a")
		

		hd_c=utils.average_hausdorff_distance(self.pareto_sets[gen_no], self.pareto_sets[gen_no-1])

		f.write("Diff betwen generation {}, and {} is {}".format(gen_no,gen_no-1,hd_c))

		hd_c_1=utils.average_hausdorff_distance(self.pareto_sets[gen_no-1], self.pareto_sets[gen_no-2])

		f.write("Diff betwen generation {}, and {} is {}".format(gen_no-1,gen_no-2,hd_c_1))

		#hd_c_2=utils.average_hausdorff_distance(self.pareto_sets[gen_no-2], self.pareto_sets[gen_no-3])

		#print("Diff betwen generation {}, and {} is {}".format(gen_no-2,gen_no-3,hd_c_2))

		f.close()

		if(hd_c<self.epsilon and hd_c_1<self.epsilon):
			return False

		return True

	#use directed Hausdorff distance - problem with outliers
	def checkImprovementStep(self,gen_no):
		if gen_no<3:
			return True

		hd_c=max(directed_hausdorff(self.pareto_sets[gen_no], self.pareto_sets[gen_no-1])[0],\
		 directed_hausdorff(self.pareto_sets[gen_no-1], self.pareto_sets[gen_no])[0])

		print("Diff betwen generation {}, and {} is {}".format(gen_no,gen_no-1,hd_c))

		hd_c_1=max(directed_hausdorff(self.pareto_sets[gen_no-1], self.pareto_sets[gen_no-2])[0],\
		 directed_hausdorff(self.pareto_sets[gen_no-2], self.pareto_sets[gen_no-1])[0])

		print("Diff betwen generation {}, and {} is {}".format(gen_no-1,gen_no-2,hd_c_1))

		hd_c_2=max(directed_hausdorff(self.pareto_sets[gen_no-2], self.pareto_sets[gen_no-3])[0],\
		 directed_hausdorff(self.pareto_sets[gen_no-3], self.pareto_sets[gen_no-2])[0])

		print("Diff betwen generation {}, and {} is {}".format(gen_no-2,gen_no-3,hd_c_2))

		if(hd_c<self.epsilon and hd_c_1<self.epsilon and hd_c_2<self.epsilon):
			return False

		return True

	def save_best(self):
		print("Saving the best for later testing phase")
		max_acc=0
		idx=-1
		for i in range(self.population.get_pop_size()):
			ind_x=self.population.get_individual_at(i)
			if ind_x.accuracy>max_acc:
				max_acc=ind_x.accuracy
				idx=i

		if idx !=-1:
			best_ind=self.population.get_individual_at(idx)
			best_pop=Population(0)
			best_pop.append_individual(best_ind)
			evaluate = Evaluate(best_pop, self.X_train, self.Y_train, self.X_valid, self.Y_valid, self.num_classes,self.N_samples,self.max_epochs, self.batch_size,self.max_complexity)
			evaluate.evaluate_best(best_ind,self.X_test,self.Y_test,self.snr_mod_pairs_test,self.snrs,self.mods)
			print("The best individual is ",str(best_ind))


		return best_ind

	def update_pareto_sets(self,gen_no):
		set_x=np.zeros((self.lambdaa,2))

		for i in range(self.population.get_pop_size()):
			ind=self.population.get_individual_at(i)
			set_x[i,0]=1.0-ind.accuracy
			set_x[i,1]=ind.complexity/10000.0
			
		self.pareto_sets[gen_no]=set_x

		#print("pareto sets ",self.pareto_sets)



	""" The main evolutionary algorithm loop. """
	def optimize( self, plotFun = lambda x : None, plotMutation = lambda x : None):

		# Initialize population
		self.initialize_popualtion()
		self.evaluate_fitness(0)
		f = open("generations.txt", "a")
		f.write("Generation 0")
		f.write("Initialized inidivudulas: "+str(self.population))

		self.update_pareto_sets(0)

		plotFun((self.population, 0))
		for i in range(self.numIters):

			print('The {}/{} generation'.format(i+1, self.numIters))
			print("Population size ",self.population.get_pop_size())
			f.write('The {}/{} generation'.format(i+1, self.numIters))

			# The evolutionary algorithm
			start = time.time()
			selected = self.selection(self.population, self.k)
			self.recombinate(selected,i)
			

			print("before elimination ")
			f.write("\n\n\n******Before elimination********\n\n "+str(self.population))
			#plotMutation((self.population,i+1))
			
			f.write(self.local_search())


			#self.elimination(self.lambdaa)
			self.eliminationWithFitnessSharing(self.lambdaa)
			print('The {}/{} generation'.format(i+1, self.numIters))
			#print("after elimination ",self.population)
			f.write("\n\n\n******After elimination********\n\n "+str(self.population))
			self.update_pareto_sets(i+1)

			

			if(self.checkStopCriterium()==True):
				f.write("\n\nTarget complexity and accuracy reached, Stop EA!")
				break

			if(self.checkImprovementStep_avg(i+1)==False):
				f.write("\n\nNo improvement over X iterations, no sense to continue. Stop EA!")
				break
			itT = time.time() - start

			f.write("\n\n time is {} \n".format(itT))

			# Show progress
			plotFun((self.population, i+1))
			

		
		f.write("\n\n*****Best individual finding ******\n\n")

		best_ind=self.save_best()

		f.write("\n\n*****Best individual is ******\n\n"+str(best_ind))

		f.close()
		print('Done')

	""" Perform k-tournament selection to select pairs of parents. """
	def selection(self, population, k):
		selected = []
		pop_size=population.get_pop_size()
		#print("pop size is ", pop_size)
		for ii in range( int(self.mu/2) ):
			all_choices=list(range(pop_size))
			shuffle(all_choices)
			ri=all_choices[:self.k]

			#ri = random.choices(range(pop_size), k = self.k)
			#print("ri ",ri)
			X=population.get_individual_subset(ri)

			min_in = np.argmin( self.dominatedFitnessWrapper(X) )
			#print("min is ", min, "ri ", ri[min])
			selected.append(population.get_individual_at(ri[min_in]))

			without_replacement=True
			while without_replacement == True:
				all_choices=list(range(pop_size))
				shuffle(all_choices)
				ri=all_choices[:self.k]

				#ri = random.choices(range(pop_size), k = self.k)
				#print("ri ",ri)
				X=population.get_individual_subset(ri)

				min_in2 = np.argmin( self.dominatedFitnessWrapper(X) )
				if min_in2 != min_in:
					#print("min is ", min, "ri ", ri[min])
					selected.append(population.get_individual_at(ri[min_in2]))
					without_replacement=False

		return selected


	def recombinate(self,X,gen_no):
		print("mutation and crossover of selected population....")
		offspring_list=[]

		#print("len X is ",len(X))
		
		for i in range(int(len(X)/2)):
			p1=X[2*i]
			p2=X[2*i+1]
			#print("parent 1: ",p1)
			#print("parent 2 ",p2)
			#crossover

			#modified one-point crossover
			#offspring1, offspring2=self.crossover(p1,p2)

			#uniform crossover
			offspring1, offspring2,nodone_cr=self.uniform_crossover(p1,p2)

			#set mutation rates for offsprings
			gama=0.22

			offspring1.m_prob=(1+(1-p1.m_prob)/p1.m_prob*math.exp(-gama*np.random.randn()))**(-1)
			offspring2.m_prob=(1+(1-p2.m_prob)/p2.m_prob*math.exp(-gama*np.random.randn()))**(-1)

			#offspring1.m_prob=1
			#offspring2.m_prob=1
			#print("offspring 1 ",offspring1)
			#print("offspring 2 ",offspring2)

			#mutation
			print("Mutation is performing")
			if nodone_cr==True:
				offspring1.mutation(self.m_prob)
				offspring2.mutation(self.m_prob)
			else:
				offspring1.mutation(1)
				offspring2.mutation(1)


			#print("offspring 1 ",offspring1)
			#print("offspring 2 ",offspring2)

			offspring_list.append(offspring1)
			offspring_list.append(offspring2)

		offspring_pops=Population(0)
		offspring_pops.set_populations(offspring_list)

		evaluate=Evaluate(offspring_pops,self.X_train,self.Y_train,self.X_valid, self.Y_valid, self.num_classes,self.N_samples,self.epochs, self.batch_size,self.max_complexity)
		evaluate.evaluate_population(gen_no)
		
		self.population.extend_population(offspring_pops)



	def uniform_crossover(self,p11,p21):
		p1 = copy.deepcopy(p11)
		p2 = copy.deepcopy(p21)

		p1_no_crossover=copy.deepcopy(p11)
		p2_no_crossover=copy.deepcopy(p21)
		#print("parent 1 ",p1)

		p1.clear_state_info()
		p2.clear_state_info()

		p1_no_crossover.clear_state_info()
		p2_no_crossover.clear_state_info()
		#print("parent 2 ",p2)

		#unit alignment

		#for different unit, we define two list, one to save their index and the other one save unit
		p1_conv_index_list = []
		p1_conv_layer_list = []
		p1_pool_index_list = []
		p1_pool_layer_list = []
		p1_dense_index_list = []
		p1_dense_layer_list = []
		p1_block_layer_list = []
		p1_block_index_list = []
		p1_global_pool_layer = ""
		p1_global_pool_index = 0

		p2_conv_index_list = []
		p2_conv_layer_list = []
		p2_pool_index_list = []
		p2_pool_layer_list = []
		p2_dense_index_list = []
		p2_dense_layer_list = []
		p2_block_layer_list = []
		p2_block_index_list = []
		p2_global_pool_layer = ""
		p2_global_pool_index = 0

		for i in range(p1.get_layer_size()):
			#print("layer in ",i)
			unit = p1.get_layer_at(i)
			#print("unit",unit)
			if unit.type == 0:
				p1_block_index_list.append(i)
				p1_block_layer_list.append(unit)
			elif unit.type == 1:
				p1_conv_index_list.append(i)
				p1_conv_layer_list.append(unit)
			elif unit.type == 2:
				p1_pool_index_list.append(i)
				p1_pool_layer_list.append(unit)
			elif unit.type == 3:
				p1_dense_index_list.append(i)
				p1_dense_layer_list.append(unit)
			elif unit.type == 6 or unit.type == 7 or unit.type == 8:
				p1_global_pool_index=i
				p1_global_pool_layer=copy.deepcopy(unit)

		for i in range(p2.get_layer_size()):
			unit = p2.get_layer_at(i)
			if unit.type == 0:
				p2_block_index_list.append(i)
				p2_block_layer_list.append(unit)
			elif unit.type == 1:
				p2_conv_index_list.append(i)
				p2_conv_layer_list.append(unit)
			elif unit.type == 2:
				p2_pool_index_list.append(i)
				p2_pool_layer_list.append(unit)
			elif unit.type == 3:
				p2_dense_index_list.append(i)
				p2_dense_layer_list.append(unit)
			elif unit.type == 6 or unit.type == 7 or unit.type == 8:
				p2_global_pool_index=i
				p2_global_pool_layer=copy.deepcopy(unit)

		
		#begin uniform crossover on blocks layer
		l = min(len(p1_block_layer_list), len(p2_block_layer_list))
		nn=[]

		if l>0:
			nn=np.random.rand(l)
		
		for i,prob in enumerate(nn):
			if prob<self.flip_prob:
				unit_p1 = copy.deepcopy(p1_block_layer_list[i])
				unit_p2 = copy.deepcopy(p2_block_layer_list[i])

				tmp=copy.deepcopy(unit_p1)
				p1_block_layer_list[i]=copy.deepcopy(unit_p2)
				p2_block_layer_list[i]=copy.deepcopy(tmp)

		#begin one-point crossover on conv layer
		l = min(len(p1_conv_layer_list), len(p2_conv_layer_list))
		nn=[]

		if l>0:
			nn=np.random.rand(l)

		for i,prob in enumerate(nn):
			if prob<self.flip_prob:
				unit_p1 = copy.deepcopy(p1_conv_layer_list[i])
				unit_p2 = copy.deepcopy(p2_conv_layer_list[i])

				tmp=copy.deepcopy(unit_p1)
				p1_conv_layer_list[i]=copy.deepcopy(unit_p2)
				p2_conv_layer_list[i]=copy.deepcopy(tmp)

		#begin one-point crossover on pool layer
		l = min(len(p1_pool_layer_list), len(p2_pool_layer_list))
		nn=[]

		if l>0:
			nn=np.random.rand(l)

		for i,prob in enumerate(nn):
			if prob<self.flip_prob:
				unit_p1 = copy.deepcopy(p1_pool_layer_list[i])
				unit_p2 = copy.deepcopy(p2_pool_layer_list[i])

				tmp=copy.deepcopy(unit_p1)
				p1_pool_layer_list[i]=copy.deepcopy(unit_p2)
				p2_pool_layer_list[i]=copy.deepcopy(tmp)

		#begin one-point crossover on dense layer
		l = min(len(p1_dense_layer_list), len(p2_dense_layer_list))
		nn=[]

		if l>0:
			nn=np.random.rand(l)

		for i,prob in enumerate(nn):
			if prob<self.flip_prob:
				unit_p1 = copy.deepcopy(p1_dense_layer_list[i])
				unit_p2 = copy.deepcopy(p2_dense_layer_list[i])

				tmp=copy.deepcopy(unit_p1)
				p1_dense_layer_list[i]=copy.deepcopy(unit_p2)
				p2_dense_layer_list[i]=copy.deepcopy(tmp)

		
		p1_units = p1.indi

		# assign these crossovered values to the p1 and p2
		for i in range(len(p1_block_index_list)):
			p1_units[p1_block_index_list[i]] = copy.deepcopy(p1_block_layer_list[i])

		for i in range(len(p1_conv_index_list)):
			p1_units[p1_conv_index_list[i]] = copy.deepcopy(p1_conv_layer_list[i])

		for i in range(len(p1_pool_index_list)):
			p1_units[p1_pool_index_list[i]] = copy.deepcopy(p1_pool_layer_list[i])

		for i in range(len(p1_dense_index_list)):
			p1_units[p1_dense_index_list[i]] = copy.deepcopy(p1_dense_layer_list[i])

		if np.random.rand()<self.flip_global_pool_prob:
			p1_units[p1_global_pool_index]=copy.deepcopy(p2_global_pool_layer)
		
		
		p1.indi = p1_units

		p2_units = p2.indi

		# assign these crossovered values to the p1 and p2
		for i in range(len(p2_block_index_list)):
			p2_units[p2_block_index_list[i]] = copy.deepcopy(p2_block_layer_list[i])
		for i in range(len(p2_conv_index_list)):
			p2_units[p2_conv_index_list[i]] = copy.deepcopy(p2_conv_layer_list[i])
		for i in range(len(p2_pool_index_list)):
			p2_units[p2_pool_index_list[i]] = copy.deepcopy(p2_pool_layer_list[i])
		for i in range(len(p2_dense_index_list)):
			p2_units[p2_dense_index_list[i]] = copy.deepcopy(p2_dense_layer_list[i])

		if np.random.rand()<self.flip_global_pool_prob:
			p2_units[p2_global_pool_index]=copy.deepcopy(p1_global_pool_layer)

		p2.indi = p2_units

		c_1=np.random.rand()
		c_2=np.random.rand()

		avg_c_prob=(p11.c_prob+p21.c_prob)/2.0

		if c_1<=avg_c_prob:
			return p1, p2,True


		# if c_1<=p11.c_prob and c_2<=p21.c_prob:
		# 	return p1, p2

		# if c_1>p11.c_prob and c_2<=p21.c_prob:
		# 	return p1_no_crossover, p2

		# if c_1<=p11.c_prob and c_2>p21.c_prob:
		# 	return p1, p2_no_crossover

		print("returning the same parents")
		return p1_no_crossover, p2_no_crossover, False

	def crossover(self, p11, p21):
		p1 = copy.deepcopy(p11)
		p2 = copy.deepcopy(p21)
		#print("parent 1 ",p1)

		p1.clear_state_info()
		p2.clear_state_info()
		#print("parent 2 ",p2)

		#for different unit, we define two list, one to save their index and the other one save unit
		p1_conv_index_list = []
		p1_conv_layer_list = []
		p1_pool_index_list = []
		p1_pool_layer_list = []
		p1_dense_index_list = []
		p1_dense_layer_list = []
		p1_block_layer_list = []
		p1_block_index_list = []
		p1_global_pool_layer = ""
		p1_global_pool_index = 0

		p2_conv_index_list = []
		p2_conv_layer_list = []
		p2_pool_index_list = []
		p2_pool_layer_list = []
		p2_dense_index_list = []
		p2_dense_layer_list = []
		p2_block_layer_list = []
		p2_block_index_list = []
		p2_global_pool_layer = ""
		p2_global_pool_index = 0

		for i in range(p1.get_layer_size()):
			#print("layer in ",i)
			unit = p1.get_layer_at(i)
			#print("unit",unit)
			if unit.type == 0:
				p1_block_index_list.append(i)
				p1_block_layer_list.append(unit)
			elif unit.type == 1:
				p1_conv_index_list.append(i)
				p1_conv_layer_list.append(unit)
			elif unit.type == 2:
				p1_pool_index_list.append(i)
				p1_pool_layer_list.append(unit)
			elif unit.type == 3:
				p1_dense_index_list.append(i)
				p1_dense_layer_list.append(unit)
			elif unit.type == 6 or unit.type == 7 or unit.type == 8:
				p1_global_pool_index=i
				p1_global_pool_layer=copy.deepcopy(unit)

		for i in range(p2.get_layer_size()):
			unit = p2.get_layer_at(i)
			if unit.type == 0:
				p2_block_index_list.append(i)
				p2_block_layer_list.append(unit)
			elif unit.type == 1:
				p2_conv_index_list.append(i)
				p2_conv_layer_list.append(unit)
			elif unit.type == 2:
				p2_pool_index_list.append(i)
				p2_pool_layer_list.append(unit)
			elif unit.type == 3:
				p2_dense_index_list.append(i)
				p2_dense_layer_list.append(unit)
			elif unit.type == 6 or unit.type == 7 or unit.type == 8:
				p2_global_pool_index=i
				p2_global_pool_layer=copy.deepcopy(unit)

		
		#begin one-point crossover on blocks layer
		l = min(len(p1_block_layer_list), len(p2_block_layer_list))
		if l>0:
			nn=np.random.randint(0,l)
		else:
			nn=0
		for i in range(nn):
			unit_p1 = copy.deepcopy(p1_block_layer_list[i])
			unit_p2 = copy.deepcopy(p2_block_layer_list[i])
			if np.random.rand()<self.rec_prob:
				tmp=copy.deepcopy(unit_p1)
				p1_block_layer_list[i]=copy.deepcopy(unit_p2)
				p2_block_layer_list[i]=copy.deepcopy(tmp)

		#begin one-point crossover on conv layer
		l = min(len(p1_conv_layer_list), len(p2_conv_layer_list))
		if l>0:
			nn=np.random.randint(0,l)
		else:
			nn=0
		for i in range(nn):
			unit_p1 = copy.deepcopy(p1_conv_layer_list[i])
			unit_p2 = copy.deepcopy(p2_conv_layer_list[i])
			if np.random.rand()<self.rec_prob:
				tmp=copy.deepcopy(unit_p1)
				p1_conv_layer_list[i]=copy.deepcopy(unit_p2)
				p2_conv_layer_list[i]=copy.deepcopy(tmp)

		#begin one-point crossover on pool layer
		l = min(len(p1_pool_layer_list), len(p2_pool_layer_list))
		if l>0:
			nn=np.random.randint(0,l)
		else:
			nn=0
		for i in range(nn):
			unit_p1 = copy.deepcopy(p1_pool_layer_list[i])
			unit_p2 = copy.deepcopy(p2_pool_layer_list[i])
			if np.random.rand()<self.rec_prob:
				tmp=copy.deepcopy(unit_p1)
				p1_pool_layer_list[i]=copy.deepcopy(unit_p2)
				p2_pool_layer_list[i]=copy.deepcopy(tmp)

		#begin one-point crossover on dense layer
		l = min(len(p1_dense_layer_list), len(p2_dense_layer_list))
		if l>0:
			nn=np.random.randint(0,l)
		else:
			nn=0
		for i in range(nn):
			unit_p1 = copy.deepcopy(p1_dense_layer_list[i])
			unit_p2 = copy.deepcopy(p2_dense_layer_list[i])
			if np.random.rand()<self.rec_prob:
				tmp=copy.deepcopy(unit_p1)
				p1_dense_layer_list[i]=copy.deepcopy(unit_p2)
				p2_dense_layer_list[i]=copy.deepcopy(tmp)

		
			

		p1_units = p1.indi

		# assign these crossovered values to the p1 and p2
		for i in range(len(p1_block_index_list)):
			p1_units[p1_block_index_list[i]] = copy.deepcopy(p1_block_layer_list[i])

		for i in range(len(p1_conv_index_list)):
			p1_units[p1_conv_index_list[i]] = copy.deepcopy(p1_conv_layer_list[i])

		for i in range(len(p1_pool_index_list)):
			p1_units[p1_pool_index_list[i]] = copy.deepcopy(p1_pool_layer_list[i])

		for i in range(len(p1_dense_index_list)):
			p1_units[p1_dense_index_list[i]] = copy.deepcopy(p1_dense_layer_list[i])

		if np.random.rand()<self.flip_global_pool_prob:
			p1_units[p1_global_pool_index]=copy.deepcopy(p2_global_pool_layer)
		
		
		p1.indi = p1_units

		p2_units = p2.indi

		# assign these crossovered values to the p1 and p2
		for i in range(len(p2_block_index_list)):
			p2_units[p2_block_index_list[i]] = copy.deepcopy(p2_block_layer_list[i])
		for i in range(len(p2_conv_index_list)):
			p2_units[p2_conv_index_list[i]] = copy.deepcopy(p2_conv_layer_list[i])
		for i in range(len(p2_pool_index_list)):
			p2_units[p2_pool_index_list[i]] = copy.deepcopy(p2_pool_layer_list[i])
		for i in range(len(p2_dense_index_list)):
			p2_units[p2_dense_index_list[i]] = copy.deepcopy(p2_dense_layer_list[i])

		if np.random.rand()<self.flip_global_pool_prob:
			p2_units[p2_global_pool_index]=copy.deepcopy(p1_global_pool_layer)

		p2.indi = p2_units

		return p1, p2

	def distances(self, x, pop):
		distances=np.zeros(len(pop))
		dist_acc=np.zeros(len(pop))
		max_complexity=x.complexity

		for i in range(len(pop)):
			ind_=pop[i]
			if ind_.complexity>max_complexity:
				max_complexity=ind_.complexity

			diff_complexity=np.abs(ind_.complexity-x.complexity)
			diff_acc=np.abs(ind_.accuracy-x.accuracy)
			#diff_acc2=np.abs(1.0-ind_.accuracy)
			#diff_acc1=np.abs(1.0-x.accuracy)
			#diff_acc=diff_acc1+diff_acc2
			#num_block1,num_conv1, num_dense1,num_pool1=ind_.get_num_types()
			#num_block2,num_conv2, num_dense2,num_pool2=x.get_num_types()
			#diff_layers_type=np.abs(num_block1-num_block2)+np.abs(num_conv1-num_conv2)+np.abs(num_dense1-num_dense2)+np.abs(num_pool1-num_pool2)

			#distances[i]=diff_complexity+diff_layers_type
			distances[i]=diff_complexity
			dist_acc[i]=diff_acc
			

		max_complexity=100000
		distances=np.divide(distances,max_complexity)
		

		mix_distances=np.zeros(len(pop))
		for i in range(len(pop)):
			mix_distances[i]=sqrt(distances[i]**2+dist_acc[i]**2)
			#mix_distances[i]=(distances[i]**4+dist_acc[i]**4)**(1.0/4.0)
			#mix_distances[i]=distances[i]+dist_acc[i]

		return mix_distances

	#apply local search only for the individuals that are not dominated by any other
	def local_search(self):
		dominatedCounts=self.dominatedFitnessWrapper(self.population.Population)
		min_d=0
		max_d=np.amax(dominatedCounts)
		avg_d=ceil(max_d/2.0)

		str_=[]
		for i in range(self.population.get_pop_size()):
			if dominatedCounts[i] == max_d or dominatedCounts[i] == 0:
				ind_=self.population.get_individual_at(i)
				k=2
				acc_x=np.zeros((k,))
				ind_best=copy.deepcopy(ind_)
				str_.append("dominated co is 0, (acc,complexity) is ({},{}) ".format(ind_.accuracy, ind_.complexity))
				for j in range(k):
					ind_k=copy.deepcopy(ind_)
					ind_k.mutation(1)
					lo_pop=Population(0)
					lo_pop.append_individual(ind_k)
					evaluate = Evaluate(lo_pop, self.X_train, self.Y_train, self.X_valid, self.Y_valid, self.num_classes,self.N_samples,self.epochs, self.batch_size,self.max_complexity)
					evaluate.evaluate_population(550)
					ind_kl=lo_pop.get_individual_at(0)
					str_.append("mutated {}, (acc,complexity) is ({},{}) ".format(j,ind_kl.accuracy, ind_kl.complexity))
				
					if (ind_kl.accuracy>ind_best.accuracy and ind_kl.complexity<=(1.1*ind_best.complexity)) or \
						(ind_kl.accuracy>=(0.9995*ind_best.accuracy) and ind_kl.complexity<ind_best.complexity) :
						ind_best=copy.deepcopy(ind_kl)
						str_.append("beter mutated")

				self.population.set_individual_at(i,ind_best)

		return '\n'.join(str_)



	def sharedFitnessWrapper(self,X,pop):
		alpha=1.0
		#sigma=self.intMax*0.1
		sigma=0.1
		modAcc=np.zeros(X.get_pop_size())

		for i in range(X.get_pop_size()):
			x=X.get_individual_at(i)
			ds=self.distances(x,pop)
			#print("\n\nin function distances ",ds)

			onePlusBeta=0
			for j,d in enumerate(ds):
				if d<=sigma:
					onePlusBeta+=(1-(d/sigma)**alpha)

			if onePlusBeta!=0:
				modAcc[i]=(1-x.accuracy)*onePlusBeta
			else:
				modAcc[i]=1-x.accuracy

		#print("mod acc is ",modAcc)

		return modAcc



	""" Select  the keep the best"""
	def elimination(self, keep):
		accuracies=[]
		for i in range(self.population.get_pop_size()):
			indi=self.population.get_individual_at(i)
			accuracies.append(indi.accuracy)

		print("accuracies are ",accuracies)
		neg_acc=np.multiply(np.array(accuracies),-1.0)
		perm = np.argsort(neg_acc)
		print("perm sur ",perm)
		survivors=self.population.get_individual_subset(perm[0:keep])

		self.population.set_populations(survivors)

	def eliminationWithFitnessSharing(self, keep):
		survivors=[]
		#add first the best acc
		# max_acc=0.0
		# ind_best=0

		# for i in range(self.population.get_pop_size()):
		# 	indi=self.population.get_individual_at(i)
		# 	if(indi.accuracy>max_acc):
		# 		max_acc=indi.accuracy
		# 		ind_best=i

		# print("The best has ", max_acc)

		#survivors.append(self.population.get_individual_at(ind_best))
		for i in range(keep):
			modAcc=self.sharedFitnessWrapper(self.population,survivors[0:i])

			idx=np.argmin(modAcc)
			survivors.append(self.population.get_individual_at(idx))
			self.population.remove_individual_at(idx)


		self.population.set_populations(survivors)
		
"""
Make a 2D visualization of the optimization landscape
"""
def plotPopulation2D(input):
	
	population=input[0]
	gen_no=input[1]
	pop_size=population.get_pop_size()

	fig=plt.figure(figsize=(15,10))

	f = open("generations_acc.txt", "a")
	f.write("Generation {}: ".format(gen_no))

	for i in range(pop_size):
		ind=population.get_individual_at(i)
		f.write("({},{}),".format(ind.accuracy,ind.complexity))
		plt.plot(1-ind.accuracy,ind.complexity/10000.0,'o',color='blue')



	f.write("\n\n")
	f.close()

	plt.xlabel('Accuracy Error')
	plt.ylabel('Complexity/10000')
	plt.xticks(fontsize=16)
	plt.yticks(fontsize=16)
	plt.rcParams["font.family"] = "Times New Roman"
	plt.rcParams["font.size"] = "18"

	plt.savefig("./images/pareto_gen"+str(gen_no)+".png")


"""
Make a 2D visualization of the mutation rates per generations
"""
def plotMutationRate(input):
	
	population=input[0]
	gen_no=input[1]
	pop_size=population.get_pop_size()

	fig=plt.figure(figsize=(15,10))
	f = open("generations_mutationrate.txt", "a")
	f.write("Generation {}: ".format(gen_no))
	

	for i in range(pop_size):
		ind=population.get_individual_at(i)
		f.write("{},".format(ind.m_prob))
		plt.plot(i,ind.m_prob,'o',color='blue')
		plt.plot(i,ind.c_prob,'o',color='red')

	plt.xticks(fontsize=16)
	plt.yticks(fontsize=16)
	plt.rcParams["font.family"] = "Times New Roman"
	plt.rcParams["font.size"] = "18"

	f.write("\n\n")
	f.close()

	plt.xlabel('Individual Index')
	plt.ylabel('Mutation Rate')
	plt.savefig("./images/mutation_gen"+str(gen_no)+".png")




X_train,Y_train,X_valid,Y_valid,X_test, Y_test, snrs, mods,snr_mod_pairs_test=get_data()

print("\n\nTrain datasets have shapes such:")
print("Train Input dataset: ",X_train.shape)
print("Train Output dataset: ", Y_train.shape)
print("Valid Input dataset: ", X_valid.shape)
print("Valid Output dataset: ", Y_valid.shape)
print("snr mod paris ",snr_mod_pairs_test.shape)

print("\n\n")

start = time.time()
dnn_opt = r0739975(X_train,Y_train,X_valid,Y_valid,X_test, Y_test, snr_mod_pairs_test,snrs,mods,len(mods),128)

dnn_opt.optimize(plotPopulation2D,plotMutationRate)

end = time.time()
period=end-start
print('total time is ',period)