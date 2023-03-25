import numpy as np
import random
import copy

from layers import ConvLayer, PoolLayer, DenseLayer, Block, IdentityBranch, MergeLayer, FlattenLayer, GlobalMaxPoolingLayer, GlobalAveragePoolingLayer
                

class Individual:
    def __init__(self, max_block_depth=5, max_block_width=32,max_nbr_blocks=10, max_nbr_fcl=4,prob_dense=0.2, max_length=20):
        self.indi = []

        self.max_block_depth=max_block_depth
        self.max_block_width=max_block_width
        self.max_nbr_blocks=max_nbr_blocks
        self.max_nbr_fcl=max_nbr_fcl
        self.prob_dense=prob_dense
        #self.m_prob=np.random.randint(5,20)/100.0 #random between 0.001 and 0.2
        self.m_prob=np.random.randint(50,100)/100.0 #random between 0.5 and 1.0
        #self.m_prob=1
        self.c_prob=1.0


        self.accuracy=0
        self.complexity=0
        self.max_length=max_length


        #####################
        self.filters_range=[2,8] #num of filters is 2^ this range ie [4,8,16,32,64,128] 
        self.filters_weights_first_conv=(5,5,10,30,30,20)
        self.filters_weights_default=(16,17,17,17,17,16)
        self.filters_weights=(16,17,17,17,17,16)

        #self.kernel_size_range=[1,3,5,7]
        self.kernel_size_range=[1,3,5]
        self.pool_kernel_size_range=[2]
        self.hidden_neurons_range=[32,256]
        self.learning_rate=0.2 #np.random.rand() #[0-0.5 means 0.001, 0.5-0.75 means 0.01, otherwise 0.0001]
        self.activations=['relu','selu','tanh','linear']
        self.identity_dimension_reduction_prob=0.2 # apply conv layer only on identity branch if less 0f 0.2


    def clear_state_info(self):
        self.complexity = 0
        self.accuracy = 0


    def initialize(self):
        self.indi = self.init_one_individual()

    def get_num_types(self):
        num_blocks=0
        num_dense=0
        num_conv=0
        num_pool=0
        for i in range(self.get_layer_size()):
            lay_i=self.get_layer_at(i)
            if lay_i.type==0:
                num_blocks+=(lay_i.width*lay_i.depth+1)
            elif lay_i.type==1:
                num_conv+=1
            elif lay_i.type==2:
                num_pool+=1
            elif lay_i.type==3:
                num_dense+=1

        return num_blocks,num_conv, num_dense,num_pool

    def determine_filters_weights(self,width):

        ind=range(self.filters_range[0],self.filters_range[1])
        filters_val=np.power(2,ind)
        merged_filters_val=np.multiply(filters_val,width)

        filters_above_max=np.argwhere(merged_filters_val>filters_val[-1]*1.1)
        filters_below_max=np.argwhere(merged_filters_val<=filters_val[-1]*1.1)

        len_all=merged_filters_val.shape[0]
        len_invalid=filters_above_max.shape[0]

        prob_val=100.0/(len_all-len_invalid)

        filters_weights=np.zeros(len_all)

        for i in filters_above_max:
            filters_weights[i]=0

        for i in filters_below_max:
            filters_weights[i]=prob_val
        
        return filters_weights


    def init_one_individual(self):

        init_num_block=np.random.randint(1,self.max_nbr_blocks+1)
        init_num_fcl=0

        if np.random.rand()<self.prob_dense: 
            init_num_fcl=np.random.randint(0,self.max_nbr_fcl+1)
        

        #print("\n\nNumber of blocks is ",init_num_block," and number of Dense layers is ", init_num_fcl, "\n\n")
        _list=[]

        self.filters_weights=self.filters_weights_first_conv
        first_conv=self.add_a_random_conv_layer()

        _list.append(first_conv)
        if (np.random.rand()<0.5):
            _list.append(self.add_a_random_pool_layer())

        for i in range(init_num_block):
            #init_depth=np.random.randint(1,self.max_block_depth+1)
            #init_width=np.random.randint(1,self.max_block_width+1)
            init_width=1
            init_depth=1
            #print("\n\nBlock {0}, width = {1} and depth = {2} \n ".format(i, init_width, init_depth))
            
            _list.append(self.add_a_random_block(init_width,init_depth))

            if (np.random.rand()<0.5):
                pooling_layer=self.add_a_random_pool_layer()
                _list.append(pooling_layer)
            
            
            

        global_pool=np.random.rand()

        if global_pool<0.5:
            _list.append(self.add_a_globalavg_layer())
        #elif global_pool<0.6:
        #    _list.append(self.add_a_globalmax_layer())
        else:
            _list.append(self.add_a_flatten_layer())


        for _ in range(init_num_fcl):
            _list.append(self.add_a_random_dense_layer())

        return _list

    def add_a_random_dense_layer(self):
        units = np.random.randint(self.hidden_neurons_range[0], self.hidden_neurons_range[1]+1)
        activation=np.random.rand() # 0-0.25 Relu, 0.25-0.5 Selu, 0.5-0.75 Tanh, 0.75-1 Linear

        dense_layer = DenseLayer(units,activation)
        return dense_layer

    def add_a_flatten_layer(self):
        flatten_layer=FlattenLayer()
        return flatten_layer

    def add_a_globalmax_layer(self):
        global_max_pool=GlobalMaxPoolingLayer()
        return global_max_pool

    def add_a_globalavg_layer(self):
        global_avg_pool=GlobalAveragePoolingLayer()
        return global_avg_pool

    def add_a_random_pool_layer(self):
        
        kernel_size_id=np.random.randint(0,len(self.pool_kernel_size_range))
        kernel_size=self.pool_kernel_size_range[kernel_size_id]
        
        pool_type=np.random.rand() # values below 0.5 means Max other wise Average
        pool_layer = PoolLayer(kernel_size, pool_type)
        return pool_layer

    def add_a_random_conv_layer(self,filters=-1):
 
        #kernel_size = np.random.randint(self.kernel_size_range[0],self.kernel_size_range[1]+1)
        kernel_size_id=np.random.randint(0,len(self.kernel_size_range))
        kernel_size=self.kernel_size_range[kernel_size_id]

        if filters==-1:
            filter_ind=random.choices(range(self.filters_range[0],self.filters_range[1]),weights=self.filters_weights,k=1)
            #print("filters ", filter_ind)
            filters=2**filter_ind[0]

        activation=np.random.rand() # 0-0.25 Relu, 0.25-0.5 Selu, 0.5-0.75 Tanh, 0.75-1 Linear


        conv_layer = ConvLayer(kernel_size, filters, activation)

        return conv_layer

    def add_a_random_merge_layer(self):
        merge_function=np.random.rand() #0-0.33 Concat, 0.3-0.66 Multiply, 0.66-1 Add
        #merge_function=0.8
        merge_layer=MergeLayer(merge_function)
        return merge_layer

    def add_a_random_block(self, width, depth):

        #If width is equal to 
        if width==1 and depth==1:
            self.filters_weights=self.filters_weights_first_conv
            block=self.add_a_random_conv_layer()
            return block

        _list=[]

        
        filters_last=1

        self.filters_weights=self.determine_filters_weights(width)

        for _ in range(depth):
            cnn_layer=self.add_a_random_conv_layer()
            filters_last=cnn_layer.filters
            _list.append(cnn_layer)

        merge_layer=self.add_a_random_merge_layer()
        pooling_layer=None
        #add a pool layer after block with 50%
        if (np.random.rand()<0.5):
            pooling_layer=self.add_a_random_pool_layer()

        identity_apply_dimension_reduction=np.random.rand()
        

        identity_conv=self.add_a_random_conv_layer(filters=filters_last*width)

        before_merge=True # apply reduction before block 

        if identity_apply_dimension_reduction<self.identity_dimension_reduction_prob:
            before_merge=False
        
        identity_branch=IdentityBranch(identity_conv,before_merge)

        block=Block(depth=depth, width=width, identity_branch=identity_branch, conv_branch=_list, merge_layer=merge_layer,pooling_layer=pooling_layer)

        return block


    def get_layer_at(self,i):
        return self.indi[i]

    def get_layer_size(self):
        return len(self.indi)


    def get_activation_str(self,activation):
        #0-0.25 Relu, 0.25-0.5 Selu, 0.5-0.75 Tanh, 0.75-1 Linear
        if activation<0.25:
            return "relu"

        if activation<0.5:
            return "selu"

        if activation<0.75:
            return "tanh"

        if activation<1:
            return "Linear"


    def generate_a_new_layer(self, unit):
        current_unit_type=unit.type

        if current_unit_type == 0:
            return self.add_a_random_block(unit.width, unit.depth)

        if current_unit_type == 1:
            self.filters_weights=self.filters_weights_default

            return self.add_a_random_conv_layer()

        if current_unit_type == 2:
            return self.add_a_random_pool_layer()

        if current_unit_type == 3:
            return self.add_a_random_dense_layer()


    def mutation_a_unit(self,unit):
        current_unit_type=unit.type
        print("mutate a type ", current_unit_type)

        if current_unit_type == 0:
            print("mutate a block ")
            return self.mutate_a_block(unit)

        if current_unit_type == 1:
            print("mutate a conv ")
            return self.mutate_a_conv_layer(unit)

        if current_unit_type == 2:
            return self.mutate_a_pool_layer(unit)

        if current_unit_type == 3:
            return self.mutate_a_dense_layer(unit)

    def mutate_a_block(self,block):
        prob=np.random.rand()

        if prob<0.3:
            if block.identity_branch.before_merge == True:
                block.identity_branch.before_merge=False
            else:
                block.identity_branch.before_merge=True

        elif prob<0.6:
            #print("inserted")
            new_cnn=self.add_a_random_conv_layer()
            cnn_units=[]
            cnn_units.append(new_cnn)
            for i in range(len(block.conv_branch)):
                cnn_units.append(block.conv_branch[i])

            block.conv_branch=copy.deepcopy(cnn_units)
            block.depth=block.depth+1

        else:
            merge_function=block.merge_layer.merge_function
            #0-0.5 Concat, 0.5-0.75 Add, 0.75-1 Multiply
            if merge_function<0.5:
                if np.random.rand()<0.5:
                    block.merge_layer.merge_function=0.6
                else:
                    block.merge_layer.merge_function=0.9
            elif merge_function<0.75:
                if np.random.rand()<0.5:
                    block.merge_layer.merge_function=0.1
                else:
                    block.merge_layer.merge_function=0.9
            else:
                if np.random.rand()<0.5:
                    block.merge_layer.merge_function=0.1
                else:
                    block.merge_layer.merge_function=0.6
        return block




    def mutate_a_conv_layer(self,unit):

        #reset filters
        filters=unit.filters
        ii=np.log2(filters)

        prob=np.random.rand()
        if(prob<0.5):
            if ii>1:
                unit.filters=2**(ii-1)
        else:
            if ii<self.filters_range[-1]-1:
                unit.filters=2**(ii+1)

        #reset kerenel size
        kernel_size_id=np.random.randint(0,len(self.kernel_size_range))
        unit.kernel_size=self.kernel_size_range[kernel_size_id]
        print(str(unit))

        return unit


    def mutate_a_pool_layer(self,unit):

        #reset kerenel size
        kernel_size_id=np.random.randint(0,len(self.pool_kernel_size_range))
        unit.kernel_size=self.pool_kernel_size_range[kernel_size_id]

        #reset kernel type
        pool_type=np.random.rand()
        unit.pool_type=pool_type

        return unit


    def mutate_a_dense_layer(self,unit):
        
        #reset units
        units = np.random.randint(self.hidden_neurons_range[0], self.hidden_neurons_range[1]+1)
        unit.units=units

        return unit
        


    #0 add random, 1 modify/reset layer  2 delete layer, 3 duplicate layer
    def mutation_ope(self, r):

        if r < 0.25:
            return 0
        elif r<0.5:
            return 1
        elif r<0.75:
            return  2
        else:
            return 3


    def mutation(self,m_prob):

        if np.random.rand()>m_prob:
            return

        unit_list = []
        str_ = []
        p_m=1.0/self.get_layer_size()
        str_.append("Mutation is performing with pm {}:".format(p_m))
        

        for i in range(self.get_layer_size()):
            cur_unit = self.get_layer_at(i)


            if(np.random.rand()<p_m and cur_unit.type<6):
                p_op=self.mutation_ope(np.random.rand())
                #p_op=0
                
                current_length=(len(unit_list)+self.get_layer_size()-i-1)
                if p_op == 0: #add a new random layer
                    new_layer=self.generate_a_new_layer(cur_unit)
                    if current_length<self.max_length and cur_unit.type != 2:
                        #print("new layer type is ",new_layer.type)
                        unit_list.append(cur_unit)
                        unit_list.append(new_layer)
                        str_.append("new random layer added,")
                    else:
                        #updated_unit=self.mutation_a_unit(cur_unit)
                        unit_list.append(new_layer)
                        str_.append("layer reset,")

                if p_op == 3: # duplicate layer
                    if current_length<self.max_length and cur_unit.type != 2:
                        duplicate_layer=copy.deepcopy(cur_unit)
                        unit_list.append(cur_unit)
                        unit_list.append(duplicate_layer)
                        str_.append("new duplicate layer added,")
                    else:
                        unit_list.append(cur_unit)
                        str_.append("nothing done,")


                if p_op == 1: #reset layer
                    new_layer=self.generate_a_new_layer(cur_unit)
                    
                    #updated_unit=self.mutation_a_unit(cur_unit)
                    #print("updated unit ",updated_unit.type)
                    unit_list.append(new_layer)
                    str_.append("layer reset,")

                if p_op == 2: #delete
                    str_.append("layer deleted,")
                    lll=1

            else:
                unit_list.append(cur_unit)
                str_.append("nothing done,")

        if len(unit_list)==0:
            #don't modify if all layers are deleted
            return

        print(str_)

        self.indi=copy.deepcopy(unit_list)


    def __str__(self):

        str_ = []
        str_.append('\n\nIndividual: Length:{}, Num:{}, mutation prob {}'.format(self.get_layer_size(), self.complexity,self.m_prob))
        str_.append('accuracy:{:.2f}, learning rate: {}, crossover rate {} '.format(self.accuracy,self.learning_rate,self.c_prob))

        for i in range(self.get_layer_size()):
            unit = self.get_layer_at(i)
            if unit.type == 0:
                str_.append("\nBlock [width={},depth={}, {},".format(unit.width,unit.depth,str(unit.merge_layer)))
                id_unit=unit.identity_branch.conv_layer
                str_.append("\nid branch conv[{},{},{}, merged before {}]".format(id_unit.filters, id_unit.kernel_size, self.get_activation_str(id_unit.activation),unit.identity_branch.before_merge))
                str_.append("\nconv branch[")
                for j in range(len(unit.conv_branch)):
                    block_unit=unit.conv_branch[j]
                    str_.append("\nconv[{},{},{}]".format(block_unit.filters, block_unit.kernel_size,self.get_activation_str(block_unit.activation)))
                str_.append("]]")
            elif unit.type == 1:
                str_.append("\nconv[{},{},{}]".format(unit.filters, unit.kernel_size,self.get_activation_str(unit.activation)))
            elif unit.type ==2:
                str_.append("\npool[{},{}]".format(unit.kernel_size, "max" if unit.kernel_type<0.5 else "average"))
            elif unit.type ==3:
                str_.append("\ndense[{},{}]\n\n".format(unit.units,self.get_activation_str(unit.activation)))
            elif unit.type == 6:
                str_.append("\n flatten, \n\n")
            elif unit.type == 7:
                str_.append("\n global max pooling, \n\n")
            elif unit.type == 8:
                str_.append("\n global average poolong, \n\n")
            else:
                raise Exception("Incorrect unit flag")
        

        return ', '.join(str_)




