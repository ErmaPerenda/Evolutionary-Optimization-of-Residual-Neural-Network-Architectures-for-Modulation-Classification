'''
All four kinds of layers can be initialized with their default parameters
'''

class ConvLayer:
    def __init__(self, kernel_size=3, filters=32, activation=0.1):
        self.kernel_size = kernel_size
        self.filters = filters
        self.activation=activation # 0-0.25 Relu, 0.25-0.5 Selu, 0.5-0.75 Tanh, 0.75-1 Linear
        self.type = 1
    def __str__(self):
        return "Conv Layer: filters:{0}, kernel size:{1}".format(self.filters, self.kernel_size)

class PoolLayer:
    def __init__(self, kernel_size=2, pool_type=0.1):
        self.kernel_size= kernel_size
        self.kernel_type = pool_type # values below 0.5 means Max other wise Average
        self.type = 2

    def __str__(self):
        return "Pool Layer: kernel size:{0}, type:{1}".format(self.kernel_size, "max" if self.kernel_type<0.5 else "average")

class DenseLayer:
    def __init__(self, units=128,activation=0.1):
        self.units = units
        self.activation=activation # 0-0.25 Relu, 0.25-0.5 Selu, 0.5-0.75 Tanh, 0.75-1 Linear
        self.type = 3

    def __str__(self):
        return "Dense Layer: hidden neurons:{0}, type {1}".format(self.units,self.type)

class FlattenLayer:
    def __init__(self):
        self.type=6
    def __str__(self):
        return "Flatten Layer."

class GlobalMaxPoolingLayer:
    def __init__(self):
        self.type=7
    def __str__(self):
        return "Global Max Pooling Layer"

class GlobalAveragePoolingLayer:
    def __init__(self):
        self.type=8
    def __str__(self):
        return "Global Average Pooling Layer"


class MergeLayer:
    def __init__(self,merge_function=0):
        self.merge_function=merge_function
        self.type=5

    def __str__(self):
        ##0-0.33 Concat, 0.3-0.66 Multiply, 0.66-1 Add
        if self.merge_function<0.33:
            merge_function="Concat"
        elif self.merge_function<0.66:
            merge_function="Multiply"
        else:
            merge_function="Add"

        return "Merge Layer with {}".format(merge_function)


class IdentityBranch:
    def __init__(self, conv_layer=None, before_merge=True):
        self.type=4
        self.conv_layer=conv_layer
        self.before_merge=before_merge

    def __str__(self):
        
        return "Identity Layer: conv applied before merge = {0} ".format(self.before_merge)

class Block:
    def __init__(self,depth, width,identity_branch, conv_branch, merge_layer,pooling_layer):
        self.type=0
        self.width=width
        self.depth=depth
        self.identity_branch=identity_branch
        self.conv_branch=conv_branch
        self.merge_layer=merge_layer
        self.pooling_layer=pooling_layer


    def __str__(self):
        return "Block of CNN layers with depth {} and width {}".format(self.depth, self.width)

