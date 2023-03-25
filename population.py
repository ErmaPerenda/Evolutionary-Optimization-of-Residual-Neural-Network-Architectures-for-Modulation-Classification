from individual import Individual
from evaluate import Evaluate
import copy


class Population:

    def __init__(self,lambdaa,max_complexity=100000,num_classes=5,N_samples=128,max_block_depth=3,max_block_width=16,max_nbr_blocks=6,max_nbr_fcl=4,prob_dense=0.2):
        self.lambdaa = lambdaa
        self.Population = []

        self.max_complexity=max_complexity
        self.num_classes=num_classes
        self.max_block_depth=max_block_depth
        self.max_block_width=max_block_width
        self.max_nbr_blocks=max_nbr_blocks
        self.prob_dense=prob_dense
        self.max_nbr_fcl=max_nbr_fcl
        self.N_samples=N_samples
        
        for i in range(lambdaa):
            print("i is ",i)
            j=True

            while j==True:
                try:
                    indi = Individual(self.max_block_depth,self.max_block_width,self.max_nbr_blocks,self.max_nbr_fcl,self.prob_dense)
                    indi.initialize()
                    evaluate=Evaluate(num_classes=self.num_classes,N_samples=self.N_samples)
                    complexity,accuracy=evaluate.evaluate_individual(indi,0,True)
                    if complexity<self.max_complexity:
                        print("success")
                        self.Population.append(indi)
                        j=False
                    else:
                        print("bad try")
                except:
                    print("raised exception")

    def get_individual_at(self, i):
        return self.Population[i]

    def remove_individual_at(self,i):
        #print("len population is ", self.get_pop_size())
        del self.Population[i]
        #print("len population is ", self.get_pop_size())

    def set_individual_at(self,i,ind):
        self.Population[i]=copy.deepcopy(ind)

    def get_individual_subset(self,i):
        X=[]
        for j in i:
            #print("j is ",j)
            X.append(self.Population[j])
        return X

    def get_pop_size(self):
        return len(self.Population)

    def set_populations(self, new_generation):
        self.Population = new_generation

    def append_individual(self, individal):
        self.Population.append(individal)

    def extend_population(self,offspring):
        #print("len pop size is ", self.get_pop_size())
        #print("len off ", offspring.get_pop_size())
       
        for i in range(offspring.get_pop_size()):
            self.Population.append(offspring.get_individual_at(i))
        #print(" after len pop size is ", self.get_pop_size())


    def __str__(self):
        _str = []
        for i in range(self.get_pop_size()):
            _str.append(str(self.get_individual_at(i)))
        return '\n'.join(_str)