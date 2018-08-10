import pandas as pd
import numpy as np
import random
import bigfloat


class snip():

    index_used = []
    test_index = []
    snip_table = 0
    ROWS = 0

    def __init__(self, ROWS):
        self.index_used = random.sample(range(1062), ROWS)
        self.test_index = random.sample(range(1062), ROWS)
        for i,t in enumerate(self.index_used):
            self.test_index[i] = t

        self.test_index.sort()
        self.index_used.sort()

        snips_df = pd.read_csv('AIRWAVE_BP_SNPData.csv', header=0, sep=',')
        snips_df = snips_df.round(decimals=0)
        #print(snips_df.shape)
        self.snip_table = snips_df.values
        self.snip_table = self.snip_table.astype(np.int64)
        #print(self.snip_table)
        self.ROWS = ROWS

    def make_current(self):
        self.test_index.sort()
        self.index_used.sort()
        for i,t in enumerate(self.test_index):
            self.index_used[i] = t

    def make_step(self):

        col_change = random.randint(0,self.ROWS-1)

        while True:
            new_col = random.randint(0,1062-1)
            self.test_index[col_change] = new_col
            if len(set(self.test_index)) == self.ROWS:
                break
        self.test_index.sort()
        self.index_used.sort()

    def calculate_optimality(self, type = 'max_one'):
        if type == 'max_one':
            bins = np.zeros((1989), dtype= np.int64)

            for exp, j in enumerate(self.test_index):
                bins += self.snip_table[:,j] * (3 ** exp)

            bin_count = np.bincount(bins)
            return bin_count.max()
        else:

            bins = np.zeros((1989), dtype=np.int64)

            for exp, j in enumerate(self.test_index):
                bins += self.snip_table[:, j] * (3 ** exp)

            bin_count = np.bincount(bins)
            non = np.count_nonzero(bin_count)

            bins = np.zeros((1989), dtype=np.int64)

            for exp, j in enumerate(self.test_index):
                bins += self.snip_table[:, j] * (3 ** exp)

            bin_count = np.bincount(bins)
            max_b = bin_count.max()

            return max_b / non

    def validate(self, index):

        bins = np.zeros((1989), dtype=np.int64)
        for exp, j in enumerate(index):
            bins += self.snip_table[:, j] * (3 ** exp)

        bin_count = np.bincount(bins)
        print(bin_count.max())

    def validate_clusters(self, index):

        bins = np.zeros((1989), dtype=np.int64)

        for exp, j in enumerate(index):
            bins += self.snip_table[:, j] * (3 ** exp)

        bin_count = np.bincount(bins)
        print( np.count_nonzero(bin_count))


def acceptance_probability(old_cost, new_cost,T ,type= 'max_one'):
    if type == 'max_one' or 'else':
        return np.exp((np.log10(new_cost) - np.log10(old_cost))/T)
    else:
        return np.exp((np.log10(old_cost)-np.log10(new_cost)) / T)

def anneal(snip, type = 'max_one'):
    old_cost = snip.calculate_optimality(type)
    max_cost = 0
    T = 1.0
    T_min = 0.0001
    alpha = 0.99
    first = True
    while T > T_min:
        i = 1

        while i <=  8*1200:
            snip.make_step()
            new_sol = snip.test_index
            new_cost = snip.calculate_optimality(type)
            ap = acceptance_probability(old_cost, new_cost, T)

            if ap > random.random():
                snip.make_current()
                sol = new_sol
                old_cost = new_cost
            i += 1

            if  old_cost   > max_cost:
                cost = old_cost
                best_sol = snip.index_used
                max_cost = cost

        first = False
        T = T*alpha
        #print("T: %0.3f best sol: %s best group: %d"%(T,best_sol,cost,))

    return best_sol, cost


def anneal_clusters(snip, type = 'else'):
    old_cost = snip.calculate_optimality(type)
    max_cost = 10000000
    T = 1.0
    T_min = 0.001
    alpha = 0.95
    first = True
    while T > T_min:
        i = 1

        while i <= 100:
            snip.make_step()
            new_sol = snip.test_index
            new_cost = snip.calculate_optimality(type)
            ap = acceptance_probability(old_cost, new_cost, T , type= type)

            if ap > random.random():
                snip.make_current()
                sol = new_sol
                old_cost = new_cost
            i += 1

            if  old_cost  < max_cost:
                cost = old_cost
                best_sol = snip.index_used
                max_cost = cost

        first = False
        T = T*alpha
        #print("T: %0.3f best sol: %s best group: %d"%(T,best_sol,cost,))

    return best_sol, cost


if __name__ == '__main__':

    #print("Starting sim_anneal # %d " % i)
    snip_obj = snip(8)



    snip_obj.validate_clusters([133, 221, 224, 398, 1012, 1051, 575, 726])
    #print(anneal_clusters(snip_obj, type='else'))

    print(anneal_clusters(snip_obj))
    for i in range(66):
        print(anneal(snip_obj))
