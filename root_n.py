import pandas as pd
import numpy as np
import random
import bigfloat
import matplotlib.pyplot as plt

class snip():

    index_used = []
    test_index = []
    snip_table = 0
    ROWS = 0

    def __init__(self, ROWS):

        self.test_index = random.sample(range(1062), ROWS)
        self.test_index.sort()


        snips_df = pd.read_csv('AIRWAVE_BP_SNPData.csv', header=0, sep=',')
        snips_df = snips_df.round(decimals=0)
        #print(snips_df.shape)
        self.snip_table = snips_df.values
        self.snip_table = self.snip_table.astype(np.int64)
        #print(self.snip_table)
        self.ROWS = ROWS


    def make_step(self):

        col_change = random.randint(0,self.ROWS-1)

        while True:
            new_col = random.randint(0,1062-1)
            self.test_index[col_change] = new_col
            if len(set(self.test_index)) == self.ROWS:
                break
        self.test_index.sort()


    def calculate_mean(self, type = 'max_one'):
        if type == 'max_one':
            bins = np.zeros((1989), dtype= np.int64)

            for exp, j in enumerate(self.test_index):
                bins += self.snip_table[:,j] * (3 ** exp)

            bin_count = np.bincount(bins)
            #return bin_count.mean(axis=0),bin_count.std(axis=0),bin_count.max()

            n = 5
            return np.sum(bin_count[np.argsort(bin_count)[-n:]])
            #return bin_count.max()


def get_pdf():

    pass

if __name__ == '__main__':
    alpha_size = 8
    snip_obj = snip(alpha_size)

    iters1 = 100000
    iters2 = 100
    bins = np.zeros(1989)
    for i in range(iters2):
        print("Completed: ", i , " out of: ", iters2)
        for j in range(iters1):
            snip_obj.make_step()
            #mean, std , max =snip_obj.calculate_mean()
            bins[int(snip_obj.calculate_mean()//1)] += 1



    np.save('sbins_n_%d'%alpha_size, bins)
    #bins = np.load('bins.npy')
    df  = pd.DataFrame(bins)
    df.to_excel('pdf_sbins_n_%d.xlsx'%alpha_size,index= False)
    print(bins)
    plt.plot(bins/(iters1*iters2))
    plt.show()