import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson
import seaborn as sb

s = np.random.poisson(0.3, 1989)

#count, bins, ignored = plt.hist(s, 100, normed=True)



s = np.random.poisson(lam=(100., 500.), size=(100, 2))
s = np.random.poisson(lam=(100., 500.), size=(100, 2))

data_binom = poisson.rvs(mu=0.3, size=1989)



prob = poisson.cdf(data_binom, 0.3)

prob = poisson.sf(1000, 0.3, loc=0)

print(prob)
ax = sb.distplot(prob,
                  kde=True,
                  color='green',
                  hist_kws={"linewidth": 25,'alpha':1})
ax.set(xlabel='Poisson', ylabel='Frequency')

plt.show()