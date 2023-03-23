import numpy as np
import matplotlib.pyplot as plt
from leastSquares import LeastSquares

fig = plt.figure(figsize=(10, 4))
plt.rcParams["figure.autolayout"] = True
ax = fig.add_subplot(111)

ls=LeastSquares()

H_all=np.random.random((ls.max_samples+5,1))

# cost_av=ls.ComputeExampleLS(H_all)
ls.ComputeExampleCovRLS(H_all)

ls.ComputeExpectedCostValues(H_all)

# plt.plot(cost_av,linewidth=2,color="orange",label="cost av")
ax.legend()
plt.show()


