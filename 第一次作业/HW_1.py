from scipy.io import loadmat
import matplotlib.pyplot as plt
from pylab import *

data = loadmat('data.mat')
x = range(len(data))
plt.plot(x,data,'ro-')