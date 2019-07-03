import h5py
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from IPython import embed
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
import sys
import os, os.path
import multiprocessing

directory = "./"
if len(sys.argv) != 2:
	print("Usage: " + str(sys.argv[0]) + "[directory]")
	sys.exit()
else:
	directory = sys.argv[1]

dataDir = directory + '/raw/'
outDir = directory + '/png/'

minFileNum = 0
maxFileNum = len(os.listdir(dataDir))

f1 = h5py.File(dataDir + '0.h5.0', 'r')
density1 = np.array(f1['density'])
levels = MaxNLocator(nbins=15).tick_values(density1.min(), density1.max())
cmap = plt.get_cmap('PiYG')
norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)

print("Writing figures...")

def plot(number):
	f = h5py.File(dataDir + str(number) + '.h5.0', 'r')
	head = f.attrs
	nx = head['dims'][0]
	ny = head['dims'][1]
	density = np.array(f['density'])
	#embed()

	X = np.linspace(0, 1, nx)
	Y = np.linspace(0, 1, ny)

	# levels = MaxNLocator(nbins=15).tick_values(density.min(), density.max())
	# cmap = plt.get_cmap('PiYG')
	# norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)

	fig = plt.figure(figsize=(10,10))
	ax1 = plt.axes([0.1,0.1,0.8,0.8])
	im = ax1.pcolormesh(X, Y, density, cmap=cmap, norm=norm)
	fig.colorbar(im, ax=ax1)
	ax1.set_title('2D Diffusion (Density)- Index: ' + str(number))
	plt.savefig(outDir + str(number) + '.png', dpi=200)
	plt.close(fig)

pool = multiprocessing.Pool(20)
pool.map(plot, range(minFileNum, maxFileNum, 1))
	
