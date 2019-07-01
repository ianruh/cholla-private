import h5py
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from IPython import embed
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
import sys
import os, os.path

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

for i in range(minFileNum, maxFileNum, 1):
	f = h5py.File(dataDir + str(i) + '.h5.0', 'r')
	head = f.attrs
	nx = head['dims'][0]
	ny = head['dims'][1]
	density = np.array(f['density'])
	#embed()

	X = np.linspace(0, 1, nx)
	Y = np.linspace(0, 1, ny)

	levels = MaxNLocator(nbins=15).tick_values(density.min(), density.max())
	cmap = plt.get_cmap('PiYG')
	norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)

	fig = plt.figure(figsize=(10,10))
	ax1 = plt.axes([0.1,0.1,0.8,0.8])
	im = ax1.pcolormesh(X, Y, density, cmap=cmap, norm=norm)
	fig.colorbar(im, ax=ax1)
	ax1.set_title('2D Cold/Hot/Cold Density - Index: ' + str(i))
	plt.savefig(outDir + str(i) + '.png', dpi=200)
	plt.close(fig)
	print("Wrote: " + str(i) + ".png")
