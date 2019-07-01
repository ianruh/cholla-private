import h5py
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from IPython import embed
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
import sys

directory = "./"
if len(sys.argv) != 2:
	print("Usage: " + str(sys.argv[0]) + " [file]")
	sys.exit()
else:
	directory = sys.argv[1]


dataDir = directory + '/h5/'
outDir = directory + '/png/'

minFileNum = 0
maxFileNum = 1

for i in range(minFileNum, maxFileNum+1, 1):
	f = h5py.File(dataDir + '0.h5.' + str(i), 'r')
	head = f.attrs
	nx = head['dims'][0]
	ny = head['dims'][1]
	nz = head['dims'][2]
	density = np.array(f['density'])
	#embed()

	X = np.linspace(0, 1, nx)
	Y = np.linspace(0, 1, ny)
	Z = np.linspace(0, 1, nz)

	levels = MaxNLocator(nbins=15).tick_values(density.min(), density.max())
	cmap = plt.get_cmap('PiYG')
	norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)

	fig = plt.figure(figsize=(10,10))
	ax1 = plt.axes([0.1,0.1,0.8,0.8])
	im = ax1.pcolormesh(X, Y, density, cmap=cmap, norm=norm)
	fig.colorbar(im, ax=ax1)
	ax1.set_title('2D Cold/Hot/Cold Density')
	plt.savefig(outDir + str(i) + '.png', dpi=200)
	plt.close(fig)
