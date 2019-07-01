import h5py
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

dataDir = './1d/h5/'
outDir = './1d/png/'

minFileNum = 0
maxFileNum = 25

for i in range(minFileNum, maxFileNum, 1):
	f = h5py.File(dataDir + str(i) + '.h5.0', 'r')
	head = f.attrs
	nx = head['dims'][0]
	density = np.array(f['density'])
	
	fig = plt.figure(figsize=(10,10))
	ax1 = plt.axes([0.1,0.1,0.8,0.8])
	plt.axis([0, nx, 0, 1.1])
	ax1.plot(density, 'o', markersize=2, color="blue")
	plt.savefig(outDir + str(i) + '.png', dpi=200)
	plt.close(fig)
