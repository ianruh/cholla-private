import h5py
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
import sys
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

def plot(i):
	f = h5py.File(dataDir + str(i) + '.h5.0', 'r')
	head = f.attrs
	nx = head['dims'][0]
	density = np.array(f['Energy'])
	
	fig = plt.figure(figsize=(10,10))
	ax1 = plt.axes([0.1,0.1,0.8,0.8])
	plt.axis([0, nx, 0, 1.1])
	ax1.plot(density, 'o', markersize=2, color="blue")
	plt.savefig(outDir + str(i) + '.png', dpi=200)
	plt.close(fig)

pool = multiprocessing.Pool(20)
pool.map(plot, range(minFileNum, maxFileNum, 1))