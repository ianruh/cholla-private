import h5py
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
import sys
import multiprocessing

dir1D = './Cond1D/'
dir2D = './Cond2D/'
dir3D = './Cond3D/'

dataDir1D = dir1D + '/raw/'
dataDir2D = dir2D + '/raw/'
dataDir3D = dir3D + '/raw/'

outDir = './ConductionTest/'

minFileNum = 0
maxFileNum = len(os.listdir(dataDir1D))

def plot(i):
	f_1d = h5py.File(dataDir1D + str(i) + '.h5.0', 'r')
	f_2d = h5py.File(dataDir2D + str(i) + '.h5.0', 'r')
	f_3d = h5py.File(dataDir3D + str(i) + '.h5.0', 'r')

	head_1d = f_1d.attrs
	head_2d = f_2d.attrs
	head_3d = f_3d.attrs

	# 1 D
	nx_1d = np.linspace(0, 1, head_1d['dims'][0])
	energy_1d = np.array(f_1d['Energy'])
	density_1d = np.array(f_1d['density'])
	temp_1d = energy_1d * (1.6666667 - 1) / density_1d
	plt.plot(nx_1d, temp_1d)

	# 2 D
	nx_2d = np.linspace(0, 1, head_2d['dims'][0])
	ny_2d = np.linspace(0, 1, head_2d['dims'][1])
	energy_2d = np.swapaxes(np.array(f_2d['Energy']),0,1)
	density_2d = np.swapaxes(np.array(f_2d['density']),0,1)
	energy_2d_slice = energy_2d[50]
	density_2d_slice = density_2d[50]
	temp_2d = energy_2d_slice * (1.6666667 - 1) / density_2d_slice
	plt.plot(nx_2d, temp_2d)

	# 3 D
	nx_3d = np.linspace(0, 1, head_3d['dims'][0])
	ny_3d = np.linspace(0, 1, head_3d['dims'][1])
	nz_3d = np.linspace(0, 1, head_3d['dims'][2])
	energy_3d = np.swapaxes(np.array(f_3d['Energy']), 0, 2)
	density_3d = np.swapaxes(np.array(f_3d['density']),0,2)

	#import code; code.interact(local=dict(globals(), **locals()))

	energy_3d_line = energy_3d[int(len(ny_3d) / 2)][int(len(nz_3d) / 2)]
	density_3d_line = density_3d[int(len(ny_3d) / 2)][int(len(nz_3d) / 2)]
	temp_3d = energy_3d_line * (1.6666667 - 1) / density_3d_line
	plt.plot(nx_3d, temp_3d)
	

	plt.legend(['1D', '2D', '3D'])
	plt.ylim([0.87,1.1])
	plt.title('Temperature in 1D, 2D, and 3D Simulations')
	plt.savefig(outDir + str(i) + '.png', dpi=200)
	plt.close()

pool = multiprocessing.Pool(20)
pool.map(plot, range(minFileNum, maxFileNum, 1))
# for num in range(minFileNum, maxFileNum, 1):
# 	plot(num)