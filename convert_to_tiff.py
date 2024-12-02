
from fenicstools import *
from dolfin import *
import numpy as np
import matplotlib.pyplot as plt

mesh = Mesh()
with HDF5File(mesh.mpi_comm(), 'mesh_and_con_3D/mesh.h5', 'r') as h5f:
    h5f.read(mesh, "mesh",False)



S= FunctionSpace(mesh,'CG',2)
c = Function(S,name='c')



import os
path = 'output3D/'
file = sorted(os.listdir(path))[8]


with HDF5File(mesh.mpi_comm(), path+'/'+file, 'r') as h5f:
    h5f.read(c, "c")


plot(c)


xlin = np.linspace(0, 1, 10)
ylin = np.linspace(0, 1, 10)
zlin = np.linspace(0, 1, 10)



X, Y, Z = np.meshgrid(xlin, ylin, zlin)


x = np.vstack((X.flatten(), Y.flatten(), Z.flatten())).T



probes = Probes(x.flatten(), c.function_space())
probes(c) # c is the concentration field
s_intp = probes.array()
s_out = s_intp.reshape((10, 10, 10))


plt.imshow(s_out[:,:,4])



s_out[:,:,2]




