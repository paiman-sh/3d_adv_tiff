import time
import os
import math
from dolfin import *

# Test for PETSc or Tpetra
if not has_linear_algebra_backend("PETSc") and not has_linear_algebra_backend("Tpetra"):
    info("DOLFIN has not been configured with Trilinos or PETSc. Exiting.")
    exit()

if not has_krylov_solver_preconditioner("amg"):
    info("Sorry, this demo is only available when DOLFIN is compiled with AMG "
	 "preconditioner, Hypre or ML.")
    exit()

if has_krylov_solver_method("minres"):
    krylov_method = "minres"
elif has_krylov_solver_method("tfqmr"):
    krylov_method = "tfqmr"
else:
    info("Default linear algebra backend was not compiled with MINRES or TFQMR "
         "Krylov subspace method. Terminating.")
    exit()


Pe = Constant(50)
t_end = 1.4
dt = 0.001


mesh = UnitCubeMesh(10, 10, 10)
plot(mesh)


# Define function spaces
V = FunctionSpace(mesh, "Lagrange", 2)


# Create boundary markers
boundary_parts = MeshFunction("size_t", mesh, mesh.topology().dim()-2, 0)

top  = AutoSubDomain(lambda x: near(x[2], 1.0))
bottom = AutoSubDomain(lambda x: near(x[2], 0.0))

top   .mark(boundary_parts, 2)
bottom.mark(boundary_parts, 3)


u = Expression(("0","0","-1"), domain=mesh, degree=0)

bc = DirichletBC(V, Constant(1), top)


# Define unknown and test function(s)
b = TestFunction(V)
c = TrialFunction(V)

c0 = Function(V)

n = FacetNormal(mesh)
theta = Constant(1.0)


# Define variational forms
a0=(1.0/Pe)*inner(grad(c0), grad(b))*dx + inner(u,grad(c0))* b *dx
a1=(1.0/Pe)*inner(grad(c), grad(b))*dx + inner(u,grad(c))* b *dx

A = (1/dt)*inner(c, b)*dx - (1/dt)*inner(c0,b)*dx + theta*a1 + (1-theta)*a0

F = A

# Create file for storing results
#vtkfile = File('output/mass_transfer.pvd')
xdmffile = XDMFFile(mesh.mpi_comm(), "output3D/c.xdmf")
xdmffile.parameters["flush_output"] = True
xdmffile.parameters["rewrite_function_mesh"] = False
xdmffile.parameters["functions_share_mesh"] = True

c = Function(V)
ffc_options = {"optimize": True, "quadrature_degree": 8}
problem = LinearVariationalProblem(lhs(F),rhs(F), c, [bc], form_compiler_parameters=ffc_options)
solver  = LinearVariationalSolver(problem)
c.assign(c0)
c.rename("c", "concentration")

# Time-steppinghttp://localhost:8888/notebooks/psi2/velocity_verification/Untitled.ipynb?kernel_name=python3#
t = 0.0
it = 0

#vtkfile << c
xdmffile.write(c, t)

with HDF5File(mesh.mpi_comm(), 'output3D/mesh.h5', 'w') as h5f:
    h5f.write(mesh, "mesh")
    

while t < t_end:
    t += dt
    it += 1
    print("t =", t, "end t=", t_end)

    # Compute
    solver.solve()
    #plot(c)
    # Save to file
    #vtkfile << c
    if it % 20 == 0:
        xdmffile.write(c, t)
        with HDF5File(mesh.mpi_comm(), "output3D/fields_tstep{:06d}.h5".format(it), "w") as h5f:
            h5f.write(c, "c")
    
    # Move to next time step
    c0.assign(c)





