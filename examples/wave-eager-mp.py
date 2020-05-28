from __future__ import division, absolute_import, print_function

__copyright__ = "Copyright (C) 2020 CEESD"

__license__ = """
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""


import numpy as np
import numpy.linalg as la  # noqa
import pyopencl as cl
import pyopencl.array as cla  # noqa
import pyopencl.clmath as clmath
from pytools.obj_array import (
        join_fields, make_obj_array,
        with_object_array_or_scalar)
from meshmode.mesh import BTAG_ALL, BTAG_NONE  # noqa
from grudge.eager import EagerDGDiscretization, with_queue
from grudge.symbolic.primitives import TracePair
from grudge.shortcuts import make_visualizer
from mirgecom.wave import wave_operator
from mirgecom.integrators import rk4_step


def bump(discr, queue, t=0):
    source_center = np.array([0.2, 0.35, 0.1])[:discr.dim]
    source_width = 0.05
    source_omega = 3

    nodes = discr.nodes().with_queue(queue)
    center_dist = join_fields([
        nodes[i] - source_center[i]
        for i in range(discr.dim)
        ])

    return (
        np.cos(source_omega*t)
        * clmath.exp(
            -np.dot(center_dist, center_dist)
            / source_width**2))


def main():
    cl_ctx = cl.create_some_context()
    queue = cl.CommandQueue(cl_ctx)

    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    i_local_rank = comm.Get_rank()
    num_parts = comm.Get_size()

    from meshmode.distributed import MPIMeshDistributor, get_partition_by_pymetis
    mesh_dist = MPIMeshDistributor(comm)

    dim = 2
    nel_1d = 16    
    order = 3
    
    if dim == 2:
        # no deep meaning here, just a fudge factor
        dt = 0.75/(nel_1d*order**2)
    elif dim == 3:
        # no deep meaning here, just a fudge factor
        dt = 0.45/(nel_1d*order**2)
    else:
        raise ValueError("don't have a stable time step guesstimate")

    if mesh_dist.is_mananger_rank():
        from meshmode.mesh.generation import generate_regular_rect_mesh
        mesh = generate_regular_rect_mesh(a=(-0.5,)*dim,
                                          b=(0.5,)*dim,
                                          n=(nel_1d,)*dim)
        print("Mesh has %d elements" % mesh.nelements)
        part_per_element = get_partition_by_pymetis(mesh, num_parts)

        local_mesh = mesh_dist.send_mesh_parts(mesh, part_per_element, num_parts)
    else:
        local_mesh = mesh_dist.receive_mesh_part()

    print("Local mesh has %d elements" % local_mesh.nelements)
    
    #    vol_discr = DGDiscretizationWithBoundaries(cl_ctx, local_mesh, order=order,
    #                                               mpi_communicator=comm)
    discr = EagerDGDiscretization(cl_ctx, local_mesh, order=order,
                                  mpi_communicator=comm)

    # ______ OLD _______

    #    discr = EagerDGDiscretization(cl_ctx, mesh, order=order)

    fields = join_fields(
            bump(discr, queue),
            [discr.zeros(queue) for i in range(discr.dim)]
            )

    vis = make_visualizer(discr, discr.order+3 if dim == 2 else discr.order)

    def rhs(t, w):
        return wave_operator(discr, c=1, w=w)

    t = 0
    t_final = 3
    istep = 0
    while t < t_final:
        fields = rk4_step(fields, t, dt, rhs)

        if istep % 10 == 0:
            if i_local_rank == 0:
                print(istep, t, la.norm(fields[0].get()))
            vis.write_vtk_file("fld-wave-eager-%(stepno)04d-%(rankno)04d.vtu"
                               % {'stepno': istep,'rankno': i_local_rank},
                               [
                                   ("u", fields[0]),
                                   ("v", fields[1:]),
                               ])
            
        t += dt
        istep += 1


if __name__ == "__main__":
    main()

# vim: foldmethod=marker
