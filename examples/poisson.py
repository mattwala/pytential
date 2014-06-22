import pyopencl as cl
import pyopencl.array  # noqa
import pyopencl.clmath  # noqa

from meshmode.mesh.io import generate_gmsh, FileSource
from meshmode.discretization import Discretization
from meshmode.discretization.poly_element import \
        QuadratureSimplexGroupFactory

from meshmode.discretization.visualization import make_visualizer


def soln_func(x, y):
    return 0.1*cl.clmath.sin(30*x)*cl.clmath.sin(20*y)


bc_func = soln_func


def rhs_func(x, y):
    return 0.1*-30*-20*cl.clmath.sin(30*x)*cl.clmath.sin(20*y)

mesh_order = 4
qbx_order = 3


def main():
    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)

    mesh = generate_gmsh(
            FileSource("blob-2d.step"), 2, order=mesh_order,
            force_ambient_dimension=2,
            other_options=["-string", "Mesh.CharacteristicLengthMax = 0.008;"]
            )

    vol_discr = Discretization(ctx, mesh, QuadratureSimplexGroupFactory(mesh_order))

    vol_vis = make_visualizer(queue, vol_discr, 20)

    x = vol_discr.nodes().with_queue(queue)
    f = rhs_func(x[0], x[1])
    soln = soln_func(x[0], x[1])

    #vol_vis.write_vtk_file("x.vtu", [("f", f)])

    from meshmode.discretization.connection import make_boundary_restriction
    bdry_mesh, bdry_discr, bdry_connection = make_boundary_restriction(
            queue, vol_discr, QuadratureSimplexGroupFactory(mesh_order))

    bdry_nodes = bdry_discr.nodes().with_queue(queue)
    bdry_f = rhs_func(bdry_nodes[0], bdry_nodes[1])
    bdry_f_2 = bdry_connection(queue, f)

    bdry_vis = make_visualizer(queue, bdry_discr, 20)
    bdry_vis.write_vtk_file("y.vtu", [("f", bdry_f_2)])

    if 0:
        vis.show_scalar_in_mayavi(f, do_show=False)
        bdry_vis.show_scalar_in_mayavi(bdry_f - bdry_f_2, line_width=10,
                do_show=False)

        import mayavi.mlab as mlab
        mlab.colorbar()
        mlab.show()

    # {{{ compute volume potential

    from sumpy.qbx import LayerPotential
    from sumpy.expansion.local import LineTaylorLocalExpansion

    def get_kernel():
        from sumpy.symbolic import pymbolic_real_norm_2
        from pymbolic.primitives import (make_sym_vector, Variable as var)

        r = pymbolic_real_norm_2(make_sym_vector("d", 3))
        expr = var("log")(r)
        scaling = 1/(-2*var("pi"))

        from sumpy.kernel import ExpressionKernel
        return ExpressionKernel(
                dim=3,
                expression=expr,
                scaling=scaling,
                is_complex_valued=False)

    laplace_2d_in_3d_kernel = get_kernel()

    # layer_pot = LayerPotential(ctx, [
    #     LineTaylorLocalExpansion(laplace_2d_in_3d_kernel,
    #         qbx_order=5)])

    # }}}

    # {{{ solve bvp

    from sumpy.kernel import LaplaceKernel
    from pytential.symbolic.pde.scalar import DirichletOperator
    op = DirichletOperator(LaplaceKernel(2), -1, use_l2_weighting=True)

    from pytential import bind, sym, norm

    sym_sigma = sym.var("sigma")
    op_sigma = op.operator(sym_sigma)

    from pytential.qbx import QBXLayerPotentialSource
    qbx = QBXLayerPotentialSource(
            bdry_discr, fine_order=4*mesh_order, qbx_order=qbx_order,
            # Don't use FMM for now
            fmm_order=3)

    bound_op = bind(qbx, op_sigma)

    bc = bc_func(bdry_nodes[0], bdry_nodes[1])
    bdry_f = rhs_func(bdry_nodes[0], bdry_nodes[1])

    rhs = bind(bdry_discr, op.prepare_rhs(sym.var("bc")))(queue, bc=bc)

    from pytential.gmres import gmres
    gmres_result = gmres(
            bound_op.scipy_op(queue, "sigma"),
            rhs, tol=1e-14, progress=True,
            hard_failure=False)

    sigma = gmres_result.solution
    print "gmres state:", gmres_result.state

    # }}}

    bvp_sol = bind(
            (qbx, vol_discr),
            op.representation(sym_sigma))(queue, sigma=sigma)

    #vol_vis.show_scalar_in_mayavi(bvp_sol)
    vol_vis.write_vtk_file("poisson.vtu", [
        ("bvp_sol", bvp_sol),
        ("soln", soln),
        ])


if __name__ == "__main__":
    main()

# vim: foldmethod=marker
