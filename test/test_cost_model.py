from __future__ import division, print_function

__copyright__ = "Copyright (C) 2018 Matt Wala"

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
import pyopencl.clmath  # noqa
import pytest
from pyopencl.tools import (  # noqa
        pytest_generate_tests_for_pyopencl as pytest_generate_tests)

from sumpy.kernel import LaplaceKernel, HelmholtzKernel

from pytential import bind, sym, norm  # noqa
from pytential.qbx.cost import CostModel


# {{{ global params

TARGET_ORDER = 8
OVSMP_FACTOR = 5
TCF = 0.9
QBX_ORDER = 5
FMM_ORDER = 10

DEFAULT_LPOT_KWARGS = {
        "_box_extent_norm": "l2",
        "_from_sep_smaller_crit": "static_l2",
        }


def get_lpot_source(queue, dim):
    from meshmode.discretization import Discretization
    from meshmode.discretization.poly_element import (
            InterpolatoryQuadratureSimplexGroupFactory)

    target_order = TARGET_ORDER

    if dim == 2:
        from meshmode.mesh.generation import starfish, make_curve_mesh
        mesh = make_curve_mesh(starfish, np.linspace(0, 1, 50), order=target_order)
    elif dim == 3:
        from meshmode.mesh.generation import generate_torus
        mesh = generate_torus(2, 1, order=target_order)
    else:
        raise ValueError("unsupported dimension: %d" % dim)

    pre_density_discr = Discretization(
            queue.context, mesh,
            InterpolatoryQuadratureSimplexGroupFactory(target_order))

    lpot_kwargs = DEFAULT_LPOT_KWARGS.copy()
    lpot_kwargs.update(
            _expansion_stick_out_factor=TCF,
            fmm_order=FMM_ORDER,
            qbx_order=QBX_ORDER,
            fmm_backend="fmmlib",
            )

    from pytential.qbx import QBXLayerPotentialSource
    lpot_source = QBXLayerPotentialSource(
            pre_density_discr, OVSMP_FACTOR*target_order,
            **lpot_kwargs)

    lpot_source, _ = lpot_source.with_refinement()

    return lpot_source


def get_density(queue, lpot_source):
    density_discr = lpot_source.density_discr
    nodes = density_discr.nodes().with_queue(queue)
    return cl.clmath.sin(10 * nodes[0])


def get_slp_cost(queue, lpot_source, k=0):
    """Return the modeled cost of on-surface evaluation for the single-layer
    potential."""

    sym_op_S_extra_kwargs = {}

    if k == 0:
        k_sym = LaplaceKernel(lpot_source.ambient_dim)
    else:
        k_sym = HelmholtzKernel(lpot_source.ambient_dim, "k")
        sym_op_S_extra_kwargs["k"] = k

    sym_op_S = sym.S(
            k_sym, sym.var("sigma"), qbx_forced_limit=+1, **sym_op_S_extra_kwargs)

    inspect_geo_data_result = []

    def inspect_geo_data(insn, bound_expr, geo_data):
        from pytential.qbx.cost import CostModel
        cost_model = CostModel()

        kernel = lpot_source.get_fmm_kernel(insn.kernels)
        kernel_arguments = insn.kernel_arguments

        result = cost_model(geo_data, kernel, kernel_arguments)
        inspect_geo_data_result.append(result)

        return False

    lpot_source = lpot_source.copy(geometry_data_inspector=inspect_geo_data)
    op_S = bind(lpot_source, sym_op_S)
    sigma = get_density(queue, lpot_source)
    op_S(queue, sigma=sigma)

    return inspect_geo_data_result[0]

# }}}


# {{{ test cost model

@pytest.mark.parametrize("dim", (2, 3))
def test_cost_model(ctx_getter, dim):
    """Test that cost model gathering can execute successfully."""
    cl_ctx = ctx_getter()
    queue = cl.CommandQueue(cl_ctx)
    lpot_source = get_lpot_source(queue, dim)
    cost_S = get_slp_cost(queue, lpot_source)
    from pytential.qbx.cost import ParametrizedCosts
    assert isinstance(cost_S, ParametrizedCosts)

# }}}


# {{{ test cost model metadata gathering

def test_cost_model_metadata_gathering(ctx_getter):
    """Test that the cost model correctly gathers metadata."""
    cl_ctx = ctx_getter()
    queue = cl.CommandQueue(cl_ctx)

    from sumpy.expansion.level_to_order import SimpleExpansionOrderFinder

    fmm_level_to_order = SimpleExpansionOrderFinder(tol=1e-5)

    dim = 2
    k = 3

    lpot_source = get_lpot_source(queue, dim).copy(
            fmm_level_to_order=fmm_level_to_order)

    cost_S = get_slp_cost(queue, lpot_source, k)
    kernel = HelmholtzKernel(dim, "k")

    geo_data = lpot_source.qbx_fmm_geometry_data(
            target_discrs_and_qbx_sides=((lpot_source.density_discr, 1),))

    tree = geo_data.tree()

    assert cost_S.params["p_qbx"] == QBX_ORDER
    assert cost_S.params["nlevels"] == tree.nlevels
    assert cost_S.params["nsources"] == tree.nsources
    assert cost_S.params["ntargets"] == tree.ntargets
    assert cost_S.params["ncenters"] == geo_data.ncenters

    for level in range(tree.nlevels):
        assert (
                cost_S.params["p_fmm_lev%d" % level]
                == fmm_level_to_order(kernel, {"k": k}, tree, level))

# }}}


# {{{ constant one wrangler

def record_op_count(f):
    def run_and_record_op_count(self, *args):
        result, op_count = f(self, *args)
        if f.__name__ in self.op_counts:
            raise RuntimeError("Would overwrite op count")
        self.op_counts[f.__name__] = op_count
        return result
    return run_and_record_op_count


class OpCountingConstantOneQBXExpansionWrangler(object):
    # This is based on ConstantOneExpansionWrangler from boxtree.

    def __init__(self, queue, geo_data):
        from pytential.qbx.fmmlib import ToHostTransferredGeoDataWrapper
        geo_data = ToHostTransferredGeoDataWrapper(queue, geo_data)

        self.geo_data = geo_data
        self.trav = geo_data.traversal()
        self.tree = geo_data.tree()

        self.op_counts = {}

    def multipole_expansion_zeros(self):
        return np.zeros(self.tree.nboxes, dtype=np.float64)

    local_expansion_zeros = multipole_expansion_zeros

    def potential_zeros(self):
        return np.zeros(self.tree.ntargets, dtype=np.float64)

    def _get_source_slice(self, ibox):
        pstart = self.tree.box_source_starts[ibox]
        return slice(
                pstart, pstart + self.tree.box_source_counts_nonchild[ibox])

    def _get_target_slice(self, ibox):
        non_qbx_box_target_lists = self.geo_data.non_qbx_box_target_lists()
        pstart = non_qbx_box_target_lists.box_target_starts[ibox]
        return slice(
                pstart, pstart
                + non_qbx_box_target_lists.box_target_counts_nonchild[ibox])

    def output_zeros(self):
        non_qbx_box_target_lists = self.geo_data.non_qbx_box_target_lists()
        return np.zeros(non_qbx_box_target_lists.nfiltered_targets)

    def full_output_zeros(self):
        from pytools.obj_array import make_obj_array
        return make_obj_array([np.zeros(self.tree.ntargets)])

    def qbx_local_expansion_zeros(self):
        return np.zeros(self.geo_data.ncenters)

    def reorder_sources(self, source_array):
        return source_array[self.tree.user_source_ids]

    def reorder_potentials(self, potentials):
        raise NotImplementedError("reorder_potentials should not "
                "be called on a QBXExpansionWrangler")

    @record_op_count
    def form_multipoles(self, level_start_source_box_nrs, source_boxes, src_weights):
        mpoles = self.multipole_expansion_zeros()
        ops = 0

        for ibox in source_boxes:
            pslice = self._get_source_slice(ibox)
            mpoles[ibox] += np.sum(src_weights[pslice])
            ops += pslice.stop - pslice.start

        return mpoles, ops

    @record_op_count
    def coarsen_multipoles(self, level_start_source_parent_box_nrs,
            source_parent_boxes, mpoles):
        tree = self.tree
        ops = 0

        # nlevels-1 is the last valid level index
        # nlevels-2 is the last valid level that could have children
        #
        # 3 is the last relevant source_level.
        # 2 is the last relevant target_level.
        # (because no level 1 box will be well-separated from another)
        for source_level in range(tree.nlevels-1, 2, -1):
            target_level = source_level - 1
            start, stop = level_start_source_parent_box_nrs[
                            target_level:target_level+2]
            for ibox in source_parent_boxes[start:stop]:
                for child in tree.box_child_ids[:, ibox]:
                    if child:
                        mpoles[ibox] += mpoles[child]
                        ops += 1

        # Modified mpoles in-place and returns None
        return None, ops

    @record_op_count
    def eval_direct(self, target_boxes, neighbor_sources_starts,
            neighbor_sources_lists, src_weights):
        pot = self.potential_zeros()
        ops = 0

        for itgt_box, tgt_ibox in enumerate(target_boxes):
            tgt_pslice = self._get_target_slice(tgt_ibox)

            src_sum = 0
            start, end = neighbor_sources_starts[itgt_box:itgt_box+2]
            #print "DIR: %s <- %s" % (tgt_ibox, neighbor_sources_lists[start:end])
            nsrcs = 0
            for src_ibox in neighbor_sources_lists[start:end]:
                src_pslice = self._get_source_slice(src_ibox)
                src_sum += np.sum(src_weights[src_pslice])
                nsrcs += src_pslice.stop - src_pslice.start

            pot[tgt_pslice] = src_sum
            ops += pot[tgt_pslice].size * nsrcs

        return pot, ops

    @record_op_count
    def multipole_to_local(self,
            level_start_target_or_target_parent_box_nrs,
            target_or_target_parent_boxes,
            starts, lists, mpole_exps):
        local_exps = self.local_expansion_zeros()
        ops = 0

        for itgt_box, tgt_ibox in enumerate(target_or_target_parent_boxes):
            start, end = starts[itgt_box:itgt_box+2]

            contrib = 0
            #print tgt_ibox, "<-", lists[start:end]
            for src_ibox in lists[start:end]:
                contrib += mpole_exps[src_ibox]
                ops += 1

            local_exps[tgt_ibox] += contrib

        return local_exps, ops

    @record_op_count
    def eval_multipoles(self, level_start_target_box_nrs, target_boxes,
            from_sep_smaller_nonsiblings_by_level, mpole_exps):
        pot = self.potential_zeros()
        ops = 0

        for ssn in from_sep_smaller_nonsiblings_by_level:
            for itgt_box, tgt_ibox in enumerate(target_boxes):
                tgt_pslice = self._get_target_slice(tgt_ibox)

                contrib = 0

                start, end = ssn.starts[itgt_box:itgt_box+2]
                nsrcs = 0
                for src_ibox in ssn.lists[start:end]:
                    contrib += mpole_exps[src_ibox]
                    nsrcs += 1

                pot[tgt_pslice] += contrib
                ops += pot[tgt_pslice].size * nsrcs

        return pot, ops

    @record_op_count
    def form_locals(self,
            level_start_target_or_target_parent_box_nrs,
            target_or_target_parent_boxes, starts, lists, src_weights):
        local_exps = self.local_expansion_zeros()
        ops = 0

        for itgt_box, tgt_ibox in enumerate(target_or_target_parent_boxes):
            start, end = starts[itgt_box:itgt_box+2]

            #print "LIST 4", tgt_ibox, "<-", lists[start:end]
            contrib = 0
            for src_ibox in lists[start:end]:
                src_pslice = self._get_source_slice(src_ibox)
                contrib += np.sum(src_weights[src_pslice])
                ops += src_pslice.stop - src_pslice.start

            local_exps[tgt_ibox] += contrib

        return local_exps, ops

    @record_op_count
    def refine_locals(self, level_start_target_or_target_parent_box_nrs,
            target_or_target_parent_boxes, local_exps):
        ops = 0

        for target_lev in range(1, self.tree.nlevels):
            start, stop = level_start_target_or_target_parent_box_nrs[
                    target_lev:target_lev+2]
            for ibox in target_or_target_parent_boxes[start:stop]:
                local_exps[ibox] += local_exps[self.tree.box_parent_ids[ibox]]
                ops += 1

        return local_exps, ops

    @record_op_count
    def eval_locals(self, level_start_target_box_nrs, target_boxes, local_exps):
        pot = self.potential_zeros()
        ops = 0

        for ibox in target_boxes:
            tgt_pslice = self._get_target_slice(ibox)
            pot[tgt_pslice] += local_exps[ibox]
            ops += pot[tgt_pslice].size

        return pot, ops

    @record_op_count
    def form_global_qbx_locals(self, src_weights):
        local_exps = self.qbx_local_expansion_zeros()
        ops = 0

        global_qbx_centers = self.geo_data.global_qbx_centers()
        qbx_center_to_target_box = self.geo_data.qbx_center_to_target_box()

        for tgt_icenter in global_qbx_centers:
            itgt_box = qbx_center_to_target_box[tgt_icenter]

            start, end = (
                    self.trav.neighbor_source_boxes_starts[itgt_box:itgt_box + 2])

            src_sum = 0
            for src_ibox in self.trav.neighbor_source_boxes_lists[start:end]:
                src_pslice = self._get_source_slice(src_ibox)
                ops += src_pslice.stop - src_pslice.start
                src_sum += np.sum(src_weights[src_pslice])

            local_exps[tgt_icenter] = src_sum

        return local_exps, ops

    @record_op_count
    def translate_box_multipoles_to_qbx_local(self, multipole_exps):
        local_exps = self.qbx_local_expansion_zeros()
        ops = 0

        global_qbx_centers = self.geo_data.global_qbx_centers()
        qbx_center_to_target_box = self.geo_data.qbx_center_to_target_box()

        for isrc_level, ssn in enumerate(self.trav.from_sep_smaller_by_level):
            for tgt_icenter in global_qbx_centers:
                icontaining_tgt_box = qbx_center_to_target_box[tgt_icenter]

                if icontaining_tgt_box == -1:
                    continue

                start, stop = (
                        ssn.starts[icontaining_tgt_box],
                        ssn.starts[icontaining_tgt_box+1])

                for src_ibox in ssn.lists[start:stop]:
                    local_exps[tgt_icenter] += multipole_exps[src_ibox]
                    ops += 1

        return local_exps, ops

    @record_op_count
    def translate_box_local_to_qbx_local(self, local_exps):
        qbx_expansions = self.qbx_local_expansion_zeros()
        ops = 0

        global_qbx_centers = self.geo_data.global_qbx_centers()
        qbx_center_to_target_box = self.geo_data.qbx_center_to_target_box()

        for tgt_icenter in global_qbx_centers:
            isrc_box = qbx_center_to_target_box[tgt_icenter]
            src_ibox = self.trav.target_boxes[isrc_box]
            qbx_expansions[tgt_icenter] += local_exps[src_ibox]
            ops += 1

        return qbx_expansions, ops

    @record_op_count
    def eval_qbx_expansions(self, qbx_expansions):
        output = self.full_output_zeros()
        ops = 0

        global_qbx_centers = self.geo_data.global_qbx_centers()
        center_to_tree_targets = self.geo_data.center_to_tree_targets()

        for src_icenter in global_qbx_centers:
            start, end = (
                    center_to_tree_targets.starts[src_icenter:src_icenter+2])
            for icenter_tgt in range(start, end):
                center_itgt = center_to_tree_targets.lists[icenter_tgt]
                output[0][center_itgt] += qbx_expansions[src_icenter]
                ops += 1

        return output, ops

    def finalize_potentials(self, potentials):
        return potentials

# }}}


# {{{ verify cost model

class OpCountingTranslationCostModel(object):
    """A translation cost model which assigns at cost of 1 to each operation."""

    def __init__(self, dim, nlevels):
        pass

    @staticmethod
    def direct():
        return 1

    p2qbxl = direct
    p2p_tsqbx = direct
    qbxl2p = direct

    @staticmethod
    def p2l(level):
        return 1

    l2p = p2l
    p2m = p2l
    m2p = p2l
    m2qbxl = p2l
    l2qbxl = p2l

    @staticmethod
    def m2m(src_level, tgt_level):
        return 1

    l2l = m2m
    m2l = m2m


STAGES = (
        "form_multipoles",
        "coarsen_multipoles",
        "eval_direct",
        "multipole_to_local",
        "eval_multipoles",
        "form_locals",
        "refine_locals",
        "eval_locals",
        "form_global_qbx_locals",
        "translate_box_local_to_qbx_local",
        "eval_qbx_expansions",
)


@pytest.mark.parametrize("dim", (2, 3))
@pytest.mark.parametrize("off_surface", (True, False))
def test_cost_model_correctness(ctx_getter, dim, off_surface):
    """Check that computed cost matches that of a constant-one FMM."""
    cl_ctx = ctx_getter()
    queue = cl.CommandQueue(cl_ctx)

    cost_model = (
            CostModel(
                translation_cost_model_factory=OpCountingTranslationCostModel))

    lpot_source = get_lpot_source(queue, dim)

    # Construct targets.
    if off_surface:
        from pytential.target import PointsTarget
        from boxtree.tools import make_uniform_particle_array
        ntargets = 10 ** 3
        targets = PointsTarget(
                make_uniform_particle_array(queue, ntargets, dim, np.float))
        target_discrs_and_qbx_sides = ((targets, 0),)
        qbx_forced_limit = None

    else:
        targets = lpot_source.density_discr
        target_discrs_and_qbx_sides = ((targets, 1),)
        qbx_forced_limit = 1

    # Run cost model for SLP.
    sigma_sym = sym.var("sigma")
    k_sym = LaplaceKernel(lpot_source.ambient_dim)
    sym_op_S = sym.S(k_sym, sigma_sym, qbx_forced_limit=qbx_forced_limit)

    inspect_geo_data_result = []

    def inspect_geo_data(insn, bound_expr, geo_data):
        kernel = lpot_source.get_fmm_kernel(insn.kernels)
        kernel_arguments = insn.kernel_arguments

        result = cost_model(geo_data, kernel, kernel_arguments)
        inspect_geo_data_result.append(result)

        return False

    lpot_source = lpot_source.copy(geometry_data_inspector=inspect_geo_data)

    op_S = bind((lpot_source, targets), sym_op_S)
    sigma = get_density(queue, lpot_source)
    op_S(queue, sigma=sigma)
    cost_S = inspect_geo_data_result[0]

    # Run FMM with ConstantOneWrangler. This can't be done with pytential's
    # high-level interface, so call the FMM driver directly.
    from pytential.qbx.fmm import drive_fmm
    geo_data = lpot_source.qbx_fmm_geometry_data(
            target_discrs_and_qbx_sides=target_discrs_and_qbx_sides)

    wrangler = OpCountingConstantOneQBXExpansionWrangler(queue, geo_data)
    nnodes = lpot_source.quad_stage2_density_discr.nnodes
    src_weights = np.ones(nnodes)

    potential = drive_fmm(wrangler, src_weights)[geo_data.ncenters:]

    # Check constant one wrangler for correctness.
    assert (potential == nnodes).all()

    op_counts = wrangler.op_counts
    modeled_time = cost_S.get_predicted_times(merge_close_lists=True)

    # Check that the cost model matches the timing data returned by the
    # constant one wrangler.
    mismatches = []
    for stage in STAGES:
        if op_counts.get(stage, 0) != modeled_time[stage]:
            mismatches.append(
                    (stage, op_counts.get(stage, 0), modeled_time[stage]))

    assert not mismatches, "\n".join(str(s) for s in mismatches)

# }}}


# {{{ test order varying by level

CONSTANT_ONE_PARAMS = dict(
        c_l2l=1,
        c_l2p=1,
        c_l2qbxl=1,
        c_m2l=1,
        c_m2m=1,
        c_m2p=1,
        c_m2qbxl=1,
        c_p2l=1,
        c_p2m=1,
        c_p2p=1,
        c_p2qbxl=1,
        c_qbxl2p=1,
        )


def test_cost_model_order_varying_by_level(ctx_getter):
    """For FMM order varying by level, this checks to ensure that the costs are
    different. The varying-level case should have larger cost.
    """

    cl_ctx = ctx_getter()
    queue = cl.CommandQueue(cl_ctx)

    # {{{ constant level to order

    def level_to_order_constant(kernel, kernel_args, tree, level):
        return 1

    lpot_source = get_lpot_source(queue, 2).copy(
            fmm_level_to_order=level_to_order_constant)

    cost_constant = (
            get_slp_cost(queue, lpot_source)
            .with_params(CONSTANT_ONE_PARAMS))

    # }}}

    # {{{ varying level to order

    varying_order_params = cost_constant.params.copy()

    nlevels = cost_constant.params["nlevels"]
    for level in range(nlevels):
        varying_order_params["p_fmm_lev%d" % level] = nlevels - level

    cost_varying = cost_constant.with_params(varying_order_params)

    # }}}

    assert (
            sum(cost_varying.get_predicted_times().values())
            > sum(cost_constant.get_predicted_times().values()))

# }}}


# You can test individual routines by typing
# $ python test_cost_model.py 'test_routine()'

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from pytest import main
        main([__file__])


# vim: foldmethod=marker
