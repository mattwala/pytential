from __future__ import division

__copyright__ = "Copyright (C) 2010-2013 Andreas Kloeckner"

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


from pytools import Record, memoize_method
from pymbolic.primitives import cse_scope
from pytential.symbolic.mappers import IdentityMapper


# {{{ instructions ------------------------------------------------------------

class Instruction(Record):
    __slots__ = ["dep_mapper_factory"]
    priority = 0

    def get_assignees(self):
        raise NotImplementedError("no get_assignees in %s" % self.__class__)

    def get_dependencies(self):
        raise NotImplementedError("no get_dependencies in %s" % self.__class__)

    def __str__(self):
        raise NotImplementedError

    def get_exec_function(self, exec_mapper):
        raise NotImplementedError


class Assign(Instruction):
    # attributes: names, exprs, do_not_return, priority
    #
    # do_not_return is a list of bools indicating whether the corresponding
    # entry in names and exprs describes an expression that is not needed
    # beyond this assignment

    comment = ""

    def __init__(self, names, exprs, **kwargs):
        Instruction.__init__(self, names=names, exprs=exprs, **kwargs)

        if not hasattr(self, "do_not_return"):
            self.do_not_return = [False] * len(names)

    def get_assignees(self):
        return set(self.names)

    def get_dependencies(self):
        try:
            return self._dependencies
        except:
            # arg is include_subscripts
            dep_mapper = self.dep_mapper_factory()

            from operator import or_
            deps = reduce(
                    or_, (dep_mapper(expr)
                    for expr in self.exprs))

            from pymbolic.primitives import Variable
            deps -= set(Variable(name) for name in self.names)

            self._dependencies = deps

            return deps

    def __str__(self):
        comment = self.comment
        if len(self.names) == 1:
            if comment:
                comment = "/* %s */ " % comment

            return "%s <- %s%s" % (self.names[0], comment, self.exprs[0])
        else:
            if comment:
                comment = " /* %s */" % comment

            lines = []
            lines.append("{" + comment)
            for n, e, dnr in zip(self.names, self.exprs, self.do_not_return):
                if dnr:
                    dnr_indicator = "-#"
                else:
                    dnr_indicator = ""

                lines.append("  %s <%s- %s" % (n, dnr_indicator, e))
            lines.append("}")
            return "\n".join(lines)

    def get_exec_function(self, exec_mapper):
        return exec_mapper.exec_assign

# }}}


# {{{ graphviz/dot dataflow graph drawing

def dot_dataflow_graph(code, max_node_label_length=30,
        label_wrap_width=50):
    origins = {}
    node_names = {}

    result = [
            "initial [label=\"initial\"]"
            "result [label=\"result\"]"]

    for num, insn in enumerate(code.instructions):
        node_name = "node%d" % num
        node_names[insn] = node_name
        node_label = str(insn)

        if max_node_label_length is not None:
            node_label = node_label[:max_node_label_length]

        if label_wrap_width is not None:
            from pytools import word_wrap
            node_label = word_wrap(node_label, label_wrap_width,
                    wrap_using="\n      ")

        node_label = node_label.replace("\n", "\\l") + "\\l"

        result.append("%s [ label=\"p%d: %s\" shape=box ];" % (
            node_name, insn.priority, node_label))

        for assignee in insn.get_assignees():
            origins[assignee] = node_name

    def get_orig_node(expr):
        from pymbolic.primitives import Variable
        if isinstance(expr, Variable):
            return origins.get(expr.name, "initial")
        else:
            return "initial"

    def gen_expr_arrow(expr, target_node):
        result.append("%s -> %s [label=\"%s\"];"
                % (get_orig_node(expr), target_node, expr))

    for insn in code.instructions:
        for dep in insn.get_dependencies():
            gen_expr_arrow(dep, node_names[insn])

    from pytools.obj_array import is_obj_array

    if is_obj_array(code.result):
        for subexp in code.result:
            gen_expr_arrow(subexp, "result")
    else:
        gen_expr_arrow(code.result, "result")

    return "digraph dataflow {\n%s\n}\n" % "\n".join(result)

# }}}


# {{{ code representation

class Code(object):
    def __init__(self, instructions, result):
        self.instructions = instructions
        self.result = result
        self.last_schedule = None

    def dump_dataflow_graph(self):
        from pytools.debug import open_unique_debug_file

        open_unique_debug_file("dataflow", ".dot")\
                .write(dot_dataflow_graph(self, max_node_label_length=None))

    def __str__(self):
        lines = []
        for insn in self.instructions:
            lines.extend(str(insn).split("\n"))
        lines.append("RESULT: " + str(self.result))

        return "\n".join(lines)

    # {{{ dynamic scheduler (generates static schedules by self-observation)
    class NoInstructionAvailable(Exception):
        pass

    @memoize_method
    def get_next_step(self, available_names, done_insns):
        from pytools import all, argmax2
        available_insns = [
                (insn, insn.priority) for insn in self.instructions
                if insn not in done_insns
                and all(dep.name in available_names
                    for dep in insn.get_dependencies())]

        if not available_insns:
            raise self.NoInstructionAvailable

        from pytools import flatten
        discardable_vars = set(available_names) - set(flatten(
            [dep.name for dep in insn.get_dependencies()]
            for insn in self.instructions
            if insn not in done_insns))

        # {{{ make sure results do not get discarded
        from pytools.obj_array import with_object_array_or_scalar

        from pytential.symbolic.mappers import DependencyMapper
        dm = DependencyMapper(composite_leaves=False)

        def remove_result_variable(result_expr):
            # The extra dependency mapper run is necessary
            # because, for instance, subscripts can make it
            # into the result expression, which then does
            # not consist of just variables.

            for var in dm(result_expr):
                from pymbolic.primitives import Variable
                assert isinstance(var, Variable)
                discardable_vars.discard(var.name)

        with_object_array_or_scalar(remove_result_variable, self.result)
        # }}}

        return argmax2(available_insns), discardable_vars

    def execute(self, exec_mapper, pre_assign_check=None):
        """Execute the instruction stream, make all scheduling decisions
        dynamically.
        """

        context = exec_mapper.context

        done_insns = set()

        while True:
            insn = None
            discardable_vars = []

            # pick the next insn
            if insn is None:
                try:
                    insn, discardable_vars = self.get_next_step(
                            frozenset(context.keys()),
                            frozenset(done_insns))

                except self.NoInstructionAvailable:
                    # no available instructions: we're done
                    break
                else:
                    for name in discardable_vars:
                        del context[name]

                    done_insns.add(insn)
                    assignments, new_futures = (
                            insn.get_exec_function(exec_mapper)
                            (exec_mapper.queue, insn, exec_mapper.bound_expr,
                                exec_mapper))

            if insn is not None:
                for target, value in assignments:
                    if pre_assign_check is not None:
                        pre_assign_check(target, value)

                    context[target] = value

                assert not new_futures

        if len(done_insns) < len(self.instructions):
            print "Unreachable instructions:"
            for insn in set(self.instructions) - done_insns:
                print "    ", str(insn).replace("\n", "\n     ")
                from pymbolic import var
                print "     missing: ", ", ".join(
                        str(s) for s in
                        set(insn.get_dependencies())
                        - set(var(v) for v in context.iterkeys()))

            raise RuntimeError("not all instructions are reachable"
                    "--did you forget to pass a value for a placeholder?")

        from pytools.obj_array import with_object_array_or_scalar
        return with_object_array_or_scalar(exec_mapper, self.result)

    # }}}

# }}}


# {{{ compiler

class OperatorCompiler(IdentityMapper):
    def __init__(self, discretizations, prefix="_expr",
            max_vectors_in_batch_expr=None):
        IdentityMapper.__init__(self)
        self.discretizations = discretizations
        self.prefix = prefix

        self.max_vectors_in_batch_expr = max_vectors_in_batch_expr

        self.code = []
        self.expr_to_var = {}

        self.assigned_names = set()

    @memoize_method
    def dep_mapper_factory(self, include_subscripts=False):
        from pytential.symbolic.mappers import DependencyMapper
        self.dep_mapper = DependencyMapper(
                #include_operator_bindings=False,
                include_subscripts=include_subscripts,
                include_calls="descend_args")

        return self.dep_mapper

    # {{{ top-level driver

    def __call__(self, expr):
        # {{{ collect operators by operand

        from pytential.symbolic.mappers import OperatorCollector
        from pytential.symbolic.primitives import IntG

        operators = [
                op
                for op in OperatorCollector()(expr)
                if isinstance(op, IntG)]

        self.group_to_operators = {}
        for op in operators:
            features = self.discretizations[op.source].op_group_features(op)
            self.group_to_operators.setdefault(features, set()).add(op)

        # }}}

        # Traverse the expression, generate code.

        result = IdentityMapper.__call__(self, expr)

        # Put the toplevel expressions into variables as well.

        from pytools.obj_array import with_object_array_or_scalar
        result = with_object_array_or_scalar(self.assign_to_new_var, result)

        return Code(self.code, result)

    # }}}

    # {{{ variables and names

    def get_var_name(self, prefix=None):
        def generate_suffixes():
            yield ""
            i = 2
            while True:
                yield "_%d" % i
                i += 1

        def generate_plain_names():
            i = 0
            while True:
                yield self.prefix + str(i)
                i += 1

        if prefix is None:
            for name in generate_plain_names():
                if name not in self.assigned_names:
                    break
        else:
            for suffix in generate_suffixes():
                name = prefix + suffix
                if name not in self.assigned_names:
                    break

        self.assigned_names.add(name)
        return name

    def assign_to_new_var(self, expr, priority=0, prefix=None):
        from pymbolic.primitives import Variable, Subscript

        # Observe that the only things that can be legally subscripted
        # are variables. All other expressions are broken down into
        # their scalar components.
        if isinstance(expr, (Variable, Subscript)):
            return expr

        new_name = self.get_var_name(prefix)
        self.code.append(self.make_assign(new_name, expr, priority))

        return Variable(new_name)

    # }}}

    # {{{ map_xxx routines

    def map_common_subexpression(self, expr):
        if expr.scope != cse_scope.EXPRESSION:
            from warnings import warn
            warn("mishandling CSE scope")
        try:
            return self.expr_to_var[expr.child]
        except KeyError:
            priority = getattr(expr, "priority", 0)

            from pytential.symbolic.primitives import IntG
            if isinstance(expr.child, IntG):
                # We need to catch operators here and
                # treat them specially. They get assigned to their
                # own variable by default, which would mean the
                # CSE prefix would be omitted.

                rec_child = self.rec(expr.child, name_hint=expr.prefix)
            else:
                rec_child = self.rec(expr.child)

            cse_var = self.assign_to_new_var(rec_child,
                    priority=priority, prefix=expr.prefix)

            self.expr_to_var[expr.child] = cse_var
            return cse_var

    def make_assign(self, name, expr, priority):
        return Assign(names=[name], exprs=[expr],
                dep_mapper_factory=self.dep_mapper_factory,
                priority=priority)

    def map_int_g(self, expr, name_hint=None):
        try:
            return self.expr_to_var[expr]
        except KeyError:
            # make sure operator assignments stand alone and don't get muddled
            # up in vector arithmetic
            density_var = self.assign_to_new_var(self.rec(expr.density))

            src_discr = self.discretizations[expr.source]

            group = self.group_to_operators[src_discr.op_group_features(expr)]
            names = [self.get_var_name() for op in group]

            from pytential.symbolic.mappers import ExpressionKernelIdentityMapper
            ekim = ExpressionKernelIdentityMapper(self.rec)

            from pytential.discretization import (
                    LayerPotentialInstruction, LayerPotentialOutput)
            outputs = [
                    LayerPotentialOutput(
                        name=name,
                        kernel=ekim(op.kernel),
                        target_name=op.target,
                        qbx_forced_limit=op.qbx_forced_limit,
                        )
                    for name, op in zip(names, group)
                    ]

            self.code.append(
                    LayerPotentialInstruction(
                        outputs=outputs,
                        density=density_var,
                        source=expr.source,
                        priority=max(getattr(op, "priority", 0) for op in group),
                        dep_mapper_factory=self.dep_mapper_factory))

            from pytential.symbolic.primitives import Variable
            for name, group_expr in zip(names, group):
                self.expr_to_var[group_expr] = Variable(name)

            return self.expr_to_var[expr]

    def map_int_g_ds(self, op):
        assert False

    # }}}

# }}}

# vim: foldmethod=marker
