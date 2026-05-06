

from dataclasses import dataclass
from typing import Callable, Optional, Union

import torch

from boundlab.sparse.table import TorchTable, Indices
from boundlab.sparse.tn import TN, Dense, Dim
IndicesOrNone = Union[Indices, None]

DEBUG_MultiCOOTensor = False
DEBUG_NO_MD_EYE_OPT = False
class COOSparsify:
    input_dim: Dim
    output_dims: list[Dim]
    torch_table: TorchTable
    symbolic_supersets: list["COOSparsify"]
    
    def __init__(
        self,
        input_dim: Dim,
        torch_table: TorchTable,
        symbolic_supersets: list["COOSparsify"] | None = None,
    ):
        self.input_dim = input_dim
        self.output_dims = list(torch_table.columns)
        self.torch_table = torch_table
        assert self.input_dim.length == self.torch_table.length
        self._infer_table_flags()
        self.symbolic_supersets = list(symbolic_supersets or [])
        self.__post_init__()

    def _infer_table_flags(self) -> None:
        if self.torch_table.is_sorted and self.torch_table.is_unique:
            return
        mat = self.torch_table.materialize()
        if mat.shape[0] <= 1:
            self.torch_table.is_sorted = True
            self.torch_table.is_unique = True
            return
        if mat.shape[1] == 0:
            self.torch_table.is_sorted = True
            self.torch_table.is_unique = mat.shape[0] <= 1
            return

        sorted_table, _ = self.torch_table.sort_dedup()
        if sorted_table.length == self.torch_table.length:
            self.torch_table.is_unique = True
            self.torch_table.is_sorted = torch.equal(mat, sorted_table.materialize())

    def __post_init__(self):
        assert self.input_dim.length == self.torch_table.length
        assert self.torch_table.columns == self.output_dims, "TorchTable columns must match output_dims."
        # TODO: sort the output_dims and torch_table.
        sorted_dims = sorted(self.output_dims)
        if self.output_dims != sorted_dims:
            positions = [self.output_dims.index(dim) for dim in sorted_dims]
            self.output_dims = sorted_dims
            self.torch_table = TorchTable(
                columns=[self.torch_table.columns[pos] for pos in positions],
                data=[self.torch_table.data[pos] for pos in positions],
                length=self.torch_table.length,
            )

    def forward(self, x: Union[Dense, TN]) -> Union[Dense, TN]:
        # TODO: for TN, deal with each Dense factor separately.
        # for Dense, if the input_dim is not present, do not change anything;
        # if the input_dim is present, apply the sparsification 
        #    initialize a new Dense with the output_dims and without intput_dim
        #    then `new_tensor[...torch_table...] = original_tensor`
        if isinstance(x, TN):
            return TN(factors=[self.forward(factor) for factor in x.factors])
        if self.input_dim not in x.dims:
            return x.clone()
        if self.is_md_eye() and not DEBUG_NO_MD_EYE_OPT:
            return x.diagonal_embed(self.input_dim, self.output_dims)

        other_dims = [dim for dim in x.dims if dim is not self.input_dim]
        perm = [x.dims.index(dim) for dim in other_dims + [self.input_dim]]
        tensor = x.tensor.permute(perm)
        other_shape = [dim.length for dim in other_dims]
        output_shape = [dim.length for dim in self.output_dims]
        result = torch.zeros(
            other_shape + output_shape,
            dtype=x.tensor.dtype,
            device=x.tensor.device,
        )

        materialized = self.torch_table.materialize().to(device=x.tensor.device)
        index = (...,) + tuple(materialized[:, idx] for idx in range(len(self.output_dims)))
        result[index] = tensor
        return Dense(tensor=result, dims=other_dims + self.output_dims)

    def backward(self, y: Union[Dense, TN]) -> Union[Dense, TN]:
        # TODO: for TN, deal with each Dense factor separately.
        # for Dense, if none of output_dims are present, do not change anything;
        # if any output_dim is present, 
        #     `new_tensor = original_tensor[...torch_table...]` with the input_dim and without output_dims
        assert self.input_dim.length == self.torch_table.length
        if isinstance(y, TN):
            return TN(factors=[self.backward(factor) for factor in y.factors])

        present_output_dims = [dim for dim in self.output_dims if dim in y.dims]
        if len(present_output_dims) == 0:
            return y.clone()
        if self.is_md_eye() and not DEBUG_NO_MD_EYE_OPT:
            return y.diagonal(present_output_dims, self.input_dim)

        other_dims = [dim for dim in y.dims if dim not in present_output_dims]
        perm = [y.dims.index(dim) for dim in other_dims + present_output_dims]
        tensor = y.tensor.permute(perm)
        materialized = self.torch_table.materialize().to(device=y.tensor.device)
        index = (...,) + tuple(
            materialized[:, self.output_dims.index(dim)]
            for dim in present_output_dims
        )
        result = tensor[index]
        # TODO-test: test forward and backward on TN reproduce the same result. Use a TN with at least 5 connected factors.
        return Dense(tensor=result, dims=other_dims + [self.input_dim])
    
    @staticmethod
    def is_all_connect(*args: "COOSparsify") -> bool:
        if len(args) <= 1:
            return True

        dim_sets = [set(op.output_dims) for op in args]
        seen = {0}
        stack = [0]
        while stack:
            idx = stack.pop()
            for other_idx, other_dims in enumerate(dim_sets):
                if other_idx in seen:
                    continue
                if dim_sets[idx] & other_dims:
                    seen.add(other_idx)
                    stack.append(other_idx)
        return len(seen) == len(args)
    
    @staticmethod
    def merge(*args: "COOSparsify") -> "COOSparsify":
        if (
            all(op.is_md_eye() for op in args)
            and COOSparsify.is_all_connect(*args)
            and not DEBUG_NO_MD_EYE_OPT
        ):
            all_dims = list(sorted(set(d for op in args for d in op.output_dims)))
            input_dim = Dim(
                length=args[0].input_dim.length,
                ordering=sum(op.input_dim.ordering for op in args) / len(args),
            )
            return COOSparsify(
                input_dim=input_dim,
                torch_table=TorchTable(
                    columns=all_dims,
                    data=[None] * len(all_dims),
                    length=input_dim.length,
                ),
                symbolic_supersets=list(args),
            )

        # TODO: Call torch_table.merge on all the tables, and construct a new COOSparsify with the merged table and the union of all output_dims.
        # Assume input_dims are different. Create a new input_dim (ordering is the mean of all input_dims).
        # Add all args into `symbolic_supersets` of the new COOSparsify.
        assert len(args) > 0, "merge() requires at least one COOSparsify."
        # assert len(set(op.input_dim for op in args)) == len(args), "COOSparsify.merge assumes input_dims are different."
        tables = []
        for op in args:
            table = op.torch_table
            if not (table.is_sorted and table.is_unique):
                table, _ = table.sort_dedup()
            tables.append(table)
        merged_table = TorchTable.merge(tables)

        output_dims = list(merged_table.columns)
        input_dim = Dim(
            length=merged_table.length,
            ordering=sum(op.input_dim.ordering for op in args) / len(args),
        )
        return COOSparsify(
            input_dim=input_dim,
            torch_table=merged_table,
            symbolic_supersets=list(args),
        )
    
    @staticmethod
    def md_eye(input_dim: Dim, output_dims: list[Dim]) -> "COOSparsify":
        assert all(dim.length == input_dim.length for dim in output_dims), "All output_dims must have the same length as input_dim."
        return COOSparsify(
            input_dim=input_dim,
            torch_table=TorchTable(
                columns=output_dims,
                data=[None] * len(output_dims),
                length=input_dim.length,
            ),
        )

    def is_md_eye(self) -> bool:
        return (
            all(data is None for data in self.torch_table.data)
            and self.torch_table.length == self.input_dim.length
            and len(self.output_dims) > 0
            and all(dim.length == self.input_dim.length for dim in self.output_dims)
        )
    
    def is_symbolic_subset(self, other: "COOSparsify", suppress_error: bool = False) -> bool | None:
        if self is other:
            return True
        if not DEBUG_NO_MD_EYE_OPT:
            if other.is_md_eye() and self.is_md_eye() and set(self.output_dims) <= set(other.output_dims):
                return True
            
            if other.is_md_eye() and len(other.output_dims) == 1:
                return True
            
        if other in self.symbolic_supersets:
            return True
        elif self in other.symbolic_supersets:
            return False
        elif set(self.output_dims).isdisjoint(set(other.output_dims)):
            return False
        else:
            if suppress_error:
                return None
            raise RuntimeError(f"Cannot symbolically determine symbolic superset relationship between these two COOSparsify based on their output_dims. Please explicitly add one as a symbolic superset {self} {other}.")

    def inverse_supersets(self, *superset_op: "COOSparsify") -> "COOSparsify":
        if self.is_md_eye() and not DEBUG_NO_MD_EYE_OPT:
            assert all(op.is_md_eye() for op in superset_op), "All superset_ops must be MD eye tables if this COOSparsify is an MD eye table."
            return COOSparsify.md_eye(
                input_dim=self.input_dim,
                output_dims=list(sorted(set(op.input_dim for op in superset_op))),
            )
        assert all(self.is_symbolic_subset(superset) for superset in superset_op), "All provided superset_ops must be symbolic supersets of this COOSparsify."
        output_dims = list(sorted(set(op.input_dim for op in superset_op)))
        # TODO: Return a new COOSparsify with `output_dims` as the output_dims
        # Call TorchTable.index for each `superset_op`` to get the indices for each output_dim, and return the list of indices as the new `torch_table`.
        # Assert the new table is unique. Do not need to sort it.
        data = []
        for op in superset_op:
            columns = list(op.output_dims)
            projected = TorchTable(
                columns=columns,
                data=[self.torch_table.column_data(column) for column in columns],
                length=self.torch_table.length,
            )
            data.append(op.torch_table.index(projected))

        data_by_dim = {op.input_dim: dat for op, dat in zip(superset_op, data)}
        # TODO-test: test superset_op.backward(self.forward(x)) == x for some random TN with at least 5 connected factors, and some random input x that has the input_dim of self.
        # Test superset_op.forward(self.backward(x)) == x in the same setings.

        return COOSparsify(
            input_dim=self.input_dim,
            torch_table=TorchTable(
                columns=output_dims,
                data=[data_by_dim[dim] for dim in output_dims],
                length=self.input_dim.length,
            ),
        )


    def filter_dims(self, dims: list[Dim]) -> tuple["COOSparsify", IndicesOrNone]:
        if self.is_md_eye() and not DEBUG_NO_MD_EYE_OPT:
            new_output_dims = [dim for dim in self.output_dims if dim in dims]
            data = [None] * len(new_output_dims)
            new_op = COOSparsify(
                input_dim=self.input_dim,
                torch_table=TorchTable(
                    columns=new_output_dims,
                    data=data,
                    length=self.input_dim.length,
                ),
            )
            return new_op, None
        # TODO: call `torch_table.filter_dims(dims)` to get the new table and the indices for each output_dim. Return a new COOSparsify with the filtered table and only the output_dims that are in `dims`. Assert the new table is unique. Do not need to sort it.
        new_output_dims = [dim for dim in self.output_dims if dim in dims]
        columns = list(new_output_dims)
        new_table, indices = self.torch_table.filter_columns(columns)
        assert new_table.is_unique, "Filtered table must be unique."
        new_input_dim = Dim(
            length=new_table.length,
            ordering=self.input_dim.ordering,
            name=self.input_dim.name,
        )
        return COOSparsify(
            input_dim=new_input_dim,
            torch_table=new_table,
            symbolic_supersets=[self],
        ), indices
    
    def __str__(self):
        if self.is_md_eye() and not DEBUG_NO_MD_EYE_OPT:
            return f"MdEye({self.input_dim} -> {self.output_dims})"
        return f"COO({self.input_dim} -> {self.output_dims})"

@dataclass
class MultiCOOSparsify:
    ops: list[COOSparsify]

    def __post_init__(self):
        # TODO: assert all ops have disjoint output_dims
        l = 0
        s = set()
        for o in self.ops:
            l += len(o.output_dims)
            s |= set(o.output_dims)
        assert l == len(s), "MultiCOOSparsify ops must have disjoint output_dims."
        assert len(set(o.input_dim for o in self.ops)) == len(self.ops), "MultiCOOSparsify ops must have different input_dims."

    def forward(self, x: Union[Dense, TN]) -> Union[Dense, TN]:
        for op in self.ops:
            x = op.forward(x)
        return x
    
    def backward(self, y: Union[Dense, TN]) -> Union[Dense, TN]:
        for op in reversed(self.ops):
            y = op.backward(y)
        return y
    
    def merge(self, other: "MultiCOOSparsify") -> "MultiCOOSparsify":
        # TODO: track the connective component of all MultiCOOSparsify,
        # then call COOSparsify.merge on all connective components
        ops = list(self.ops) + list(other.ops)
        components: list[list[COOSparsify]] = []
        for op in ops:
            op_dims = set(op.output_dims)
            matched = []
            for component in components:
                component_dims = set(dim for component_op in component for dim in component_op.output_dims)
                if op_dims & component_dims:
                    matched.append(component)
            if len(matched) == 0:
                components.append([op])
                continue

            merged_component = [op]
            for component in matched:
                merged_component.extend(component)
                components.remove(component)
            components.append(merged_component)

        return MultiCOOSparsify(ops=[
            COOSparsify.merge(*component) for component in components
        ])
    
    def is_symbolic_subset(self, other: "MultiCOOSparsify", suppress_error: bool = False) -> bool:
        return all(any(op.is_symbolic_subset(other_op, suppress_error=suppress_error) for other_op in other.ops) for op in self.ops)
    
    def inverse_supersets(self, superset_op: "MultiCOOSparsify") -> "MultiCOOSparsify":
        # TODO: for each op in ops, find all superset_ops in superset_op that are symbolic supersets of it,
        # call `op.inverse_supersets` on those superset_ops to get the new op and the indices,
        # and return a new MultiCOOSparsify with all the new ops.
        new_ops = []
        for op in self.ops:
            supersets = [
                other_op for other_op in superset_op.ops
                if op.is_symbolic_subset(other_op)
            ]
            assert len(supersets) > 0, "Every op must have at least one symbolic superset."
            new_op = op.inverse_supersets(*supersets)
            assert new_op.input_dim.length == new_op.torch_table.length
            new_ops.append(new_op)
        return MultiCOOSparsify(ops=new_ops)

    def filter_dims(self, dims: list[Dim]) -> tuple["MultiCOOSparsify", list[tuple[Dim, IndicesOrNone]]]:
        # Call `filter_dims` for each op, and return a new MultiCOOSparsify with the filtered ops and the list of (output_dim, indices) for each op.
        new_ops = []
        indices = []
        for op in self.ops:
            new_op, index = op.filter_dims(dims)
            new_ops.append(new_op)
            indices.append((new_op.input_dim, index))
        return MultiCOOSparsify(ops=new_ops), indices

    def __str__(self):
        return " & ".join(str(op) for op in self.ops)

@dataclass
class MultiCOOTensor:
    tn: TN
    sparsify: MultiCOOSparsify
    debug_dense: Optional[Dense] = None

    def dense(self) -> torch.Tensor:
        return self.to_dense().tensor

    def clone(self) -> "MultiCOOTensor":
        return MultiCOOTensor(
            tn=self.tn.clone(),
            sparsify=self.sparsify,
            debug_dense=self.debug_dense.clone() if self.debug_dense is not None else None,
        )

    def real_numel(self) -> int:
        return self.tn.real_numel()

    def to_dense(self) -> Dense:
        tn = self.tn.clone()
        for op in self.sparsify.ops:
            if op.input_dim not in tn.dims:
                tn.factors.append(
                    Dense(
                        tensor=torch.ones(op.input_dim.length),
                        dims=[op.input_dim],
                    )
                )
        return self.sparsify.forward(tn).to_dense()
    
    @property
    def dims(self) -> list[Dim]:
        dims: list[Dim] = []
        for op in self.sparsify.ops:
            dims.extend(op.output_dims)
        return sorted(dims)
    
    @property
    def shape(self) -> torch.Size:
        return torch.Size([dim.length for dim in self.dims])
    
    def expand_to(self, op: MultiCOOSparsify) -> "MultiCOOTensor":
        assert self.sparsify.is_symbolic_subset(op, suppress_error=True), "Can only expand to a MultiCOOSparsify that is a symbolic superset of the current sparsify."
        tn = self.sparsify.inverse_supersets(op).forward(self.tn)
        return MultiCOOTensor(tn=tn, sparsify=op)

    def shrink_to(self, op: MultiCOOSparsify) -> "MultiCOOTensor":
        assert op.is_symbolic_subset(self.sparsify), "Can only shrink to a MultiCOOSparsify that is a symbolic superset of the current sparsify."
        tn = op.inverse_supersets(self.sparsify).backward(self.tn)
        return MultiCOOTensor(tn=tn, sparsify=op)

    def __add__(self, other: "MultiCOOTensor") -> Optional["MultiCOOTensor"]:
        if self.sparsify.is_symbolic_subset(other.sparsify, suppress_error=True):
            return MultiCOOTensor(
                tn=self.expand_to(other.sparsify).tn + other.tn,
                sparsify=other.sparsify
            )
        elif other.sparsify.is_symbolic_subset(self.sparsify, suppress_error=True):
            return MultiCOOTensor(
                tn=self.tn + other.expand_to(self.sparsify).tn,
                sparsify=self.sparsify
            )
        else:
            return None
    
    def __neg__(self) -> "MultiCOOTensor":
        return MultiCOOTensor(
            tn=-self.tn,
            sparsify=self.sparsify
        )
    
    def __sub__(self, other: "MultiCOOTensor") -> Optional["MultiCOOTensor"]:
        neg_other = -other
        return self.__add__(neg_other)

    def __mul__(self, other: "MultiCOOTensor") -> "MultiCOOTensor":
        op = self.sparsify.merge(other.sparsify)
        self_0 = self.shrink_to(op)
        other_0 = other.shrink_to(op)
        result = MultiCOOTensor(
            tn=self_0.tn * other_0.tn,
            sparsify=op
        )
        if DEBUG_MultiCOOTensor:
            assert result.to_dense().allclose(self.to_dense() * other.to_dense())
        # TODO-test: test this function with `DEBUG_MultiCOOTensor` enabled. Generate random high dimensional MCTs with at least 8 connected factors and at least 6 different COOSparsify ops (each with at least 2 output_dims).
        return result
    
    def apply_multiplicative(self, f: Callable[[torch.Tensor], torch.Tensor]) -> "MultiCOOTensor":
        return MultiCOOTensor(
            tn=self.tn.apply_multiplicative(f),
            sparsify=self.sparsify
        )
    
    def sum(self, dims: list[Dim]) -> "MultiCOOTensor":
        # TODO: 
        # For all partial covered `COOSparsify`, call `filter_dims` to get `indices`, call `index_reduce_sum` with `indices`.
        # After that, call `self.tn.sum` on the remaining dims that are not fully reduced by the previous step.
        dims_to_reduce = set(dims)
        tn = self.tn
        new_ops = []
        tn_sum_dims = [dim for dim in dims if dim in tn.dims]

        for op in self.sparsify.ops:
            reduced_outputs = [dim for dim in op.output_dims if dim in dims_to_reduce]
            if len(reduced_outputs) == 0:
                new_ops.append(op)
                continue

            kept_outputs = [dim for dim in op.output_dims if dim not in dims_to_reduce]
            if len(kept_outputs) == 0:
                tn_sum_dims.append(op.input_dim)
                continue

            new_op, index = op.filter_dims(kept_outputs)
            if index is not None:
                tn = tn.index_reduce_sum(op.input_dim, index, new_op.input_dim)
            new_ops.append(new_op)

        result = MultiCOOTensor(
            tn=tn.sum(tn_sum_dims),
            sparsify=MultiCOOSparsify(ops=new_ops),
        )

        if DEBUG_MultiCOOTensor:
            assert self.to_dense().sum(dims).allclose(result.to_dense())
        # TODO-test: test this function with `DEBUG_MultiCOOTensor` enabled. Generate random high dimensional MCTs with at least 8 connected factors and at least 6 different COOSparsify ops (each with at least 2 output_dims).
        return result

    def tensordot(self, other: "MultiCOOTensor", dims) -> "MultiCOOTensor":
        if isinstance(dims, tuple) and len(dims) == 2:
            left_axes, right_axes = dims
            assert len(left_axes) == len(right_axes), "tensordot axes must have the same length."
            dims = [self.dims[axis] for axis in left_axes]
        return (self * other).sum(dims)
    
    def add_intersection_to(self, other: "MultiCOOTensor", neg: bool = False) -> "MultiCOOTensor":
        op = self.sparsify.merge(other.sparsify)
        tmp = self.shrink_to(op)
        if neg:
            tmp = -tmp
        return other + tmp
    

@dataclass
class MultiCOOTensorSum:
    terms: list[MultiCOOTensor]

    def __post_init__(self):
        self._assert_sorted()

    def add_term(self, term: MultiCOOTensor):
        for t in range(len(self.terms)):
            if term.real_numel() > self.terms[t].real_numel():
                break
            if tensor := self.terms[t].__add__(term):
                self.terms[t] = tensor
                self._assert_sorted()
                return
        else:
            t = len(self.terms)
        rest = self.terms[t:]
        self.terms = self.terms[:t] + [term]
        for term in rest:
            if tensor := term.__add__(self.terms[t]):
                self.terms[t] = tensor
            else:
                self.terms.append(term)
        self._assert_sorted()

    def _assert_sorted(self):
        assert all(
            t.real_numel() >= self.terms[i + 1].real_numel()
            for i, t in enumerate(self.terms[:-1])
        ), "Terms must be sorted by real_numel in descending order"

    def apply_multiplicative(
        self, fn: Callable[[torch.Tensor], torch.Tensor]
    ) -> "MultiCOOTensorSum":
        old_terms = self.terms
        new_terms: list[MultiCOOTensor] = []
        for i, t1 in enumerate(old_terms):
            new_t = t1.clone()
            for t2 in old_terms[:i]:
                new_t = t2.add_intersection_to(new_t)
            new_t = new_t.apply_multiplicative(fn)
            for t in new_terms:
                new_t = t.add_intersection_to(new_t, neg=True)
            new_terms.append(new_t)
        return MultiCOOTensorSum(new_terms)


__all__ = [
    "COOSparsify",
    "MultiCOOSparsify",
    "MultiCOOTensor",
    "MultiCOOTensorSum",
]




    
    
    
