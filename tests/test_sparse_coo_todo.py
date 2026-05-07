import torch
import pytest

from boundlab import utils
from boundlab.sparse import coo
from boundlab.sparse.coo import COOSparsify, MultiCOOSparsify, MultiCOOTensor, MultiCOOTensorSum
from boundlab.sparse.table import TorchTable
from boundlab.sparse.tn import Dense, Dim, TN


def _mark_unique_sorted(table: TorchTable) -> TorchTable:
    table.is_sorted = True
    table.is_unique = True
    return table


def _diag_table(*dims: Dim) -> TorchTable:
    return _mark_unique_sorted(
        TorchTable(
            columns=list(dims),
            data=[None for _ in dims],
            length=dims[0].length,
        )
    )


def _random_unique_table(dims: list[Dim], length: int, seed: int) -> TorchTable:
    generator = torch.Generator().manual_seed(seed)
    grid = torch.cartesian_prod(*[torch.arange(dim.length) for dim in dims])
    perm = torch.randperm(grid.shape[0], generator=generator)[:length]
    rows = grid[perm]
    if len(dims) == 1:
        rows = rows.unsqueeze(1)
    table = TorchTable(
        columns=list(dims),
        data=[rows[:, i].clone() for i in range(len(dims))],
        length=length,
    )
    table, _ = table.sort_dedup()
    return table


def _dense_allclose(actual: Dense, expected: Dense, atol: float = 1e-6) -> None:
    assert actual.dims == expected.dims
    assert torch.allclose(actual.tensor, expected.tensor, atol=atol)


def _connected_tn(dims: list[Dim], seed: int) -> TN:
    torch.manual_seed(seed)
    factors = []
    for i in range(len(dims)):
        left = dims[i]
        right = dims[(i + 1) % len(dims)]
        factors.append(Dense(torch.randn(left.length, right.length), [left, right]))
    factors.append(Dense(torch.randn(dims[0].length, dims[2].length), [dims[0], dims[2]]))
    factors.append(Dense(torch.randn(dims[1].length, dims[-1].length), [dims[1], dims[-1]]))
    return TN(factors=factors)


def _high_dimensional_mct(seed: int) -> MultiCOOTensor:
    input_dims = [Dim(2, float(i), f"k{i}") for i in range(6)]
    output_pairs = [
        [Dim(2, 10.0 + 2 * i, f"o{i}a"), Dim(2, 11.0 + 2 * i, f"o{i}b")]
        for i in range(6)
    ]
    ops = [
        COOSparsify(input_dim=input_dim, torch_table=_diag_table(*pair))
        for input_dim, pair in zip(input_dims, output_pairs)
    ]
    print(MultiCOOSparsify(ops), [d for op in ops for d in op.output_dims], set([d for op in ops for d in op.output_dims]))
    a = [d for op in ops for d in op.output_dims]
    assert len(set(a)) == len(a)
    return MultiCOOTensor(
        tn=_connected_tn(input_dims, seed=seed),
        sparsify=MultiCOOSparsify(ops),
    )


def _same_sparsify_mct(template: MultiCOOTensor, seed: int) -> MultiCOOTensor:
    input_dims = [op.input_dim for op in template.sparsify.ops]
    return MultiCOOTensor(
        tn=_connected_tn(input_dims, seed=seed),
        sparsify=template.sparsify,
    )


def _pair_table(row: Dim, col: Dim, pairs: list[tuple[int, int]]) -> TorchTable:
    return _mark_unique_sorted(
        TorchTable(
            columns=[row, col],
            data=[
                torch.tensor([pair[0] for pair in pairs]),
                torch.tensor([pair[1] for pair in pairs]),
            ],
            length=len(pairs),
        )
    )


def _pair_mct(
    row: Dim,
    col: Dim,
    pairs: list[tuple[int, int]],
    values: torch.Tensor,
    term_idx: int,
) -> MultiCOOTensor:
    input_dim = Dim(len(pairs), -10.0 + term_idx, f"mctsum_k{term_idx}")
    op = COOSparsify(input_dim=input_dim, torch_table=_pair_table(row, col, pairs))
    return MultiCOOTensor(
        tn=TN.from_dense(Dense(values, [input_dim])),
        sparsify=MultiCOOSparsify([op]),
    )


def test_todo_forward_backward_tn_with_five_connected_factors_roundtrip():
    k = Dim(3, 0.0, "k")
    a = Dim(3, 1.0, "a")
    b = Dim(3, 2.0, "b")
    spectators = [Dim(2, 3.0 + i, f"s{i}") for i in range(5)]
    op = COOSparsify(input_dim=k, torch_table=_diag_table(a, b))

    tn = _connected_tn([k] + spectators, seed=1)

    roundtrip = op.backward(op.forward(tn))

    _dense_allclose(roundtrip.to_dense(), tn.to_dense())


def test_todo_inverse_supersets_expands_connected_tn_consistently():
    a = Dim(3, 1.0, "a")
    b = Dim(3, 2.0, "b")
    c = Dim(3, 3.0, "c")
    d = Dim(3, 4.0, "d")
    k_ab = Dim(3, 0.0, "k_ab")
    k_cd = Dim(3, 0.5, "k_cd")

    op_ab = COOSparsify(input_dim=k_ab, torch_table=_diag_table(a, b))
    op_cd = COOSparsify(input_dim=k_cd, torch_table=_diag_table(c, d))
    merged = COOSparsify.merge(op_ab, op_cd)

    spectators = [Dim(2, 5.0 + i, f"s{i}") for i in range(5)]
    tn = _connected_tn([merged.input_dim] + spectators, seed=2)

    fine = MultiCOOSparsify([merged])
    supersets = MultiCOOSparsify([op_ab, op_cd])
    inverse = fine.inverse_supersets(supersets)

    via_inverse = supersets.forward(inverse.forward(tn)).to_dense()
    direct = fine.forward(tn).to_dense()

    _dense_allclose(via_inverse, direct)


def test_todo_multicootensor_mul_debug_high_dimensional(monkeypatch):
    monkeypatch.setattr(coo, "DEBUG_MultiCOOTensor", True)
    lhs = _high_dimensional_mct(seed=3)
    rhs = _same_sparsify_mct(lhs, seed=4)

    result = lhs * rhs

    _dense_allclose(result.to_dense(), lhs.to_dense() * rhs.to_dense())


def test_todo_multicootensor_sum_debug_high_dimensional(monkeypatch):
    monkeypatch.setattr(coo, "DEBUG_MultiCOOTensor", True)
    tensor = _high_dimensional_mct(seed=5)
    dims = [
        tensor.sparsify.ops[0].output_dims[0],
        tensor.sparsify.ops[1].output_dims[1],
        *tensor.sparsify.ops[5].output_dims,
    ]

    result = tensor.sum(dims)

    _dense_allclose(result.to_dense(), tensor.to_dense().sum(dims))


def test_dense_forward_backward_tuple_indexing_with_batch_dims():
    batch = Dim(4, -1.0, "batch")
    k = Dim(3, 0.0, "k")
    row = Dim(2, 1.0, "row")
    col = Dim(3, 2.0, "col")
    table = _mark_unique_sorted(
        TorchTable(
            columns=[row, col],
            data=[torch.tensor([0, 1, 1]), torch.tensor([2, 0, 2])],
            length=3,
        )
    )
    op = COOSparsify(input_dim=k, torch_table=table)
    x = Dense(torch.arange(12.0).reshape(4, 3), [batch, k])

    y = op.forward(x)
    expected = torch.zeros(4, 2, 3)
    expected[:, [0, 1, 1], [2, 0, 2]] = x.tensor

    assert y.dims == [batch, row, col]
    assert torch.equal(y.tensor, expected)
    _dense_allclose(op.backward(y), x)


def test_backward_with_partial_output_dims_uses_tuple_gather():
    batch = Dim(2, -1.0, "batch")
    k = Dim(3, 0.0, "k")
    row = Dim(2, 1.0, "row")
    col = Dim(3, 2.0, "col")
    table = _mark_unique_sorted(
        TorchTable(
            columns=[row, col],
            data=[torch.tensor([0, 1, 1]), torch.tensor([2, 0, 2])],
            length=3,
        )
    )
    op = COOSparsify(input_dim=k, torch_table=table)
    y = Dense(torch.arange(4.0).reshape(2, 2), [batch, row])

    gathered = op.backward(y)
    expected = y.tensor[:, torch.tensor([0, 1, 1])]

    assert gathered.dims == [batch, k]
    assert torch.equal(gathered.tensor, expected)


def test_multicootensor_sum_partial_reduction_coalesces_duplicate_rows():
    k = Dim(4, 0.0, "k")
    row = Dim(2, 1.0, "row")
    col = Dim(2, 2.0, "col")
    table = _mark_unique_sorted(
        TorchTable(
            columns=[row, col],
            data=[torch.tensor([0, 0, 1, 1]), torch.tensor([0, 1, 0, 1])],
            length=4,
        )
    )
    op = COOSparsify(input_dim=k, torch_table=table)
    tensor = MultiCOOTensor(
        tn=TN.from_dense(Dense(torch.tensor([1.0, 2.0, 4.0, 8.0]), [k])),
        sparsify=MultiCOOSparsify([op]),
    )

    result = tensor.sum([col])
    expected = tensor.to_dense().sum([col])

    _dense_allclose(result.to_dense(), expected)


def test_multicootensor_sum_add_coalesces_compatible_terms():
    row = Dim(3, 1.0, "mctsum_add_row")
    col = Dim(3, 2.0, "mctsum_add_col")
    pairs = [(0, 0), (1, 1), (2, 2)]
    lhs = _pair_mct(row, col, pairs, torch.tensor([1.0, -2.0, 3.0]), term_idx=0)
    rhs = MultiCOOTensor(
        tn=TN.from_dense(
            Dense(
                torch.tensor([4.0, 5.0, -6.0]),
                [lhs.sparsify.ops[0].input_dim],
            )
        ),
        sparsify=lhs.sparsify,
    )

    result = MultiCOOTensorSum([lhs]) + MultiCOOTensorSum([rhs])

    assert len(result.terms) == 1
    _dense_allclose(result.to_dense(), lhs.to_dense() + rhs.to_dense())


def test_multicootensor_sum_apply_multiplicative_handles_six_overlapping_terms():
    row = Dim(4, 1.0, "mctsum_square_row")
    col = Dim(4, 2.0, "mctsum_square_col")
    pairs_by_term = [
        [(0, 0), (0, 1), (1, 1), (2, 3), (3, 0)],
        [(0, 0), (1, 1), (1, 2), (2, 3), (3, 1)],
        [(0, 1), (1, 1), (2, 0), (2, 3), (3, 2)],
        [(0, 2), (1, 1), (2, 0), (2, 3), (3, 3)],
        [(0, 0), (0, 3), (1, 1), (2, 1), (2, 3)],
        [(0, 1), (1, 0), (1, 1), (2, 3), (3, 0)],
    ]
    terms = [
        _pair_mct(
            row,
            col,
            pairs,
            torch.tensor([1.0, -2.0, 0.5, 3.0, -1.5]) + term_idx,
            term_idx=term_idx,
        )
        for term_idx, pairs in enumerate(pairs_by_term)
    ]
    tensor_sum = MultiCOOTensorSum(terms)
    dense_sum = tensor_sum.to_dense()

    result = tensor_sum.apply_multiplicative(lambda tensor: tensor.square())
    expected = Dense(dense_sum.tensor.square(), dense_sum.dims)

    assert len(tensor_sum.terms) == 6
    _dense_allclose(result.to_dense(), expected)


def test_multicootensor_mul_debug_connected_merge(monkeypatch):
    monkeypatch.setattr(coo, "DEBUG_MultiCOOTensor", True)
    left_k = Dim(3, 0.0, "left_k")
    right_k = Dim(2, 0.5, "right_k")
    row = Dim(2, 1.0, "row")
    col = Dim(2, 2.0, "col")
    left_table = _mark_unique_sorted(
        TorchTable(
            columns=[row, col],
            data=[torch.tensor([0, 0, 1]), torch.tensor([0, 1, 1])],
            length=3,
        )
    )
    right_table = _diag_table(row)
    left = MultiCOOTensor(
        tn=TN.from_dense(Dense(torch.tensor([2.0, 3.0, 5.0]), [left_k])),
        sparsify=MultiCOOSparsify([COOSparsify(input_dim=left_k, torch_table=left_table)]),
    )
    right = MultiCOOTensor(
        tn=TN.from_dense(Dense(torch.tensor([7.0, 11.0]), [right_k])),
        sparsify=MultiCOOSparsify([COOSparsify(input_dim=right_k, torch_table=right_table)]),
    )

    result = left * right

    _dense_allclose(result.to_dense(), left.to_dense() * right.to_dense())


@pytest.mark.parametrize("seed", range(8))
def test_random_coosparsify_graph_merge_components(seed: int):
    generator = torch.Generator().manual_seed(1200 + seed)
    left_count = 3
    right_count = 2
    edge_count = 4 + int(torch.randint(0, 3, (), generator=generator).item())

    left_output_dims = [
        [
            Dim(2, 10.0 * i, f"graph_l{seed}_{i}_private_a"),
            Dim(2, 10.0 * i + 1.0, f"graph_l{seed}_{i}_private_b"),
        ]
        for i in range(left_count)
    ]
    right_output_dims = [
        [
            Dim(3, 100.0 + 10.0 * i, f"graph_r{seed}_{i}_private_a"),
            Dim(2, 100.0 + 10.0 * i + 1.0, f"graph_r{seed}_{i}_private_b"),
        ]
        for i in range(right_count)
    ]

    edges = {(0, 0), (1, 0), (2, 1), (0, 1)}
    while len(edges) < edge_count:
        left_idx = int(torch.randint(0, left_count, (), generator=generator).item())
        right_idx = int(torch.randint(0, right_count, (), generator=generator).item())
        edges.add((left_idx, right_idx))

    for edge_idx, (left_idx, right_idx) in enumerate(sorted(edges)):
        shared = Dim(
            2,
            1000.0 + edge_idx,
            f"graph_shared{seed}_{left_idx}_{right_idx}",
        )
        left_output_dims[left_idx].append(shared)
        right_output_dims[right_idx].append(shared)

    left_ops = []
    for i, output_dims in enumerate(left_output_dims):
        input_dim = Dim(4, -100.0 - i, f"graph_l{seed}_{i}_k")
        left_ops.append(
            COOSparsify(
                input_dim=input_dim,
                torch_table=_random_unique_table(
                    output_dims,
                    input_dim.length,
                    1300 + 20 * seed + i,
                ),
            )
        )

    right_ops = []
    for i, output_dims in enumerate(right_output_dims):
        input_dim = Dim(4, -200.0 - i, f"graph_r{seed}_{i}_k")
        right_ops.append(
            COOSparsify(
                input_dim=input_dim,
                torch_table=_random_unique_table(
                    output_dims,
                    input_dim.length,
                    1400 + 20 * seed + i,
                ),
            )
        )

    expected_components = []
    for op in left_ops + right_ops:
        op_dims = set(op.output_dims)
        matches = [
            component
            for component in expected_components
            if op_dims & set().union(*(set(member.output_dims) for member in component))
        ]
        if not matches:
            expected_components.append([op])
            continue

        merged = [op]
        for component in matches:
            merged.extend(component)
            expected_components.remove(component)
        expected_components.append(merged)

    merged_sparsify = MultiCOOSparsify(left_ops).merge(MultiCOOSparsify(right_ops))

    actual_dim_sets = [frozenset(op.output_dims) for op in merged_sparsify.ops]
    expected_dim_sets = [
        frozenset(dim for op in component for dim in op.output_dims)
        for component in expected_components
    ]

    assert set(actual_dim_sets) == set(expected_dim_sets)
    assert len(merged_sparsify.ops) == len(expected_components)

    left_input_dims = [op.input_dim for op in left_ops]
    right_input_dims = [op.input_dim for op in right_ops]
    lhs = MultiCOOTensor(
        tn=TN.from_dense(
            Dense(
                torch.randn([dim.length for dim in left_input_dims], generator=generator),
                left_input_dims,
            )
        ),
        sparsify=MultiCOOSparsify(left_ops),
    )
    rhs = MultiCOOTensor(
        tn=TN.from_dense(
            Dense(
                torch.randn([dim.length for dim in right_input_dims], generator=generator),
                right_input_dims,
            )
        ),
        sparsify=MultiCOOSparsify(right_ops),
    )

    product = lhs * rhs
    dense_product = lhs.to_dense() * rhs.to_dense()
    _dense_allclose(product.to_dense(), dense_product)

    sum_dims = [
        left_output_dims[0][0],
        right_output_dims[0][0],
        *left_output_dims[-1][-1:],
    ]
    _dense_allclose(product.sum(sum_dims).to_dense(), dense_product.sum(sum_dims))


@pytest.mark.parametrize("seed", range(8))
def test_random_dense_forward_backward_tuple_indexing(seed: int):
    torch.manual_seed(100 + seed)
    batch = Dim(2 + seed % 3, -1.0, f"batch{seed}")
    k = Dim(4, 0.0, f"k{seed}")
    out0 = Dim(2 + seed % 2, 1.0, f"out{seed}_0")
    out1 = Dim(3, 2.0, f"out{seed}_1")
    table = _random_unique_table([out0, out1], k.length, seed=200 + seed)
    op = COOSparsify(input_dim=k, torch_table=table)
    x = Dense(torch.randn(batch.length, k.length), [batch, k])

    y = op.forward(x)
    materialized = table.materialize()
    expected = torch.zeros(batch.length, out0.length, out1.length)
    expected[(..., materialized[:, 0], materialized[:, 1])] = x.tensor

    assert y.dims == [batch, out0, out1]
    assert torch.equal(y.tensor, expected)
    _dense_allclose(op.backward(y), x)


@pytest.mark.parametrize("seed", range(8))
def test_random_backward_partial_output_dims(seed: int):
    torch.manual_seed(300 + seed)
    batch = Dim(2 + seed % 2, -1.0, f"partial_batch{seed}")
    k = Dim(5, 0.0, f"partial_k{seed}")
    out0 = Dim(3, 1.0, f"partial_out{seed}_0")
    out1 = Dim(4, 2.0, f"partial_out{seed}_1")
    table = _random_unique_table([out0, out1], k.length, seed=400 + seed)
    op = COOSparsify(input_dim=k, torch_table=table)
    y = Dense(torch.randn(batch.length, out1.length), [batch, out1])

    gathered = op.backward(y)
    materialized = table.materialize()
    expected = y.tensor[(..., materialized[:, 1])]

    assert gathered.dims == [batch, k]
    assert torch.equal(gathered.tensor, expected)


@pytest.mark.parametrize("seed", range(6))
def test_random_multicootensor_partial_sums_match_dense(seed: int):
    torch.manual_seed(500 + seed)
    input_dims = [Dim(4, float(i), f"sum_k{seed}_{i}") for i in range(3)]
    output_pairs = [
        [Dim(2, 10.0 + 2 * i, f"sum_o{seed}_{i}_a"), Dim(3, 11.0 + 2 * i, f"sum_o{seed}_{i}_b")]
        for i in range(3)
    ]
    ops = [
        COOSparsify(
            input_dim=input_dim,    
            torch_table=_random_unique_table(pair, input_dim.length, seed=600 + 10 * seed + i),
        )
        for i, (input_dim, pair) in enumerate(zip(input_dims, output_pairs))
    ]
    tensor = MultiCOOTensor(
        tn=_connected_tn(input_dims, seed=700 + seed),
        sparsify=MultiCOOSparsify(ops),
    )
    dims = [output_pairs[0][seed % 2], output_pairs[1][1], *output_pairs[2]]

    result = tensor.sum(dims)

    _dense_allclose(result.to_dense(), tensor.to_dense().sum(dims))


@pytest.mark.parametrize("seed", range(5))
def test_random_multicootensor_same_sparsify_mul_matches_dense(seed: int, monkeypatch):
    monkeypatch.setattr(coo, "DEBUG_MultiCOOTensor", True)
    torch.manual_seed(800 + seed)
    input_dims = [Dim(3, float(i), f"mul_k{seed}_{i}") for i in range(4)]
    output_pairs = [
        [Dim(2, 10.0 + 2 * i, f"mul_o{seed}_{i}_a"), Dim(3, 11.0 + 2 * i, f"mul_o{seed}_{i}_b")]
        for i in range(4)
    ]
    ops = [
        COOSparsify(
            input_dim=input_dim,
            torch_table=_random_unique_table(pair, input_dim.length, seed=900 + 10 * seed + i),
        )
        for i, (input_dim, pair) in enumerate(zip(input_dims, output_pairs))
    ]
    sparsify = MultiCOOSparsify(ops)
    lhs = MultiCOOTensor(tn=_connected_tn(input_dims, seed=1000 + seed), sparsify=sparsify)
    rhs = MultiCOOTensor(tn=_connected_tn(input_dims, seed=1100 + seed), sparsify=sparsify)

    product = lhs * rhs

    _dense_allclose(product.to_dense(), lhs.to_dense() * rhs.to_dense())


@pytest.mark.parametrize("seed", range(2))
def test_random_graphs_connect(seed: int, monkeypatch):
    monkeypatch.setattr(coo, "DEBUG_MultiCOOTensor", True)
    if seed % 2 == 0:
        monkeypatch.setattr(coo, "DEBUG_NO_MD_EYE_OPT", True)

    torch.manual_seed(1200 + seed)

    l_input_dims = [Dim(3, float(i), f"lhs_{i}") for i in range(5)]
    r_input_dims = [Dim(3, float(i + 5), f"rhs_{i}") for i in range(5)]
    output_dims = [Dim(3, float(i), f"out_{i}") for i in range(9)]
    graph1 = [
        [0, 1, 2],
        [3, 5],
        [4, 8],
        [6],
        [7],
    ]

    graph2 = [
        [0, 3],
        [1, 4],
        [6, 8],
        [5, 7],
        [2],
    ]

    left_ops = [
        COOSparsify.md_eye(
            input_dim=input_dim,
            output_dims=[output_dims[i] for i in graph1[idx]],
        )
        for idx, input_dim in enumerate(l_input_dims)
    ]

    right_ops = [
        COOSparsify.md_eye(
            input_dim=input_dim,
            output_dims=[output_dims[i] for i in graph2[idx]],
        )
        for idx, input_dim in enumerate(r_input_dims)
    ]

    lhs = MultiCOOTensor(
        tn=TN.from_dense(
            Dense(
                torch.ones([dim.length for dim in l_input_dims]),
                l_input_dims,
            )
        ),
        sparsify=MultiCOOSparsify(left_ops),
    )
    rhs = MultiCOOTensor(
        tn=TN.from_dense(
            Dense(
                torch.ones([dim.length for dim in r_input_dims]),
                r_input_dims,
            )
        ),
        sparsify=MultiCOOSparsify(right_ops),
    )

    result = lhs * rhs
    _dense_allclose(result.to_dense(), lhs.to_dense() * rhs.to_dense())
    result = result.sum([output_dims[0], output_dims[3], output_dims[6]])
    _dense_allclose(result.to_dense(), (lhs.to_dense() * rhs.to_dense()).sum([output_dims[0], output_dims[3], output_dims[6]]))

    assert result.to_dense().tensor.sum().allclose(torch.tensor(3.0))


@pytest.mark.parametrize("seed", range(5))
def test_random_graphs_value(seed: int, monkeypatch):
    monkeypatch.setattr(coo, "DEBUG_MultiCOOTensor", True)
    torch.manual_seed(1200 + seed)

    l_input_dims = [Dim(10, float(i), f"lhs_{i}") for i in range(2)]
    r_input_dims = [Dim(10, float(i + 5), f"rhs_{i}") for i in range(2)]
    output_dims = [Dim(4, float(i), f"out_{i}") for i in range(4)]
    graph1 = [
        [0, 1],
        [2, 3],
    ]

    graph2 = [
        [0, 2],
        [1, 3],
    ]

    left_ops = [
        COOSparsify(
            input_dim=input_dim,
            torch_table=_random_unique_table(
                [output_dims[i] for i in graph1[idx]],
                input_dim.length,
                900 + 10 * seed + idx,
            ),
        )
        for idx, input_dim in enumerate(l_input_dims)
    ]

    right_ops = [
        COOSparsify(
            input_dim=input_dim,
            torch_table=_random_unique_table(
                [output_dims[i] for i in graph2[idx]],
                input_dim.length,
                1000 + 10 * seed + idx,
            ),
        )
        for idx, input_dim in enumerate(r_input_dims)
    ]

    lhs = MultiCOOTensor(
        tn=TN.from_dense(
            Dense(
                torch.ones([dim.length for dim in l_input_dims]),
                l_input_dims,
            )
        ),
        sparsify=MultiCOOSparsify(left_ops),
    )
    rhs = MultiCOOTensor(
        tn=TN.from_dense(
            Dense(
                torch.ones([dim.length for dim in r_input_dims]),
                r_input_dims,
            )
        ),
        sparsify=MultiCOOSparsify(right_ops),
    )

    result = lhs * rhs
    _dense_allclose(result.to_dense(), lhs.to_dense() * rhs.to_dense())
    result = result.sum([output_dims[0], output_dims[3]])
    _dense_allclose(result.to_dense(), (lhs.to_dense() * rhs.to_dense()).sum([output_dims[0], output_dims[3]]))
    assert len(result.sparsify.ops) == 1
    assert result.to_dense().tensor.sum() > 0 
    assert result.to_dense().tensor.sum() < 100
