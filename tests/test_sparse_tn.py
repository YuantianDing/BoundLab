import itertools

import torch

from boundlab.sparse.tn import Dense, Dim, TN


def test_dense_diagonal_one_dim_renames_dim():
    source = Dim(3, 0.0, "source")
    target = Dim(3, 1.0, "target")
    dense = Dense(torch.arange(3.0), [source])

    result = dense.diagonal([source], target)

    assert result.dims == [target]
    assert torch.equal(result.tensor, dense.tensor)


def test_dense_diagonal_embed_one_dim_renames_dim():
    source = Dim(3, 0.0, "source")
    target = Dim(3, 1.0, "target")
    dense = Dense(torch.arange(3.0), [source])

    result = dense.diagonal_embed(source, [target])

    assert result.dims == [target]
    assert torch.equal(result.tensor, dense.tensor)


def test_dense_diagonal_two_dims_matches_torch_diagonal():
    row = Dim(3, 0.0, "row")
    col = Dim(3, 1.0, "col")
    diag = Dim(3, 2.0, "diag")
    tensor = torch.arange(9.0).reshape(3, 3)
    dense = Dense(tensor, [row, col])

    result = dense.diagonal([row, col], diag)

    assert result.dims == [diag]
    assert torch.equal(result.tensor, tensor.diagonal())


def test_dense_diagonal_two_dims_matches_torch_diagonal_nontrailing_dims():
    row = Dim(3, 0.0, "row")
    batch = Dim(2, 1.0, "batch")
    col = Dim(3, 2.0, "col")
    diag = Dim(3, 3.0, "diag")
    tensor = torch.arange(18.0).reshape(3, 2, 3)
    dense = Dense(tensor, [row, batch, col])

    result = dense.diagonal([row, col], diag)

    assert result.dims == [batch, diag]
    assert torch.equal(result.tensor, tensor.diagonal(dim1=0, dim2=2))


def test_dense_diagonal_embed_two_dims_matches_torch_diag_embed():
    batch = Dim(2, 0.0, "batch")
    diag = Dim(3, 1.0, "diag")
    row = Dim(3, 2.0, "row")
    col = Dim(3, 3.0, "col")
    tensor = torch.arange(6.0).reshape(2, 3)
    dense = Dense(tensor, [batch, diag])

    result = dense.diagonal_embed(diag, [row, col])

    assert result.dims == [batch, row, col]
    assert torch.equal(result.tensor, torch.diag_embed(tensor))


def test_dense_diagonal_embed_two_dims_matches_torch_diag_embed_nontrailing_dim():
    source = Dim(3, 0.0, "source")
    batch = Dim(2, 1.0, "batch")
    row = Dim(3, 2.0, "row")
    col = Dim(3, 3.0, "col")
    tensor = torch.arange(6.0).reshape(3, 2)
    dense = Dense(tensor, [source, batch])

    result = dense.diagonal_embed(source, [row, col])

    assert result.dims == [batch, row, col]
    assert torch.equal(result.tensor, torch.diag_embed(tensor.permute(1, 0)))


def test_dense_diagonal_three_dims():
    i = Dim(3, 0.0, "i")
    j = Dim(3, 1.0, "j")
    k = Dim(3, 2.0, "k")
    diag = Dim(3, 3.0, "diag")
    tensor = torch.arange(27.0).reshape(3, 3, 3)
    dense = Dense(tensor, [i, j, k])

    result = dense.diagonal([i, j, k], diag)

    assert result.dims == [diag]
    assert torch.equal(result.tensor, torch.tensor([0.0, 13.0, 26.0]))


def test_dense_diagonal_embed_creates_multidimensional_eye():
    diag = Dim(4, 0.0, "diag")
    i = Dim(4, 1.0, "i")
    j = Dim(4, 2.0, "j")
    k = Dim(4, 3.0, "k")
    dense = Dense(torch.ones(4), [diag])

    eye = dense.diagonal_embed(diag, [i, j, k])

    assert eye.dims == [i, j, k]
    assert eye.tensor.sum().item() == 4.0
    for a, b, c in itertools.product(range(4), repeat=3):
        expected = 1.0 if a == b == c else 0.0
        assert eye.tensor[a, b, c].item() == expected


def test_dense_diagonal_embed_and_diagonal_with_spectator_dim_roundtrip():
    batch = Dim(2, 0.0, "batch")
    diag = Dim(3, 1.0, "diag")
    row = Dim(3, 2.0, "row")
    col = Dim(3, 3.0, "col")
    recovered = Dim(3, 4.0, "recovered")
    tensor = torch.arange(6.0).reshape(2, 3)
    dense = Dense(tensor, [batch, diag])

    scattered = dense.diagonal_embed(diag, [row, col])
    viewed = scattered.diagonal([row, col], recovered)

    assert scattered.dims == [batch, row, col]
    for b, r, c in itertools.product(range(2), range(3), range(3)):
        expected = tensor[b, r].item() if r == c else 0.0
        assert scattered.tensor[b, r, c].item() == expected
    assert viewed.dims == [batch, recovered]
    assert torch.equal(viewed.tensor, tensor)


def test_dense_norm_reduces_requested_dims_and_honors_p():
    row = Dim(2, 0.0, "row")
    col = Dim(3, 1.0, "col")
    tensor = torch.tensor([[1.0, -2.0, 2.0], [-3.0, 4.0, -12.0]])
    dense = Dense(tensor, [row, col])

    l1 = dense.norm([col], p=1)
    l2 = dense.norm([col], p=2)

    assert l1.dims == [row]
    assert l2.dims == [row]
    assert torch.equal(l1.tensor, tensor.abs().sum(dim=1))
    assert torch.allclose(l2.tensor, torch.linalg.vector_norm(tensor, ord=2, dim=1))


def test_dense_norm_scales_dims_not_present_in_tensor():
    row = Dim(2, 0.0, "row")
    missing = Dim(4, 1.0, "missing")
    dense = Dense(torch.tensor([3.0, -5.0]), [row])

    l1 = dense.norm([missing], p=1)
    l2 = dense.norm([missing], p=2)

    assert l1.dims == [row]
    assert l2.dims == [row]
    assert torch.equal(l1.tensor, dense.tensor.abs() * missing.length)
    assert torch.equal(l2.tensor, dense.tensor.abs() * missing.length ** 0.5)


def test_tn_norm_matches_dense_norm_for_connected_factors():
    row = Dim(2, 0.0, "row")
    shared = Dim(3, 1.0, "shared")
    col = Dim(2, 2.0, "col")
    left = Dense(torch.tensor([[1.0, -2.0, 3.0], [-4.0, 5.0, -6.0]]), [row, shared])
    right = Dense(torch.tensor([[2.0, -1.0], [-3.0, 4.0], [5.0, -2.0]]), [shared, col])
    scale = Dense(torch.tensor([0.5, -2.0]), [col])
    tn = TN([left, right, scale])
    dense = tn.to_dense()

    l1 = tn.norm([shared], p=1).to_dense()
    l2 = tn.norm([shared], p=2).to_dense()

    assert l1.allclose(dense.norm([shared], p=1))
    assert l2.allclose(dense.norm([shared], p=2))


def test_tn_norm_scales_dims_not_present_in_tensor():
    row = Dim(2, 0.0, "row")
    missing = Dim(4, 1.0, "missing")
    tn = TN.from_dense(Dense(torch.tensor([3.0, -5.0]), [row]))

    l1 = tn.norm([missing], p=1).to_dense()
    l2 = tn.norm([missing], p=2).to_dense()

    assert l1.allclose(tn.to_dense().norm([missing], p=1))
    assert l2.allclose(tn.to_dense().norm([missing], p=2))


def test_dense_diagonal_rejects_invalid_dims():
    dim = Dim(3, 0.0, "dim")
    missing = Dim(3, 1.0, "missing")
    target = Dim(3, 2.0, "target")
    wrong_length = Dim(4, 3.0, "wrong_length")
    dense = Dense(torch.arange(3.0), [dim])

    try:
        dense.diagonal([missing], target)
    except AssertionError:
        pass
    else:
        raise AssertionError("diagonal should reject missing dims")

    try:
        dense.diagonal([dim], wrong_length)
    except AssertionError:
        pass
    else:
        raise AssertionError("diagonal should reject length mismatches")


def test_dense_diagonal_embed_rejects_invalid_dims():
    dim = Dim(3, 0.0, "dim")
    existing = Dim(2, 1.0, "existing")
    target = Dim(3, 2.0, "target")
    wrong_length = Dim(4, 3.0, "wrong_length")
    dense = Dense(torch.arange(6.0).reshape(2, 3), [existing, dim])

    try:
        dense.diagonal_embed(target, [Dim(3, 4.0, "out")])
    except AssertionError:
        pass
    else:
        raise AssertionError("diagonal_embed should reject missing source dims")

    try:
        dense.diagonal_embed(dim, [existing])
    except AssertionError:
        pass
    else:
        raise AssertionError("diagonal_embed should reject existing target dims")

    try:
        dense.diagonal_embed(dim, [wrong_length])
    except AssertionError:
        pass
    else:
        raise AssertionError("diagonal_embed should reject length mismatches")
