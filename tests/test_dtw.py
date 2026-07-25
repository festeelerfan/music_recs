import torch

from src.index.dtw import dtw_distance


def test_identical_sequence_gives_diagonal_alignment_cost():
    # unit vectors: self dot product is 1 per step, so the diagonal
    # alignment (each step matched with itself) is optimal - cost -1/step,
    # normalized by (n+m)=6 -> -3/6 = -0.5 exactly.
    seq = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
    assert dtw_distance(seq, seq) == -0.5


def test_similar_shape_scores_better_than_dissimilar_shape():
    # anchor: rises then falls
    anchor = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
    # same shape, just stretched (extra repeated middle step) - DTW should
    # align this well despite the different length
    similar = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.0, 1.0], [1.0, 0.0]])
    # same segments, totally different order - shouldn't align as well
    different_order = torch.tensor([[0.0, 1.0], [1.0, 0.0], [1.0, 0.0]])

    d_similar = dtw_distance(anchor, similar)
    d_different = dtw_distance(anchor, different_order)
    assert d_similar < d_different


def test_handles_different_length_sequences():
    a = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    b = torch.tensor([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0], [0.0, 1.0]])
    d = dtw_distance(a, b)
    assert d == d  # not NaN