"""Dynamic Time Warping over sequences of embeddings - compares two tracks'
full segment-embedding sequences (their "shape" over time) instead of
collapsing each to a single mean vector, which discards how a song's
character evolves over its duration.
"""

import numpy as np


def dtw_distance(seq_a, seq_b):
    """Lower is more similar (this is a cost/distance, not a similarity
    score). Per-step cost is negative dot product - the same similarity
    convention MuQMuLan.calc_similarity uses elsewhere, no separate cosine
    normalization. Normalized by combined sequence length so tracks of
    different durations remain comparable."""
    cost_matrix = -(seq_a @ seq_b.T)
    cost_matrix = cost_matrix.detach().cpu().numpy()
    n, m = cost_matrix.shape

    dp = np.full((n + 1, m + 1), np.inf)
    dp[0, 0] = 0.0
    for i in range(1, n + 1):
        row = cost_matrix[i - 1]
        dp_prev = dp[i - 1]
        dp_cur = dp[i]
        for j in range(1, m + 1):
            dp_cur[j] = row[j - 1] + min(dp_prev[j], dp_cur[j - 1], dp_prev[j - 1])

    return dp[n, m] / (n + m)