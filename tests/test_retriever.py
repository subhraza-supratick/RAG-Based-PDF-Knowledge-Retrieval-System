import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.retriever import compute_cosine_similarity


class TestCosineSimilarity:
    def test_identical_vectors_score_one(self):
        v = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        score = compute_cosine_similarity(v, v)
        assert abs(score - 1.0) < 1e-5

    def test_orthogonal_vectors_score_zero(self):
        a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        b = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        score = compute_cosine_similarity(a, b)
        assert abs(score) < 1e-5

    def test_opposite_vectors_score_minus_one(self):
        a = np.array([1.0, 0.0], dtype=np.float32)
        b = np.array([-1.0, 0.0], dtype=np.float32)
        score = compute_cosine_similarity(a, b)
        assert abs(score - (-1.0)) < 1e-5

    def test_zero_vector_returns_zero(self):
        a = np.zeros(5, dtype=np.float32)
        b = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
        score = compute_cosine_similarity(a, b)
        assert score == 0.0

    def test_score_in_range_minus_one_to_one(self):
        rng = np.random.RandomState(42)
        for _ in range(20):
            a = rng.randn(768).astype(np.float32)
            b = rng.randn(768).astype(np.float32)
            score = compute_cosine_similarity(a, b)
            assert -1.0 <= score <= 1.0

    def test_semantic_higher_than_unrelated(self):
        from backend.embeddings import get_embedding
        v_revenue = get_embedding("total revenue and net income")
        v_related = get_embedding("annual revenue growth and profits")
        v_unrelated = get_embedding("weather forecast for tomorrow")
        score_related = compute_cosine_similarity(v_revenue, v_related)
        score_unrelated = compute_cosine_similarity(v_revenue, v_unrelated)
        assert score_related >= score_unrelated
