import pickle
import numpy as np


class BPRRetriever:
    def __init__(self, artifact_path):
        with open(artifact_path, "rb") as f:
            artifact = pickle.load(f)

        self.model = artifact["model"]
        self.item_to_idx = artifact["item_to_idx"]
        self.idx_to_item = np.asarray(artifact["idx_to_item"])
        self.item_factors = self.model.item_factors

        factor_norms = np.linalg.norm(
            self.item_factors,
            axis=1,
            keepdims=True,
        )

        self.normalized_item_factors = (
            self.item_factors
            / np.clip(factor_norms, 1e-12, None)
        )

    def _session_vector(self, observed_aids):
        vectors = []
        weights = []

        for recency_rank, aid in enumerate(reversed(observed_aids)):
            idx = self.item_to_idx.get(aid)
            if idx is None:
                continue

            vectors.append(self.item_factors[idx])
            weights.append(1.0 / (recency_rank + 1))

        if not vectors:
            return None

        session_vector = np.average(
            np.vstack(vectors),
            axis=0,
            weights=np.asarray(weights),
        )

        norm = np.linalg.norm(session_vector)
        if norm < 1e-12:
            return None

        return session_vector / norm

    def recommend(self, observed_aids, k=50):
        session_vector = self._session_vector(observed_aids)

        if session_vector is None:
            return []

        scores = self.normalized_item_factors @ session_vector

        n_candidates = min(max(k, 1), len(scores))

        candidate_idx = np.argpartition(
            -scores,
            n_candidates - 1,
        )[:n_candidates]

        candidate_idx = candidate_idx[
            np.argsort(-scores[candidate_idx])
        ]

        return [
            int(self.idx_to_item[i])
            for i in candidate_idx[:k]
        ]
