import torch
from sklearn.cluster import KMeans
from typing import List, Dict, Tuple


class ClusterNSMoCo:
    def __init__(self, num_clusters=100, embedding_dim=128, device="cuda"):
        self.num_clusters = num_clusters
        self.embedding_dim = embedding_dim
        self.device = device
        self.centroids = None
        self.memory_bank_cluster_ids = None
        self.initialized = False

    def update_centroids(self, memory_bank: torch.Tensor, step: int, update_step: int):
        """
        Cluster the memory bank using cosine similarity (via normalized vectors).
        This will initialize once, and re-cluster every 'update_step' steps after that.
        """
        if memory_bank.shape[0] < self.num_clusters:
            return  # not enough samples to cluster

        if not self.initialized or (step % update_step == 0):
            kmeans = KMeans(n_clusters=self.num_clusters, random_state=42)
            kmeans.fit(memory_bank.cpu().numpy())

            self.centroids = torch.tensor(kmeans.cluster_centers_, device=self.device)
            self.centroids = torch.nn.functional.normalize(self.centroids, dim=1)
            self.memory_bank_cluster_ids = torch.tensor(kmeans.labels_, device=self.device)
            self.memory_bank_embeddings = memory_bank.clone().detach()  
            self.initialized = True

    def assign_to_centroids(self, embeddings: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Assign embeddings to the closest centroid and return sorted centroid indices.
        """
        sims = torch.matmul(embeddings, self.centroids.T)
        sorted_indices = torch.argsort(sims, dim=1, descending=True)
        cluster_ids = sorted_indices[:, 0]
        return cluster_ids, sorted_indices

    def get_negatives_by_cluster(
        self, queries: torch.Tensor, memory_bank: torch.Tensor
    ) -> Dict[str, List[torch.Tensor]]:
        """
        For each query:
        - return false negatives (same cluster) [both indices and embeddings]
        - return hard negatives (second closest cluster) [both indices and embeddings]
        - return assigned centroid
        - return all centroids and memory bank cluster assignments
        """
        sims = torch.matmul(queries, self.centroids.T)
        sorted_indices = torch.argsort(sims, dim=1, descending=True)

        results = {
            "false_negative_indices": [],
            "false_negative_embeddings": [],
            "hard_negative_indices": [],
            "hard_negative_embeddings": [],
            "query_centroids": [],
            "centroids": self.centroids,
            "memory_bank_cluster_ids": self.memory_bank_cluster_ids,
        }

        for i in range(queries.shape[0]):
            top1 = sorted_indices[i, 0].item()
            top2 = sorted_indices[i, 1].item()
            top3 = sorted_indices[i, 2].item()

            fn_idx = (self.memory_bank_cluster_ids == top1).nonzero(as_tuple=True)[0]
            hn2_idx = (self.memory_bank_cluster_ids == top2).nonzero(as_tuple=True)[0]
            hn3_idx = (self.memory_bank_cluster_ids == top3).nonzero(as_tuple=True)[0]
            hn_idx = torch.cat([hn2_idx, hn3_idx], dim=0)

            results["false_negative_indices"].append(fn_idx)
            results["false_negative_embeddings"].append(memory_bank[fn_idx])

            results["hard_negative_indices"].append(hn_idx)
            results["hard_negative_embeddings"].append(memory_bank[hn_idx])

            results["query_centroids"].append(self.centroids[top1])

        return results

    def get_memory_bank_clusters(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            memory_bank_embeddings: (N, D)
            memory_bank_cluster_ids: (N,)
            centroids: (num_clusters, D)
        """
        return self.memory_bank_embeddings, self.memory_bank_cluster_ids, self.centroids 