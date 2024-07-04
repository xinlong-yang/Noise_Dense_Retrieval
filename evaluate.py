import torch
import pickle
import os
# p@K
def mean_average_precision_K(query,
                           database,
                           query_labels,
                           database_labels,
                           device,
                           topk
                           ):
    num_query = query_labels.shape[0]
    mean_AP = 0.0

    for i in range(num_query):
        retrieval = (query_labels[i, :] @ database_labels.t() > 0).float()
        hamming_dist = (1 - query[i, :] @ database.t())
        retrieval = retrieval[torch.argsort(hamming_dist)][:topk]
        retrieval_cnt = retrieval.sum().int().item()
        if retrieval_cnt == 0:
            continue
        score = torch.linspace(1, retrieval_cnt, retrieval_cnt).to(device)
        index = (torch.nonzero(retrieval == 1).squeeze() + 1.0).float()
        mean_AP += (score / index).mean()
    mean_AP = mean_AP / num_query 
    return mean_AP


def mutil_task_eval(query_code,
                           database_code,
                           query_labels,
                           database_labels,
                           device,
                           topk=None,
                           ):
    num_query = query_labels.shape[0]
    mean_AP = 0.0
    p_num = 0.0

    for i in range(num_query):
        retrieval = (query_labels[i, :] @ database_labels.t() > 0).float()
        hamming_dist = 0.5 * (1 - query_code[i, :] @ database_code.t())
        retrieval = retrieval[torch.argsort(hamming_dist)][:topk]
        retrieval_cnt = retrieval.sum().int().item()
        if retrieval_cnt == 0:
            continue

        score = torch.linspace(1, retrieval_cnt, retrieval_cnt).to(device)
        index = (torch.nonzero(retrieval == 1).squeeze() + 1.0).float().to(device)
        
        p_num += (retrieval[0] == 1).float()

        mean_AP += (score / index).mean()

    mean_AP = mean_AP / num_query
    p_num = p_num / num_query 

    return mean_AP , p_num




import numpy as np
from sklearn.metrics import normalized_mutual_info_score, adjusted_mutual_info_score
from loguru import logger
import logging
import faiss

def get_knn(
    reference_embeddings, test_embeddings, k, embeddings_come_from_same_source=False
):

    d = reference_embeddings.shape[1]
    logging.info("running k-nn with k=%d"%k)
    logging.info("embedding dimensionality is %d"%d)
    index = faiss.IndexFlatL2(d) 
    if faiss.get_num_gpus() > 0:
        index = faiss.index_cpu_to_all_gpus(index)
    index.add(reference_embeddings)
    _, indices = index.search(test_embeddings, k + 1) 
    if embeddings_come_from_same_source:
        return indices[:, 1:] 
    return indices[:, :k] 

def run_kmeans(x, nmb_clusters):
    n_data, d = x.shape
    logging.info("running k-means clustering with k=%d"%nmb_clusters)
    logging.info("embedding dimensionality is %d"%d)

    clus = faiss.Clustering(d, nmb_clusters)
    clus.niter = 20
    clus.max_points_per_centroid = 10000000
    index = faiss.IndexFlatL2(d)
    if faiss.get_num_gpus() > 0:
        index = faiss.index_cpu_to_all_gpus(index)
    clus.train(x, index)
    _, idxs = index.search(x, 1)
    return [int(n[0]) for n in idxs]

def get_relevance_mask(shape, gt_labels, embeddings_come_from_same_source=False, label_counts=None):
    if label_counts is None:
        label_counts = {k: v for k, v in zip(*np.unique(gt_labels, return_counts=True))}
    relevance_mask = np.zeros(shape=shape, dtype=np.int) 
    for k, v in label_counts.items():
        matching_rows = np.where(gt_labels == k)[0]
        max_column = v - 1 if embeddings_come_from_same_source else v
        relevance_mask[matching_rows, :max_column] = 1
    return relevance_mask 


def r_precision(knn_labels, gt_labels, embeddings_come_from_same_source=False, label_counts=None):
    relevance_mask = get_relevance_mask(knn_labels.shape, gt_labels, embeddings_come_from_same_source, label_counts)
    matches_per_row = np.sum((knn_labels == gt_labels) * relevance_mask.astype(bool), axis=1)
    max_possible_matches_per_row = np.sum(relevance_mask, axis=1)
    return np.mean(matches_per_row / max_possible_matches_per_row)


def mean_average_precision_at_r(knn_labels, gt_labels, embeddings_come_from_same_source=False, label_counts=None):
    relevance_mask = get_relevance_mask(knn_labels.shape, gt_labels, embeddings_come_from_same_source, label_counts)
    num_samples, num_k = knn_labels.shape
    equality = (knn_labels == gt_labels) * relevance_mask.astype(bool)
    cumulative_correct = np.cumsum(equality, axis=1)
    k_idx = np.tile(np.arange(1, num_k + 1), (num_samples, 1))
    precision_at_ks = (cumulative_correct * equality) / k_idx  
    summed_precision_per_row = np.sum(precision_at_ks * relevance_mask, axis=1) 
    max_possible_matches_per_row = np.sum(relevance_mask, axis=1)
    return np.mean(summed_precision_per_row / max_possible_matches_per_row) 

def mean_average_precision_at_100(knn_labels, gt_labels, embeddings_come_from_same_source=False, label_counts=None):
    k_precisions=[]
    for k in range(1,101):
        k_precisions.append(precision_at_k(knn_labels,gt_labels,k,reduction='none'))
    k_precisions=np.stack(k_precisions)
    ap_at_100=np.mean(k_precisions,axis=0)
    map_at_100=np.mean(ap_at_100)
    return map_at_100

def precision_at_k(knn_labels, gt_labels, k,reduction='mean'):
    curr_knn_labels = knn_labels[:, :k]
    if reduction=='mean': 
        precision = np.mean(np.sum(curr_knn_labels == gt_labels, axis=1) / k)
    else:
        precision = np.sum(curr_knn_labels == gt_labels, axis=1) / k
    return precision


def get_label_counts(reference_labels):
    unique_labels, label_counts = np.unique(reference_labels, return_counts=True)
    num_k = min(1023, int(np.max(label_counts)))  
    return {k: v for k, v in zip(unique_labels, label_counts)}, num_k


class AccuracyCalculator:
    def __init__(self, include=(), exclude=()):
        self.function_keyword = "calculate_"
        function_names = [x for x in dir(self) if x.startswith(self.function_keyword)]
        metrics = [x.replace(self.function_keyword, "", 1) for x in function_names]
        self.original_function_dict = {x: getattr(self, y) for x, y in zip(metrics, function_names)}
        self.original_function_dict = self.get_function_dict(include, exclude)
        self.curr_function_dict = self.get_function_dict()

    def get_function_dict(self, include=(), exclude=()):
        if len(include) == 0:
            include = list(self.original_function_dict.keys())
        included_metrics = [k for k in include if k not in exclude]
        return {k: v for k, v in self.original_function_dict.items() if k in included_metrics}

    def get_curr_metrics(self):
        return [k for k in self.curr_function_dict.keys()]

    def requires_clustering(self):
        return ["NMI", "AMI"]

    def requires_knn(self):
        return ["precision_at_1", "mean_average_precision_at_r", "r_precision"]

    def requires_100nn(self):
        return [ "mean_average_precision_at_100"]

    def get_cluster_labels(self, query, query_labels, **kwargs):
        num_clusters = len(set(query_labels.flatten()))
        return run_kmeans(query, num_clusters)

    def calculate_NMI(self, query_labels, cluster_labels, **kwargs):
        return normalized_mutual_info_score(query_labels, cluster_labels)

    def calculate_AMI(self, query_labels, cluster_labels, **kwargs):
        return adjusted_mutual_info_score(query_labels, cluster_labels)

    def calculate_precision_at_1(self, knn_labels, query_labels, **kwargs):
        return precision_at_k(knn_labels, query_labels[:, None], 1)

    def calculate_mean_average_precision_at_r(self, knn_labels, query_labels, embeddings_come_from_same_source=False,
                                              label_counts=None, **kwargs):
        return mean_average_precision_at_r(knn_labels, query_labels[:, None], embeddings_come_from_same_source,
                                           label_counts)

    def calculate_mean_average_precision_at_100(self, knn_labels_100, query_labels, embeddings_come_from_same_source=False,
                                              label_counts=None, **kwargs):
        return mean_average_precision_at_100(knn_labels_100, query_labels[:, None], embeddings_come_from_same_source,
                                           label_counts)
    def calculate_r_precision(self, knn_labels, query_labels, embeddings_come_from_same_source=False, label_counts=None,
                              **kwargs):
        return r_precision(knn_labels, query_labels[:, None], embeddings_come_from_same_source, label_counts)

    def get_accuracy(self, query, reference, query_labels, reference_labels, embeddings_come_from_same_source,
                     include=(), exclude=()):
        embeddings_come_from_same_source = embeddings_come_from_same_source or (query is reference)
        self.curr_function_dict = self.get_function_dict(include, exclude)

        kwargs = {"query": query,
                  "reference": reference,
                  "query_labels": query_labels,
                  "reference_labels": reference_labels,
                  "embeddings_come_from_same_source": embeddings_come_from_same_source}
        if any(x in self.requires_knn() for x in self.get_curr_metrics()):
            label_counts, num_k = get_label_counts(reference_labels) 
            knn_indices = get_knn(reference, query, num_k, embeddings_come_from_same_source)
            knn_labels = reference_labels[knn_indices]
            kwargs["label_counts"] = label_counts
            kwargs["knn_labels"] = knn_labels

        if any(x in self.requires_clustering() for x in self.get_curr_metrics()):
            kwargs["cluster_labels"] = self.get_cluster_labels(**kwargs)

        if any(x in self.requires_100nn() for x in self.get_curr_metrics()):
            knn_indices = get_knn(reference, query, 100, embeddings_come_from_same_source)
            knn_labels = reference_labels[knn_indices]
            kwargs["knn_labels_100"] = knn_labels
        
        return self._get_accuracy(self.curr_function_dict, **kwargs)

    def _get_accuracy(self, function_dict, **kwargs):
        return {k: v(**kwargs) for k, v in function_dict.items()}





