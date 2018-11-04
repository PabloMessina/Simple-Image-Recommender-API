import tensorflow as tf
import numpy as np
import random
from os import path as ospath
from Networks import YoutubeLikeNetwork
from utils import read_ids_file, get_top_k_indexes

class YoutubeLike2StagesModelHandler:
    def __init__(
            self,
            youtube_model_path,
            youtube_precomputed_tensors_path,
            resnet_precomputed_tensors_path):
        self._network = YoutubeLikeNetwork(user_model_mode='BIGGER')
        self._sess = tf.Session()
        saver = tf.train.Saver()
        saver.restore(self._sess, tf.train.latest_checkpoint(youtube_model_path))

        self._youtube_item_vectors = np.load(
            ospath.join(youtube_precomputed_tensors_path, 'item_vectors.npy'))        
        self._youtube_index2id,\
        self._youtube_id2index = read_ids_file(youtube_precomputed_tensors_path, 'ids')

        self._resnet_simmat = np.load(
            ospath.join(resnet_precomputed_tensors_path, 'flatten_1_pca200_simmat.npy'))
        self._resnet_cluster_labels = np.load(
            ospath.join(resnet_precomputed_tensors_path, 'cluster_labels.npy'))
        self._resnet_index2id,\
        self._resnet_id2index = read_ids_file(resnet_precomputed_tensors_path, 'ids')

    def _filter_dataset_with_youtube(self, liked_ids, dataset_ids, rec_size):
        id2index = self._youtube_id2index
        index2id = self._youtube_index2id
        liked_ids_set = set(liked_ids)

        profile_indexes = [id2index[_id] for _id in liked_ids]
        candidate_indexes = [id2index[_id] for _id in dataset_ids if _id not in liked_ids_set]
        
        match_scores = self._network.get_match_scores(
            sess = self._sess,
            precomputed_item_vectors = self._youtube_item_vectors,
            profile_item_indexes = profile_indexes,
            candidate_item_indexes = candidate_indexes)

        assert match_scores.shape == (len(candidate_indexes),)
        prefilter_k = min(rec_size * 10, rec_size + 100)
        filtered_ids = [index2id[candidate_indexes[i]] for i in get_top_k_indexes(
            match_scores, prefilter_k)]
        assert len(filtered_ids) == prefilter_k
        return  filtered_ids

    def _rerank_and_recommend_with_resnet(self, liked_ids, filtered_ids, rec_size, explained):
        simmat = self._resnet_simmat
        cluster_labels = self._resnet_cluster_labels
        id2index = self._resnet_id2index
        index2id = self._resnet_index2id
        filtered_indexes = [id2index[_id] for _id in filtered_ids]
        liked_indexes = [id2index[_id] for _id in liked_ids]

        # n_likes = len(liked_ids)
        n_filt = len(filtered_ids)
        assert n_filt >= rec_size

        n_clusters = cluster_labels.shape[0]
        cluster_offset = np.zeros((n_clusters,), dtype=int)
        cluster_count = np.zeros((n_clusters,), dtype=int)        
        filt_maxsim = np.empty((n_filt,), dtype=float)
        filt_maxl = np.empty((n_filt,), dtype=int)
        filt_clabel = np.empty((n_filt,), dtype=int)
        filt_i2f = np.empty((n_filt,), dtype=int)
        
        for i, f in enumerate(filtered_indexes):
            maxsim = -99999
            maxl = None
            for l in liked_indexes:
                if simmat[f][l] > maxsim:
                    maxsim = simmat[f][l]
                    maxl = l
            assert maxl != None
            cl = cluster_labels[maxl]
            cluster_count[cl] += 1
            filt_maxsim[i] = maxsim
            filt_maxl[i] = maxl
            filt_clabel[i] = cl
            filt_i2f[i] = f

        filt_is = list(range(n_filt))
        filt_is.sort(key=lambda i: (filt_clabel[i], filt_maxsim[i]))

        for cl in range(1, n_clusters):
            cluster_offset[cl] = cluster_offset[cl-1] + cluster_count[cl-1]
            
        used_clabels = [cl for cl in range(n_clusters) if cluster_count[cl] > 0]
        top_is = []
        while True:
            used_clabels.sort(key=lambda cl: filt_maxsim[
                filt_is[cluster_offset[cl] + cluster_count[cl] - 1]
            ], reverse=True)
            for cl in used_clabels:
                i = filt_is[cluster_offset[cl] + cluster_count[cl] - 1]
                top_is.append(i)
                if (len(top_is) == rec_size): break
                cluster_count[cl] -= 1
            if (len(top_is) == rec_size): break
            used_clabels = [cl for cl in used_clabels if cluster_count[cl] > 0]
        
        assert len(top_is) == rec_size
        top_is.sort(key=lambda i: filt_maxsim[i], reverse=True)

        if explained:
            recommendation = [dict(
                id = index2id[filt_i2f[i]],
                explanation = ((filt_maxsim[i], index2id[filt_maxl[i]]),)
            ) for i in top_is]
        else:
            recommendation = [dict(id = index2id[filt_i2f[i]]) for i in top_is]

        return recommendation

    def get_recommendation(self, liked_ids, dataset_ids, rec_size, explained):
        if len(liked_ids) > 50:
            liked_ids = random.sample(liked_ids, 50)
        filtered_ids = self._filter_dataset_with_youtube(
            liked_ids, dataset_ids, rec_size)
        recommendation = self._rerank_and_recommend_with_resnet(
            liked_ids, filtered_ids, rec_size, explained)        
        return recommendation