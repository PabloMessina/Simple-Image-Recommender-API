import json
import heapq
from os import path as ospath
import numpy as np

def read_ids_file(dirpath, ids_filename):
    with open(ospath.join(dirpath, ids_filename)) as f:
        if ids_filename[-4:] == 'json':
            index2id = json.load(f)
        else:
            assert ids_filename[-3:] == 'ids'
            index2id = [int(x) for x in f.readlines()]
        id2index = {_id:i for i, _id in enumerate(index2id)}
    return index2id, id2index

def get_top_k_indexes(scores_array, k):
    heap = []
    for i, score in enumerate(scores_array):
        t = (score, i)
        if len(heap) < k:
            heapq.heappush(heap, t)
        else:
            heapq.heappushpop(heap, t)
    heap.sort(reverse=True)
    assert len(heap) == k
    return [t[1] for t in heap]