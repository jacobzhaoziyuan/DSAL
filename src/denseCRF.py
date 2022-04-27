#!/usr/bin/python3  
# -*- coding: utf-8 -*-
from constants import *
import numpy as np
import pydensecrf.densecrf as dcf
from pydensecrf.utils import unary_from_softmax, create_pairwise_bilateral, create_pairwise_gaussian
from multiprocessing import Pool


class DenseCRF:
    def __init__(self, cls_num=2, scords_bi=(80, 80), scords_gau=(3, 3), schan=(10,), compat=(3, 10),
                 iter_num=20, gt_conf=0.51):
        self.cls_num = cls_num
        self.scords_bilateral = scords_bi
        self.scords_gauss = scords_gau
        self.schan = schan
        self.compat = compat
        self.iter_num = iter_num
        self.gt_conf = gt_conf
        self.scores = []

    def __call__(self, I, pred):
        assert I.shape == (img_rows, img_cols, 1), f"Shape error of I, got shape {I.shape}"
        if 1 in pred.shape:
            pred = np.squeeze(pred, axis=list(pred.shape).index(1))

        I = (I - I.min()) / (I.max() - I.min())
        d = dcf.DenseCRF2D(img_rows, img_cols, self.cls_num)
        pred = (pred - pred.min()) / (pred.max() - pred.min())
        probs = np.tile(pred[np.newaxis, :, :], (2, 1, 1))
        probs[1, :, :] = 1 - probs[0, :, :]
        probs = probs[[1, 0], ...]
        U = unary_from_softmax(probs)
        d.setUnaryEnergy(U)
        pair_eng = create_pairwise_bilateral(sdims=self.scords_bilateral, schan=self.schan,
                                             img=I.astype(np.uint8), chdim=-1)
        d.addPairwiseEnergy(pair_eng, compat=self.compat[1])
        d.addPairwiseGaussian(sxy=self.scords_gauss, compat=self.compat[0])

        Q = d.inference(self.iter_num)
        return np.argmax(Q, axis=0).reshape((img_rows, img_cols))

    @property
    def avg_score(self):
        if len(self.scores):
            return sum(self.scores) / len(self.scores)
        return 0


def average_arbitrator(maps: [np.array]):
    assert all(mp.shape == maps[0].shape for mp in maps)
    return np.array(np.mean(np.stack(maps), axis=0) >= .5).astype(maps[0].dtype)


class EnsembleDenseCRF:
    def __init__(self, configs, cls_num=2, arbitrator=average_arbitrator):
        self.arbitrator = arbitrator
        self.crfs = [DenseCRF(cls_num, **config) for config in configs]
        self.scores_per_crf = []

    def __call__(self, I, pred, output_bases=True, mp=True):
        if isinstance(pred, list) and len(pred) > 1:
            assert all(mp.shape == pred[0].shape for mp in pred)
            task = lambda crf: self.arbitrator([crf(I, p) for p in pred])
            if mp:
                pool = Pool(len(self.crfs))
                results = [pool.apply_async(task, args=(crf,)) for crf in self.crfs]
                pool.close()
                pool.join()
                maps = [res.get() for res in results]
            else:
                maps = [task(crf) for crf in self.crfs]
            ensemb = self.arbitrator(maps)
        else:
            pred = pred[0] if isinstance(pred, list) and len(pred) == 1 else pred
            if mp:
                pool = Pool(len(self.crfs))
                results = [pool.apply_async(crf, args=(I, pred)) for crf in self.crfs]
                maps = [res.get() for res in results]
                pool.close()
                pool.join()
            else:
                maps = [crf(I, pred) for crf in self.crfs]
            ensemb = self.arbitrator(maps)

        if output_bases:
            return ensemb, np.stack(maps)
        else:
            return ensemb

    @property
    def avg_score(self):
        if len(self.scores_per_crf):
            return sum(self.scores_per_crf) / len(self.scores_per_crf)
        return 0
