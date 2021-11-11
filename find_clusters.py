import itertools
from collections import defaultdict
from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture
from pandas import *
from sklearn.mixture import BayesianGaussianMixture
from scipy import linalg
from collections import Counter
import numpy as np
import warnings
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
import sys
from sklearn import preprocessing
from bbox_IoU import bbox_IoU
import time

warnings.filterwarnings('ignore')

color_iter = itertools.cycle(['navy', 'c', 'cornflowerblue', 'gold', 'darkorange'])


# record experiment logs
class Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass
path = os.path.abspath(os.path.dirname(__file__))
type = sys.getfilesystemencoding()
sys.stdout = Logger('iou.txt')


# plot points and gaussian ellipses
def plot_results(X, Y_, means, covariances, covtype, index, title):
    splot = plt.subplot(2, 1, 1+index)
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.5)
    for i, (mean, covar, color) in enumerate(zip(
            means, covariances, color_iter)):
        if covtype == 'full':
            covariances = covar[:2, :2]
        elif covtype == 'tied':
            covariances = covariances[:]
        elif covtype == 'diag':
            covariances = np.diag(covar[:2])
        elif covtype == 'spherical':
            covariances = np.eye(mean.shape[0]) * covar

        v, w = linalg.eigh(covariances)
        v = 2. * np.sqrt(2.) * np.sqrt(v)
        u = w[0] / linalg.norm(w[0])
        # as the DP will not use every component it has access to
        # unless it needs it, we shouldn't plot the redundant
        # components.
        if not np.any(Y_ == i):
            continue
        plt.scatter(X[Y_ == i, 0], X[Y_ == i, 1], 2, color=color)

        # Plot an ellipse to show the Gaussian component
        angle = np.arctan(u[1] / u[0])
        angle = 180. * angle / np.pi  # convert to degrees
        ell = mpl.patches.Ellipse(mean, v[0], v[1], 180. + angle, color=color)
        ell.set_clip_box(splot.bbox)
        ell.set_alpha(0.5)
        splot.add_artist(ell)

    # plt.xticks(())
    # plt.yticks(())
    plt.xlabel('rcs')
    plt.ylabel('direction vector')
    plt.title(title)


def ext_objects_and_locations(lst):
    """
        count how many objects and summary their indices
        :param lst: Input list
        :return: k: list of objects, v: corresponding indices
    """

    k = []
    v = []
    dd = defaultdict(list)
    for val, key in enumerate(lst):
        dd[key].append(val)
    for key, value in dd.items():
        k.append(key)
        v.append(value)
    return k, v


# see the clustering results, (can ignore this function)
def calc_accuracy(pred, track):
    """pred: the predicted points by GMM/VBGM
       track: corresponding tracks of points
       return: the total accuracy of the current batch"""
    gt_ids, gt_cnts = np.unique(track, return_counts=True)  # gt track_ids and their counts
    print('gt_ids:', gt_ids, 'gt_cnts:', gt_cnts)
    pred_clus, pred_locs = ext_objects_and_locations(pred)  # predicted clusters and their locations
    total_accuracy = 0
    for i in range(len(pred_clus)):
        locs = pred_locs[i]
        q = []
        for j in range(len(locs)):
            q.append(track[locs[j]])
        word_counts = Counter(q)
        print('word_counts:', word_counts, 'length of locs:', len(locs))
        top_one = word_counts.most_common(1)
        top_clus = top_one[0][0]  # the most common track_ids in the current cluster, the label of the cluster
        print('top_clus:', top_clus, 'top_clus_num:', top_one[0][1])
        top_idx = np.where(gt_ids == top_clus) # find the corresponding track_id in ground-truth
        top_gt_cnts = gt_cnts[top_idx[0]]
        accuracy = top_one[0][1] / top_gt_cnts # the predicted points/ the gt points
        total_accuracy += accuracy

    return total_accuracy/len(gt_ids)


# localization accuracy (IoU) calculation
def region_comparison(gtfeature, predlabel, unscaled_data, gttracks):
    gt_clus, gt_locs = ext_objects_and_locations(gttracks)
    for g in range(len(gt_clus)):
        if gt_clus[g] != b'':
            gt_idx = gt_locs[g]
    gt_points = gtfeature[gt_idx]

    max_iou = 0
    pred_clus, pred_locs = ext_objects_and_locations(predlabel)
    for i in range(len(pred_locs)):
        pre_idx = pred_locs[i]
        pred_points = unscaled_data[pre_idx]
        iou = bbox_IoU(pred_points, gt_points)
        if iou > max_iou:
            max_iou = iou

    return max_iou


def calc_gmm(uni_segdata, gtdata, gttrack):

    # data scale
    array0 = uni_segdata[:, 3].reshape(-1, 1)
    array1 = uni_segdata[:, 5].reshape(-1, 1)

    unscaled_segdata = np.concatenate((array0, array1), axis=1)

    b0 = preprocessing.MinMaxScaler().fit_transform(array0)
    b1 = preprocessing.MaxAbsScaler().fit_transform(array1)
    scaled_segdata = np.concatenate((b0, b1), axis=1)

    gtdata0 = gtdata[:, 3].reshape(-1, 1)
    gtdata1 = gtdata[:, 5].reshape(-1, 1)
    gtdata_2d = np.concatenate((gtdata0, gtdata1), axis=1)

    # DBSCAN + GMM
    clustering = DBSCAN(eps=0.3, min_samples=3).fit(scaled_segdata)
    core_samples_mask = np.zeros_like(clustering.labels_, dtype=bool)
    core_samples_mask[clustering.core_sample_indices_] = True
    labels = clustering.labels_

    #num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    #num_noise = list(labels).count(-1)
    num_clusters = len(set(labels))
    print('number of clusters estimated by DBSCAN:', num_clusters)

    if num_clusters == 0:
        num_clusters = 1
        print('orignal num_cluster is 0')

    gmm_dbscan = GaussianMixture(n_components=num_clusters).fit(scaled_segdata)

    # BIC criterion + GMM
    lowest_bic = np.infty
    bic = []
    n_components_range = range(1, 10)
    try:
        for n_components in n_components_range:
            gmm_bic = GaussianMixture(n_components=n_components,
                                      covariance_type='full').fit(scaled_segdata)
            bic.append(gmm_bic.bic(scaled_segdata))
            if bic[-1] < lowest_bic:
                lowest_bic = bic[-1]
                best_gmm = gmm_bic
    except:
        best_gmm = GaussianMixture(n_components=num_clusters).fit(scaled_segdata)
        print('bic error, same as DBSCAN')

    print('gmm_bic:', best_gmm)

    gmm_diou = region_comparison(gtdata_2d, gmm_dbscan.predict(scaled_segdata), unscaled_segdata, gttrack)
    gmm_biou = region_comparison(gtdata_2d, best_gmm.predict(scaled_segdata), unscaled_segdata, gttrack)
    print('gmm_diou', gmm_diou)
    print('gmm_biou', gmm_biou)

    # GMM_accuracy = calc_accuracy(gmm_dbscan.predict(data), track)
    # print('GMM_dbscan accuracy:', GMM_accuracy)
    # GMM_bic_accuracy = calc_accuracy(best_gmm.predict(data), track)
    # print('GMM_bic accuracy:', GMM_bic_accuracy)

    # plt.figure()
    # plot_results(data, gmm_dbscan.predict(data), gmm_dbscan.means_, gmm_dbscan.covariances_, gmm_dbscan.covariance_type, 0,
    #             'DBSCAN + GMM')
    # plot_results(data, best_gmm.predict(data), best_gmm.means_, best_gmm.covariances_, best_gmm.covariance_type, 1,
    #              'BIC + GMM')
    #
    # path = 'F:/Desktop/figs'
    # filename = 'gmm %d' % (cnt) + '.pdf'
    # plt.savefig(os.path.join(path, filename))
    #plt.show()

    # DBSCAN + VBGM(dirichlet process)
    vbgm1 = BayesianGaussianMixture(n_components=num_clusters, weight_concentration_prior_type='dirichlet_process',
                                    covariance_type='full').fit(scaled_segdata)

    # DBSCAN + VBGM(dirichlet distribution)
    vbgm2 = BayesianGaussianMixture(n_components=num_clusters, weight_concentration_prior_type='dirichlet_distribution',
                                    covariance_type='full').fit(scaled_segdata)
    vbgm1_iou = region_comparison(gtdata_2d, vbgm1.predict(scaled_segdata), unscaled_segdata, gttrack)
    vbgm2_iou = region_comparison(gtdata_2d, vbgm2.predict(scaled_segdata), unscaled_segdata, gttrack)
    print('vbgm1_iou', vbgm1_iou)
    print('vbgm2_iou', vbgm2_iou)


    # vbgm1_accuracy = calc_accuracy(vbgm1.predict(data), track)
    # print('vbgm1 accuracy:', vbgm1_accuracy)
    # vbgm2_accuracy = calc_accuracy(vbgm2.predict(data), track)
    # print('vbgm2 accuracy:', vbgm2_accuracy)

    # plt.figure()
    # plot_results(data, vbgm1.predict(data), vbgm1.means_, vbgm1.covariances_, vbgm1.covariance_type, 0,
    #  'VBGM + Dir-Process')
    # plot_results(data, vbgm2.predict(data), vbgm2.means_, vbgm2.covariances_, vbgm2.covariance_type, 1,
    #  'VBGM + Dir-Distribution')
    #
    # filename1 = 'vbgm %d' % (cnt) + '.pdf'
    # plt.savefig(os.path.join(path, filename1))
    #plt.show()

    return gmm_diou, gmm_biou, vbgm1_iou, vbgm2_iou


if __name__ == '__main__':
    time_start = time.time()

    # load segmented features and gt-features
    path1 = 'F:/Desktop/seg_trackids'
    path2 = 'F:/Desktop/seg_features'
    gtpath1 = 'F:/Desktop/ini_trackids'
    gtpath2 = 'F:/Desktop/ini_features'

    ct = 0
    gdTotal = 0
    gbTotal = 0
    v1Total = 0
    v2Total = 0

    diff = []
    FN = 0
    FP = 0
    TP = 0

    for i in range(1512):
        print('i:', i)

        # load segmented features and gt-features
        rdf = 'feature %d' % (i) + '.npy'
        rdfilepath = os.path.join(path2, rdf)
        seg_feature = np.load(rdfilepath)

        tf = 'track_id %d' % (i) + '.npy'
        tfilepath = os.path.join(path1, tf)
        seg_track = np.load(tfilepath)

        gt_tf = 'track_id %d' % (i) + '.npy'
        gt_tfilepath = os.path.join(gtpath1, gt_tf)
        gt_tracks = np.load(gt_tfilepath)

        gt_rf = 'feature %d' % (i) + '.npy'
        gt_rfilepath = os.path.join(gtpath2, gt_rf)
        gt_feature = np.load(gt_rfilepath)

        if (len(seg_track) == 0) or (seg_track[0] == 0) or (len(seg_feature) < 2):
            continue  # no segmented points or only one segmented point
        else:
            if (len(set(seg_track)) == 1) and (seg_track[0] == b'') and (len(set(gt_tracks)) > 1):
                FN += 1
                # gt exists, no detection
                # continue
            elif (len(set(gt_tracks)) == 1) and (gt_tracks[0] == b'') and (len(set(seg_track)) > 1):
                FP += 1
                # detection exists, no gt
            elif (len(set(gt_tracks)) == 1) and (gt_tracks[0] == b'') and (len(set(seg_track)) == 1) and (seg_track[0] == b''):
                # no detection and no gt
                continue
            else:
                # both detection and gt exist
                # make segmented features unique points
                uni_segfeatures = DataFrame(seg_feature).drop_duplicates().values
                TP += 1
                # track_unique = []
                # for j in range(len(uni_segfeatures)):
                #     for k in range(len(seg_feature)):
                #         if (seg_feature[k, 3] == uni_segfeatures[j, 3]) and (seg_feature[k, 5] == uni_segfeatures[j, 5]):
                #             track_unique.append(seg_track[k])
                #             break
                if len(uni_segfeatures) > 1:
                    # calculate gmm as long as there are points in the scene
                    gd, gb, v1, v2 = calc_gmm(uni_segfeatures, gt_feature, gt_tracks)
                    ct += 1
                    gdTotal += gd
                    gbTotal += gb
                    v1Total += v1
                    v2Total += v2

                    if (abs(gd-v1) > 0.2) or (abs(gd-v2) > 0.2):
                        diff.append(i)

                print('-' * 50)

    gdTotal = gdTotal/ct
    gbTotal = gbTotal/ct
    v1Total = v1Total/ct
    v2Total = v2Total/ct

    print('-' * 50)
    print('Average IoU by GMM with DBSCAN:', gdTotal)
    print('Average IoU by GMM with BIC:', gbTotal)
    print('Average IoU by VBGM with Dirichlet Process:', v1Total)
    print('Average IoU by VBGM with Dirichlet Distribution:', v2Total)

    print('TP:', TP, 'FP:', FP, 'FN:', FN)
    print('ct:', ct)

    print('difference > 0.2 scenes:', diff)

    time_end = time.time()
    print('time cost:', time_end - time_start, 's')
