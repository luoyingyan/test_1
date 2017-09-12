# -*- coding: utf-8 -*-
"""
Created on Sat Sep  9 21:27:22 2017
Reffence ： sklearn.kmean
"""

import numpy as np
from math import sqrt
from scipy import linalg
from scipy.sparse import issparse, csr_matrix

# calculate distance :
def norms(X):

    X = np.asarray(X)
    nrm2, = linalg.get_blas_funcs(['nrm2'], [X])
    return nrm2(X)

def row_norms(X, squared = False):
    """
    :param X:  matrix or array
    :param squared: squared Eulidean norm or not
    :return: Row-wise Euclidean norm of X
    eg. X = [[0,1,2],[3,4,5]]
    return: [5, 50]
    """
    norms = np.einsum('ij, ij->i', X, X)

    if not squared:
        norms = np.sqrt(norms)
    return norms

def euclidean_distances(X, Y, squared = False):

    XX = row_norms(X, squared = True)[:,np.newaxis]
    YY = row_norms(Y, squared = True)[np.newaxis,]

    distances = np.dot(X, Y.T)
    distances *= -2
    distances += XX
    distances += YY
    distances = np.maximum(distances, 0)

    return distances if squared else np.sqrt(distances)

def distEclud(u, v):
    """
    return scala, u and v are vector, array-like
    """
    return np.sqrt(sum(np.power(u - v, 2)))

def _ini_cent(X, n_clusters, n_local_trials = None):
    """

    k-means++ algorithm
    returns
    -------
    centers: array, shape(k, n_feature)
    """
    n_samples, n_features = X.shape
    
    centers = np.empty((n_clusters, n_features), dtype = X.dtype)

    if n_local_trials is None:

        n_local_trials = 2 + int(np.log(n_clusters))
    
    #pick first center randomly
    center_id = np.random.randint(n_samples)
    centers[0] = X[center_id]
    
    #Initialize list of closest distances and calculate current potential
    closest_dist_sq = np.array(euclidean_distances(centers[0, np.newaxis], X, squared = True))
    current_pot = closest_dist_sq.sum()

    #pick the remaining n_clusters - 1 points
    for c in range(1, n_clusters):
        # Choose center candidated by sampling with probability proportional
        # to the squared distance D(x)^2 to the closest existing center
        rand_vals = np.random.random_sample(n_local_trials) * current_pot
        candidate_ids = np.searchsorted(closest_dist_sq.cumsum(), rand_vals)

        # Compute distance to center candidates
        distance_to_candidates = np.array(euclidean_distances(X[candidate_ids], X, squared = True))

        # Decide which candidate is the best
        best_candidate = None
        best_pot = None
        best_dist_sq = None

        for trial in range(n_local_trials):
            # Compute potential when including center candidate
            new_dist_sq = np.minimum(closest_dist_sq, distance_to_candidates[trial])
            new_pot = new_dist_sq.sum()

            # Store result if it is the best local trial so far
            if(best_candidate is None) or (new_pot < best_pot):
                best_candidate = candidate_ids[trial]
                best_pot = new_pot
                best_dist_sq = new_dist_sq

        # Permanently add best center candidate found in local tries
        centers[c] = X[best_candidate]
        current_pot = best_pot
        closest_dist_sq = best_dist_sq

    return centers


def my_kmean(X, n_clusters,new_centroids = None, tol = 1e-9):
    n_samples, n_features = X.shape
    clusterAssment = np.mat(np.zeros((n_samples, 2)))
    # initial centers
    if new_centroids is None:
        centroids =  _ini_cent(X, n_clusters)
    else:
        centroids = new_centroids
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
        for i in range(n_samples):
            distJ = np.array(euclidean_distances(centroids, X[i,:],squared = True))
            minIndex = np.argmin(distJ)
            minDist = distJ[minIndex][0]

            clusterAssment[i, :] = minIndex, minDist

        old_centroids = centroids
        for cent in range(n_clusters):
            ptsInClust = X[np.nonzero(clusterAssment[:,0].A == cent)[0]]
            centroids[cent, :] = np.mean(ptsInClust, axis = 0)

        if(norms(centroids - old_centroids) > tol):
            clusterChanged = True

    return centroids, clusterAssment


####################################################################################
####################################################################################
# 伪代码部分
# random divide raw data D into several parts
# different computer (site) deal with different subset of D
#  we use S(i) to represent computer(i) (site(i))
# D(i) is the local data of S(i)

# Initial：for each site S(i) do in parallel
new_centroids(i) = my_kmean(D(i), n_clusters)

# for each site S(i) broadcast new_centroids(i), for k = 1 to n_clusters to all other sites
# for each site S(i) do parallel
# for k = 1 to n_clusters do
for cent in range(n_clusters):
    new_centroids[cent] = np.mean(new_centroids(i)[cent], axis = 0)

# repeat refresh new_centroids util converge
# for each Site S(i)
new_centroids(i) = my_kmean(D(i), n_clusters, new_centroids(i))
# broadcast to other sites and received other sites replace
for cent in range(n_clusters):
    new_centroids[cent] = np.mean(new_centroids(i)[cent], axis = 0)
# until all new_centroids are stable
# end






















    
    

   
 



