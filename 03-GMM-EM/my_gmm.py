# Author: Sining Sun , Zhanheng Yang
# Modified by Hallboo

import sys
import numpy as np
from utils import *
import scipy.cluster.vq as vq

num_gaussian = 5
num_iterations = 5
targets = ['Z', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'O']

class GMM(object):
    def __init__(self, D, K=5, data_for_init = 'train/feats.scp'):
        assert(D>0)
        self.dim = D
        self.K   = K

        self.mu, self.sigma, self.pi = self.kmeans_initial(data_for_init)

    def kmeans_initial(self, data_path):
        mu    = []
        sigma = []
        data  = read_all_data(data_path)
        centroids, labels = vq.kmeans2(data, self.K, minit="points", iter=100)

        clusters = [[] for i in range(self.K)]
        for (l,d) in zip(labels, data):
            clusters[l].append(d)
        for cluster in clusters:
            mu.append(np.mean(cluster, axis=0))
            sigma.append(np.cov(cluster, rowvar=0))
        mu = np.array(mu)
        sigma = np.array(sigma)
        pi = np.array([float(len(c)) / len(data) for c in clusters])

        return mu, sigma, pi

    def gaussian(self, x, mu, sigma, eps = 1e-4):
        D = x.shape[0]
        det_sigma = np.linalg.det(sigma)
        inv_sigma = np.linalg.inv(sigma + eps)
        mahalanobis = np.dot(np.transpose(x-mu), inv_sigma)
        mahalanobis = np.dot(mahalanobis, (x-mu))
        const_a = 1/((2*np.pi)**(D/2))
        const_b = (det_sigma)**(-0.5)
        return const_a * const_b * np.exp(-0.5 * mahalanobis)

    def calc_log_likelihood(self, X):
        log_llh = 0.0

        for n in range(X.shape[0]):
            p = 0.0
            for k in range(self.K):
                p += self.pi[k] * self.gaussian(X[n], self.mu[k], self.sigma[k])
            log_llh += np.log(p)
        return log_llh

    def calc_gama_Znk(self, X):
        gama_Znk = []
        for n in range(X.shape[0]):
            ps = []
            for k in range(self.K):
                ps.append(self.pi[k] * self.gaussian(
                    X[n], self.mu[k], self.sigma[k]))
            ps = np.array(ps)
            gama_Znk.append(ps/np.sum(ps))
        return np.array(gama_Znk)

    def update_sigma(self, X, gama_Znk, Nk, mu_new):
        temp = []
        for k in range(self.K):
            sums = np.zeros((self.dim, self.dim))
            for n in range(X.shape[0]):
                diff = X[n] - mu_new[k]
                diff = diff.reshape(self.dim, 1)
                sums+= gama_Znk[n][k] * np.dot(diff, diff.T)
            temp.append(sums)
        Nk = np.array([t*np.ones((self.dim,self.dim)) for t in Nk])
        return np.array(temp)/Nk

    def update_mu(self, X, gama_Znk, Nk):
        mu_new = np.zeros((self.K, self.dim))
        sums = np.zeros((self.K, self.dim))
        for n in range(X.shape[0]):
            sums+=np.dot(gama_Znk[n].reshape(self.K,1), X[n].reshape(1,self.dim))
        return sums/(np.dot(Nk.reshape(self.K,1), np.ones((1,self.dim))))

    def em_estimator(self, X):

        log_llh = 0.0

        log_llh = self.calc_log_likelihood(X)
        gama_Znk = self.calc_gama_Znk(X)
        Nk = np.sum(gama_Znk, axis=0)
        N  = gama_Znk.shape[0]

        mu_new = self.update_mu(X, gama_Znk, Nk)
        sigma_new = self.update_sigma(X, gama_Znk, Nk, mu_new)
        pi_new = Nk / N

        self.mu = mu_new
        self.sigma = sigma_new
        self.pi = pi_new

        return log_llh

def train(gmms, num_iterations = num_iterations):
    dict_utt2feat, dict_target2utt = read_feats_and_targets(
            'train/feats.scp', 'train/text')
    
    for target in targets:
        feats = get_feats(target, dict_utt2feat, dict_target2utt)   #
        for i in range(num_iterations):
            log_llh = gmms[target].em_estimator(feats)
            print("Target:{} iteration:{} likelyhood:{}".format(
                target, i, log_llh))
    return gmms

def test(gmms):
    correction_num = 0
    error_num = 0
    acc = 0.0
    dict_utt2feat, dict_target2utt = read_feats_and_targets('test/feats.scp', 'test/text')
    dict_utt2target = {}
    for target in targets:
        utts = dict_target2utt[target]
        for utt in utts:
            dict_utt2target[utt] = target
    for utt in dict_utt2feat.keys():
        feats = kaldi_io.read_mat(dict_utt2feat[utt])
        scores = []
        for target in targets:
            scores.append(gmms[target].calc_log_likelihood(feats))
        predict_target = targets[scores.index(max(scores))]
        if predict_target == dict_utt2target[utt]:
            correction_num += 1
        else:
            error_num += 1
    acc = correction_num * 1.0 / (correction_num + error_num)
    return acc

def main():
    gmms = {}
    for target in targets:
        gmms[target] = GMM(39, num_gaussian, 'train/feats.scp')
    gmms = train(gmms)
    acc = test(gmms)
    print('Recognition accuracy: %f' % acc)
    fid = open('acc.txt', 'w')
    fid.write(str(acc))
    fid.close()

if __name__=="__main__":
    main()
