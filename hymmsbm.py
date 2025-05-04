import numpy as np
import math
from scipy.sparse import csr_matrix
import hypergraph
import random
from itertools import combinations

class HyMMSBM:
    N = 0
    M = 0
    B = []
    K = 0
    U = []
    W = []
    poi_lambda = []
    S = []
    C = []
    C_for_U = []
    C_for_W = []
    D = 0
    random_state = None

    def __init__(self, G: hypergraph.HyperGraph, K, random_state=None):
        self.E = np.array([set(G.E[m]) for m in range(0, G.M)], dtype=set)
        self.A = np.array([int(G.A[m]) for m in range(0, G.M)], dtype=int)
        self.N = G.N
        self.M = G.M
        self.B = np.zeros((G.N, G.M), dtype=int)
        self.K = K
        self.U = np.zeros((self.N, self.K), dtype=float)
        self.W = np.zeros((self.K, self.K), dtype=float)
        self.poi_lambda = np.zeros(self.M, dtype=float)
        self.S = np.zeros((self.M, self.K), dtype=float)
        self.C_for_U = np.zeros((G.N, K), dtype=float)
        self.C_for_W = np.zeros((K, K), dtype=float)
        self.D = max([len(e) for e in G.E])
        self.EPS = 1e-10
        self.random_state = random_state

        # Incidence matrix
        self.B = np.zeros((self.N, self.M), dtype=int)
        for m in range(0, self.M):
            for i in self.E[m]:
                self.B[i][m] = 1
        self.B = csr_matrix(self.B)

        # Constant C
        self.C = sum([float(2)/(s*(s-1)) for s in range(2, self.D+1)])
        self.C_for_U = np.full((self.N, self.K), self.C)
        self.C_for_W = np.full((self.K, self.K), self.C)

        return

    def initialize_params(self, random_state):
        rng = np.random.default_rng(random_state)

        # Matrix U
        self.U = rng.random((self.N, self.K))

        # Matrix W
        self.W = rng.random((self.K, self.K))
        self.W = np.triu(self.W, 0) + np.triu(self.W, 1).T

        # Matrix S
        self.S = self.B.transpose() @ self.U

        # lambda_e
        first_addend = ((self.S @ self.W) * self.S).sum(axis=-1)
        second_addend = self.B.T @ (((self.U @ self.W) * self.U).sum(axis=-1))
        self.poi_lambda = 0.5 * (first_addend - second_addend)
        self.poi_lambda = np.where(self.poi_lambda < self.EPS, self.EPS, self.poi_lambda)

        return

    def check_initial_parameters(self):
        U_sum = self.U.sum(axis=1)
        for i in range(0, self.N):
            if math.isclose(U_sum[i], 0):
                print("Error: sum of u_ik for i=" + str(i) + " is zero.")
                return False

        for k in range(0, self.K):
            for q in range(0, self.K):
                if math.isclose(self.W[k][q], 0):
                    print("Error: sum of w_kq for k=" + str(k) + " and q=" + str(q) + " is zero.")
                    return False

        S_sum = self.S.sum(axis=1)
        for m in range(0, self.M):
            if math.isclose(S_sum[m], 0):
                print("Error: sum of s_mk for m=" + str(m) + " is zero.")
                return False

        for m in range(0, self.M):
            if math.isclose(self.poi_lambda[m], 0):
                print("Error: lambda_m for m=" + str(m) + " is zero.")
                return False

        return True

    def update_u(self):
        # Numerator
        multiplier = self.A / (2.0 * self.poi_lambda)
        weighting = self.B.multiply(multiplier[None, :])
        first_addend = weighting @ self.S
        weighting_sum = np.asarray(weighting.sum(axis=1)).reshape(-1, 1)
        second_addend = weighting_sum * self.U
        num = 2.0 * (self.U * np.matmul(first_addend - second_addend, self.W))

        # Denominator
        U_sum = self.U.sum(axis=0)
        den = (self.C_for_U * (np.matmul(self.W, U_sum)[None, :] - np.matmul(self.U, self.W)))
        den = np.where(den < self.EPS, self.EPS, den)

        # Update U
        self.U = num / den

        # Matrix S
        self.S = self.B.transpose() @ self.U

        # lambda_e
        first_addend = ((self.S @ self.W) * self.S).sum(axis=-1)
        second_addend = self.B.T @ (((self.U @ self.W) * self.U).sum(axis=-1))
        self.poi_lambda = 0.5 * (first_addend - second_addend)
        self.poi_lambda = np.where(self.poi_lambda < self.EPS, self.EPS, self.poi_lambda)

        return

    def update_w(self):
        # Numerator
        multiplier = self.A / (2.0 * self.poi_lambda)
        first_addend = np.matmul(self.S.T, self.S * multiplier[:, None])
        weighting = self.B.multiply(multiplier[None, :]).sum(axis=1)
        weighting = np.asarray(weighting).reshape(-1)
        second_addend = np.matmul(self.U.T, self.U * weighting[:, None])
        num = 2.0 * (self.W * (first_addend - second_addend))

        # Denominator
        u_sum = self.U.sum(axis=0)
        den = (np.outer(u_sum, u_sum) - np.matmul(self.U.T, self.U)) * self.C_for_W
        den = np.where(den < self.EPS, self.EPS, den)

        # Update W
        self.W = num / den

        # Matrix S
        self.S = self.B.transpose() @ self.U

        # lambda_e
        first_addend = ((self.S @ self.W) * self.S).sum(axis=-1)
        second_addend = self.B.T @ (((self.U @ self.W) * self.U).sum(axis=-1))
        self.poi_lambda = 0.5 * (first_addend - second_addend)
        self.poi_lambda = np.where(self.poi_lambda < self.EPS, self.EPS, self.poi_lambda)

        return

    def calc_loglik(self):
        U_sum = self.U.sum(axis=0)
        first_addend = self.C * 0.5 * (((U_sum @ self.W) * U_sum).sum(axis=-1) - ((self.U @ self.W) * self.U).sum())
        second_addend = np.dot(self.A, np.log(self.poi_lambda))

        return (-1) * first_addend + second_addend

    def fit(self, initial_r=10, num_step=100, tol=1e-3):

        best_loglik = float("-inf")
        best_param = None
        r_count = 0

        num_step_lst = []

        for i in range(0, initial_r):
            if self.random_state == None:
                self.initialize_params(None)
            else:
                self.initialize_params(self.random_state + r_count)
            r_count += 1

            while not self.check_initial_parameters():
                self.initialize_params(self.random_state + r_count)
                r_count += 1

            j = 0
            loglik_conv = False
            pre_loglik = float("-inf")
            while j < num_step and loglik_conv == False:
                self.update_u()
                self.update_w()

                L = self.calc_loglik()
                if j > 0:
                    loglik_conv = float(math.fabs(L - pre_loglik)) / math.fabs(pre_loglik) < tol
                pre_loglik = L
                j += 1

            L = self.calc_loglik()

            if L > best_loglik:
                best_loglik = L
                best_param = (self.U, self.W)

            num_step_lst.append(j)

        print("Numbers of steps", num_step_lst)

        return best_loglik, best_param

class HyperParamTuning:
    K_lst = []

    def __init__(self, G):
        #max_K = max(G.Z, 2)
        max_K = 15
        self.K_lst = list(range(2, max_K + 1))

        return

    def construct_train_and_test_sets(self, G, p=0.2):
        hyperedge_indices = list(range(0, G.M))
        random.shuffle(hyperedge_indices)

        E_train, E_test = [G.E[hyperedge_indices[i]] for i in range(0, int((1 - p) * G.M))], [G.E[hyperedge_indices[i]] for i in range(int((1 - p) * G.M), G.M)]
        A_train, A_test = [G.A[hyperedge_indices[i]] for i in range(0, int((1 - p) * G.M))], [G.A[hyperedge_indices[i]] for i in range(int((1 - p) * G.M), G.M)]

        G_train = hypergraph.HyperGraph(G.N, int(len(E_train)), G.Z)
        G_train.E = E_train
        G_train.A = A_train
        G_train.X = G.X

        G_test = hypergraph.HyperGraph(G.N, int(len(E_test)), G.Z)
        G_test.E = E_test
        G_test.A = A_test
        G_test.X = G.X

        return G_train, G_test

    def grid_search(self, G, num_runs=100):

        auc_score = np.zeros((len(self.K_lst), num_runs), dtype=float)
        for k in range(0, len(self.K_lst)):
            K = self.K_lst[k]
            for r in range(0, num_runs):
                G_train, G_test = self.construct_train_and_test_sets(G)

                model = HyMMSBM(G_train, K)
                best_loglik, (U, W) = model.fit()

                auc = 0.0
                sampled_edges = []
                for m in range(0, G_test.M):
                    e, e_ = set(G_test.E[m]), set()
                    s = len(e)
                    flag = True
                    while flag:
                        e_ = set(random.sample(range(G_test.N), k=s))
                        if len(e_) == s and e_ not in G.E and e_ not in sampled_edges:
                            flag = False
                            sampled_edges.append(e_)

                    param0 = sum([(U[i_] * W * U[j_].T).sum() for (i_, j_) in list(combinations(sorted(list(e_)), 2))])
                    param1 = sum([(U[i_] * W * U[j_].T).sum() for (i_, j_) in list(combinations(sorted(list(e)), 2))])

                    if param1 > param0:
                        auc += 1.0
                    elif math.isclose(param0, param1):
                        auc += 0.5

                auc = float(auc) / G_test.M
                auc_score[k][r] = auc

            print(K, np.mean(auc_score[k]), np.std(auc_score[k]))

        return auc_score

    def run(self, G: hypergraph.HyperGraph):
        auc_score = self.grid_search(G)

        best_auc, best_hyperparm = (-1, -1), (-1, -1)
        for k in range(0, auc_score.shape[0]):
            if np.mean(auc_score[k]) > best_auc[0]:
                best_auc = (np.mean(auc_score[k]), np.std(auc_score[k]))
                best_hyperparm = self.K_lst[k]

        return best_auc, best_hyperparm, auc_score
