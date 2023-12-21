import math
from scipy.sparse import csr_matrix
import numpy as np
import hypergraph

class HyperNEO:
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
    Z = 0
    Beta = []
    X = []
    gamma = 0.0
    D = 0
    random_state = None

    def __init__(self, G: hypergraph.HyperGraph, K, gamma, random_state=None):
        self.E = np.array([set(G.E[m]) for m in range(0, G.M)], dtype=set)
        self.A = np.array([int(G.A[m]) for m in range(0, G.M)], dtype=int)
        self.X = np.zeros((G.N, G.Z), dtype=int)
        self.N = G.N
        self.M = G.M
        self.B = np.zeros((G.N, G.M), dtype=int)
        self.K = K
        self.Z = G.Z
        self.gamma = gamma
        self.U = np.zeros((self.N, self.K), dtype=float)
        self.W = np.zeros((self.K, self.K), dtype=float)
        self.poi_lambda = np.zeros(self.M, dtype=float)
        self.S = np.zeros((self.M, self.K), dtype=float)
        self.Beta = np.zeros((self.K, self.Z), dtype=float)
        self.C_for_U = np.zeros((G.N, K), dtype=float)
        self.C_for_W = np.zeros((K, K), dtype=float)
        self.D = max([len(e) for e in G.E])
        self.tol = 1e-3
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

        # Attribute matrix
        for i in range(0, self.N):
            z = int(G.X[i])
            self.X[i][z] = 1

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

        # Matrix Beta
        self.Beta = rng.random((self.K, self.Z))
        Beta_sum = self.Beta.sum(axis=1)
        Beta_sum = np.where(Beta_sum < self.EPS, self.EPS, Beta_sum)
        for k in range(0, self.K):
            self.Beta[k] /= Beta_sum[k]

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

        Beta_sum = self.Beta.sum(axis=1)
        for k in range(0, self.K):
            if math.isclose(Beta_sum[k], 0):
                print("Error: sum of beta_kz for k=" + str(k) + " is zero.")
                return False

        return True

    def update_u(self, gamma):
        # Numerator
        ## First term
        multiplier = self.A / (2.0 * self.poi_lambda)
        weighting = self.B.multiply(multiplier[None, :])
        first_addend = weighting @ self.S
        weighting_sum = np.asarray(weighting.sum(axis=1)).reshape(-1, 1)
        second_addend = weighting_sum * self.U
        first_term = (1 - gamma) * (self.U * np.matmul(first_addend - second_addend, self.W))

        ## Second term
        divider = self.U @ self.Beta
        divider = np.where(divider < self.EPS, self.EPS, divider)
        X_ = self.X / divider
        second_term = gamma * (self.U * (X_ @ self.Beta.T))

        num = 2.0 * (first_term + second_term)

        # Denominator
        U_sum = self.U.sum(axis=0)
        den = (1 - gamma) * (self.C_for_U * (np.matmul(self.W, U_sum)[None, :] - np.matmul(self.U, self.W)))
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

    def update_beta(self):
        # Numerator
        divider = self.U @ self.Beta
        divider = np.where(divider < self.EPS, self.EPS, divider)
        X_ = self.X / divider
        num = self.Beta * (self.U.T @ X_)

        # Denominator
        den = np.zeros((self.K, self.Z), dtype=float)
        num_sum = num.sum(axis=1)
        den[:, :] = num_sum[:, None]
        den = np.where(den < self.EPS, self.EPS, den)

        # Update Beta
        self.Beta = num / den

        return

    def calc_structural_loglik(self):
        U_sum = self.U.sum(axis=0)
        first_addend = self.C * 0.5 * (((U_sum @ self.W) * U_sum).sum(axis=-1) - ((self.U @ self.W) * self.U).sum())
        second_addend = np.dot(self.A, np.log(self.poi_lambda))

        return (-1) * first_addend + second_addend

    def calc_attribute_loglik(self):
        num = self.U @ self.Beta
        den = np.zeros((self.N, self.Z), dtype=float)
        den[:, :] = self.U.sum(axis=1)[:, None]
        den = np.where(den < self.EPS, self.EPS, den)
        lik = num / den
        lik = np.where(lik < self.EPS, self.EPS, lik)
        log_term = np.log(lik)

        return (self.X * log_term).sum()

    def calc_loglik(self, gamma):
        structural_loglik = self.calc_structural_loglik()
        attribute_loglik = self.calc_attribute_loglik()
        total_loglik = (1 - gamma) * structural_loglik + gamma * attribute_loglik

        return structural_loglik, attribute_loglik, total_loglik

    def fit(self, initial_r=10, num_step=20):

        best_loglik = float("-inf")
        best_param = None
        r_count = 0

        for i in range(0, initial_r):
            if self.random_state == None:
                self.initialize_params(None)
            else:
                self.initialize_params(self.random_state + r_count)
            r_count += 1

            while not self.check_initial_parameters():
                self.initialize_params(self.random_state + r_count)
                r_count += 1

            for j in range(0, num_step):
                self.update_u(self.gamma)
                self.update_w()
                self.update_beta()

            L = self.calc_loglik(self.gamma)[2]

            if L > best_loglik:
                best_loglik = L
                best_param = (self.U, self.W, self.Beta)

        return best_loglik, best_param

