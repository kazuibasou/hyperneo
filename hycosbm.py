import math
import random
from scipy.sparse import csr_matrix
import numpy as np
import hypergraph
from itertools import combinations


class HyCoSBM_old:
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
    gammas = []
    D = 0
    normalize_U = False
    normalize_loglik = False
    test = False
    random_state = None

    def __init__(self, G: hypergraph.HyperGraph, K, gammas, random_state=None, test=False):
        self.E = np.array([set(G.E[m]) for m in range(0, G.M)], dtype=set)
        self.A = np.array([int(G.A[m]) for m in range(0, G.M)], dtype=int)
        self.X = np.zeros((G.N, G.Z), dtype=int)
        self.N = G.N
        self.M = G.M
        self.B = np.zeros((G.N, G.M), dtype=int)
        self.K = K
        self.Z = G.Z
        self.gammas = gammas
        self.U = np.zeros((self.N, self.K), dtype=float)
        self.W = np.zeros((self.K, self.K), dtype=float)
        self.poi_lambda = np.zeros(self.M, dtype=float)
        self.S = np.zeros((self.M, self.K), dtype=float)
        self.Beta = np.zeros((self.K, self.Z), dtype=float)
        self.C_for_U = np.zeros((G.N, K), dtype=float)
        self.C_for_W = np.zeros((K, K), dtype=float)
        self.D = max([len(e) for e in G.E])
        self.test = test
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

        if self.test:
            for i in range(0, self.N):
                for k in range(0, self.K):
                    if self.U[i][k] < 0 or self.U[i][k] > 1:
                        print("Error: u_ik is not in [0, 1].")
                        print(self.U[i][k])
                        exit()

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

        if self.test:
            for m in range(0, self.M):
                lambda_e = 0.0
                e = self.E[m]
                for i in e:
                    for j in e:
                        if not i < j:
                            continue
                        lambda_e += sum([self.U[i][k] * self.U[j][q] * self.W[k][q] for k in range(0, self.K) for q in range(0, self.K)])
                lambda_e = max(lambda_e, self.EPS)
                if not math.isclose(lambda_e, self.poi_lambda[m], rel_tol=self.tol):
                    print("Error: lambda_e is not correctly calculated.")
                    print(lambda_e, self.poi_lambda[m])
                    exit()

        # Matrix Beta
        self.Beta = rng.random((self.K, self.Z))
        Beta_sum = self.Beta.sum(axis=0)
        Beta_sum = np.where(Beta_sum < self.EPS, self.EPS, Beta_sum)
        for z in range(0, self.Z):
            self.Beta[:, z] /= Beta_sum[z]

        if self.test:
            Beta_sum = self.Beta.sum(axis=0)
            for z in range(0, self.Z):
                if not math.isclose(Beta_sum[z], 1.0, rel_tol=self.tol):
                    print("Error: Beta is not correctly normalized.")
                    print(Beta_sum[z], 1.0)
                    exit()

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
        # First coefficient a_ik
        U_sum = self.U.sum(axis=0)
        first_coeff = (1 - gamma) * (self.C_for_U * (np.matmul(self.W, U_sum)[None, :] - np.matmul(self.U, self.W)))

        if self.test:
            for i in range(0, self.N):
                for k in range(0, self.K):
                    u_ik = 0.0
                    for q in range(0, self.K):
                        u_ik += self.W[k][q] * sum([self.U[j][q] for j in range(0, self.N) if j != i])
                    u_ik *= self.C * (1 - gamma)
                    u_ik = max(u_ik, self.EPS)
                    if not math.isclose(first_coeff[i][k], u_ik, rel_tol=self.tol):
                        print("Error: first coefficient a_ik is not correctly calculated.")
                        print(first_coeff[i][k], u_ik)
                        exit()

        # Second coefficient b_ik
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

        second_coeff = 2.0 * (first_term + second_term)

        if self.test:
            for i in range(0, self.N):
                for k in range(0, self.K):
                    first_u_ik = 0
                    for m in range(0, self.M):
                        e = self.E[m]
                        if i not in e:
                            continue
                        first_u_ik += self.A[m] * sum([float(self.U[i][k] * self.U[j][q] * self.W[k][q]) / self.poi_lambda[m] for j in e if j != i for q in range(0, self.K)])
                    second_u_ik = 0
                    for z in range(0, self.Z):
                        second_u_ik += self.X[i][z] * float(self.U[i][k] * self.Beta[k][z]) / max(sum([self.U[i][k_] * self.Beta[k_][z] for k_ in range(0, self.K)]), self.EPS)
                    if not math.isclose(second_coeff[i][k], (1 - gamma) * first_u_ik + 2 * gamma * second_u_ik, rel_tol=self.tol):
                        print("Error: second coefficient b_ik is not correctly calculated.")
                        print(second_coeff[i][k], (1 - gamma) * first_u_ik + 2 * gamma * second_u_ik)
                        exit()

        # Third coefficient
        one_minus_U, one_minus_X = 1 - self.U, 1 - self.X
        divider = one_minus_U @ self.Beta
        divider = np.where(divider < self.EPS, self.EPS, divider)
        X_ = one_minus_X / divider
        third_coeff = gamma * (one_minus_U * (X_ @ self.Beta.T))

        if self.test:
            for i in range(0, self.N):
                for k in range(0, self.K):
                    c_ik = 0
                    for z in range(0, self.Z):
                        c_ik += (1 - self.X[i][z]) * float((1 - self.U[i][k]) * self.Beta[k][z]) / max(sum([(1 - self.U[i][k_]) * self.Beta[k_][z] for k_ in range(0, self.K)]), self.EPS)
                    c_ik = gamma * c_ik
                    if not math.isclose(third_coeff[i][k], c_ik, rel_tol=self.tol):
                        print("Error: third coefficient c_ik is not correctly calculated.")
                        print(third_coeff[i][k], c_ik)
                        exit()

        a_ = first_coeff
        b_ = first_coeff + second_coeff + third_coeff
        c_ = second_coeff
        Delta = (b_ * b_) - (4 * a_ * c_)
        self.U = (b_ - np.sqrt(Delta)) / (2 * a_)

        self.U = np.where(self.U < self.EPS, self.EPS, self.U)
        self.U = np.where(self.U > 1.0 - self.EPS, 1.0 - self.EPS, self.U)

        if self.test:
            for i in range(0, self.N):
                for k in range(0, self.K):
                    if self.U[i][k] < 0 or self.U[i][k] > 1:
                        print("Error: u_ik is not in [0, 1].")
                        print(self.U[i][k])
                        exit()

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

        if self.test:
            for k in range(0, self.K):
                for q in range(0, self.K):
                    w_kq = 0
                    for m in range(0, self.M):
                        e = self.E[m]
                        w_kq += self.A[m] * sum([float(self.U[i][k] * self.U[j][q] * self.W[k][q]) / self.poi_lambda[m] for i in e for j in e if i != j])
                    if not math.isclose(w_kq, num[k][q], rel_tol=self.tol):
                        print("Error: Numerator of W_kq is not correctly calculated.")
                        print(w_kq, num[k][q])
                        exit()

        # Denominator
        u_sum = self.U.sum(axis=0)
        den = (np.outer(u_sum, u_sum) - np.matmul(self.U.T, self.U)) * self.C_for_W
        den = np.where(den < self.EPS, self.EPS, den)

        if self.test:
            for k in range(0, self.K):
                for q in range(0, self.K):
                    w_kq = self.C * sum([self.U[i][k] * self.U[j][q] for i in range(0, self.N) for j in range(0, self.N) if i != j])
                    w_kq = max(w_kq, self.EPS)
                    if not math.isclose(w_kq, den[k][q], rel_tol=self.tol):
                        print("Error: Denominator of W_kq is not correctly calculated.")
                        print(w_kq, den[k][q])
                        exit()

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
        first_num = self.Beta * (self.U.T @ X_)

        one_minus_U, one_minus_X = 1 - self.U, 1 - self.X
        divider = one_minus_U @ self.Beta
        divider = np.where(divider < self.EPS, self.EPS, divider)
        X_ = one_minus_X / divider
        second_num = self.Beta * (one_minus_U.T @ X_)

        num = first_num + second_num

        if self.test:
            for k in range(0, self.K):
                for z in range(0, self.Z):
                    beta_kz = 0
                    for i in range(0, self.N):
                        h_izk = float(self.U[i][k] * self.Beta[k][z]) / max(sum([self.U[i][k_] * self.Beta[k_][z] for k_ in range(0, self.K)]), self.EPS)
                        h_dash_izk = float((1 - self.U[i][k]) * self.Beta[k][z]) / max(sum([(1 - self.U[i][k_]) * self.Beta[k_][z] for k_ in range(0, self.K)]), self.EPS)
                        beta_kz += self.X[i][z] * h_izk + (1 - self.X[i][z]) * h_dash_izk

                    if not math.isclose(num[k][z], beta_kz, rel_tol=self.tol):
                        print("Error: Beta_kz is not correctly calculated.")
                        print(num[k][z], beta_kz)
                        exit()

        # Denominator
        den = np.zeros((self.K, self.Z), dtype=float)
        num_sum = num.sum(axis=0)
        den[:, :] = num_sum[None, :]
        den = np.where(den < self.EPS, self.EPS, den)

        # Update Beta
        self.Beta = num / den

        if self.test:
            Beta_sum = self.Beta.sum(axis=0)
            for z in range(0, self.Z):
                if not math.isclose(Beta_sum[z], 1.0, rel_tol=self.tol):
                    print("Error: Beta is not correctly normalized.")
                    print(Beta_sum[z], 1.0)
                    exit()

        return

    def calc_structural_loglik(self):
        U_sum = self.U.sum(axis=0)
        first_addend = self.C * 0.5 * (((U_sum @ self.W) * U_sum).sum(axis=-1) - ((self.U @ self.W) * self.U).sum())
        second_addend = np.dot(self.A, np.log(self.poi_lambda))

        if self.test:
            first_term = self.C * sum([self.U[i][k] * self.U[j][q] * self.W[k][q] for i in range(0, self.N) for j in range(0, self.N) if i < j for k in range(0, self.K) for q in range(0, self.K)])
            if not math.isclose(first_term, first_addend, rel_tol=self.tol):
                print("Error: first term in the log-likelihood is not correct.")
                print(first_addend, first_term)
                exit()

            second_term = 0
            for m in range(0, self.M):
                e = self.E[m]
                second_term += self.A[m] * math.log(max(self.EPS, sum([self.U[i][k] * self.U[j][q] * self.W[k][q] for i in e for j in e if i < j for k in range(0, self.K) for q in range(0, self.K)])))
            if not math.isclose(second_term, second_addend, rel_tol=self.tol):
                print("Error: second term in the log-likelihood is not correct.")
                print(second_term, second_addend)
                exit()

        return (-1) * first_addend + second_addend

    def calc_attribute_loglik(self):
        # First term
        lik = self.U @ self.Beta
        lik = np.where(lik < self.EPS, self.EPS, lik)
        log_term = np.log(lik)
        first_term = (self.X * log_term).sum()

        # Second term
        one_minus_U, one_minus_X = 1 - self.U, 1 - self.X
        lik = one_minus_U @ self.Beta
        lik = np.where(lik < self.EPS, self.EPS, lik)
        log_term = np.log(lik)
        second_term = (one_minus_X * log_term).sum()

        loglik = first_term + second_term

        if self.test:
            loglik_ = 0
            for i in range(0, self.N):
                for z in range(0, self.Z):
                    first_term = self.X[i][z] * math.log(sum([self.U[i][k] * self.Beta[k][z] for k in range(0, self.K)]))
                    second_term = (1 - self.X[i][z]) * math.log(sum([(1 - self.U[i][k]) * self.Beta[k][z] for k in range(0, self.K)]))
                    loglik_ += first_term + second_term
            if not math.isclose(loglik, loglik_, rel_tol=self.tol):
                print("Error: log-likelihood for attribute dimension is not correct.")
                print((self.X * log_term).sum(), loglik_)
                exit()

        return loglik

    def calc_loglik(self, gamma):

        return (1 - gamma) * self.calc_structural_loglik() + gamma * self.calc_attribute_loglik()

    def fit_for_given_gamma(self, params):
        (gamma, initial_r, num_step) = params

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
                self.update_u(gamma)
                self.update_w()
                self.update_beta()

            L = self.calc_loglik(gamma)
            if L > best_loglik:
                best_loglik = L
                best_param = (self.U, self.W, self.Beta)

        return best_loglik, best_param

    def fit(self, initial_r=10, num_step=20):
        if len(self.gammas) < 1:
            print("Error: Number of hyperparameters gamma should be positive.")
            exit()

        inffered_U, inffered_W = {}, {}
        for g in range(0, len(self.gammas)):
            gamma = self.gammas[g]
            best_loglik, best_param = self.fit_for_given_gamma((gamma, initial_r, num_step))
            inffered_U[g] = best_param[0]
            inffered_W[g] = best_param[1]

        return inffered_U, inffered_W

class HyCoSBM:
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
        self.X = np.zeros((self.N, self.Z), dtype=int)
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
        # First coefficient a_ik
        U_sum = self.U.sum(axis=0)
        first_coeff = (1 - gamma) * (self.C_for_U * (np.matmul(self.W, U_sum)[None, :] - np.matmul(self.U, self.W)))

        # Second coefficient b_ik
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

        second_coeff = 2.0 * (first_term + second_term)

        # Third coefficient
        one_minus_U, one_minus_X = 1 - self.U, 1 - self.X
        divider = one_minus_U @ self.Beta
        divider = np.where(divider < self.EPS, self.EPS, divider)
        X_ = one_minus_X / divider
        third_coeff = gamma * (one_minus_U * (X_ @ self.Beta.T))

        a_ = first_coeff
        b_ = first_coeff + second_coeff + third_coeff
        c_ = second_coeff
        Delta = (b_ * b_) - (4 * a_ * c_)
        self.U = (b_ - np.sqrt(Delta)) / (2 * a_)

        self.U = np.where(self.U < self.EPS, self.EPS, self.U)
        self.U = np.where(self.U > 1.0 - self.EPS, 1.0 - self.EPS, self.U)

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
        first_num = self.Beta * (self.U.T @ X_)

        one_minus_U, one_minus_X = 1 - self.U, 1 - self.X
        divider = one_minus_U @ self.Beta
        divider = np.where(divider < self.EPS, self.EPS, divider)
        X_ = one_minus_X / divider
        second_num = self.Beta * (one_minus_U.T @ X_)

        num = first_num + second_num

        # Denominator
        den = np.zeros((self.K, self.Z), dtype=float)
        num_sum = num.sum(axis=0)
        den[:, :] = num_sum[None, :]
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
        # First term
        lik = self.U @ self.Beta
        lik = np.where(lik < self.EPS, self.EPS, lik)
        log_term = np.log(lik)
        first_term = (self.X * log_term).sum()

        # Second term
        one_minus_U, one_minus_X = 1 - self.U, 1 - self.X
        lik = one_minus_U @ self.Beta
        lik = np.where(lik < self.EPS, self.EPS, lik)
        log_term = np.log(lik)
        second_term = (one_minus_X * log_term).sum()

        loglik = first_term + second_term

        return loglik

    def calc_loglik(self, gamma):
        structural_loglik = self.calc_structural_loglik()
        attribute_loglik = self.calc_attribute_loglik()
        total_loglik = (1 - gamma) * structural_loglik + gamma * attribute_loglik

        return structural_loglik, attribute_loglik, total_loglik

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
                self.update_u(self.gamma)
                self.update_w()
                self.update_beta()

                L = self.calc_loglik(self.gamma)[2]
                if j > 0:
                    loglik_conv = float(math.fabs(L - pre_loglik)) / math.fabs(pre_loglik) < tol
                pre_loglik = L
                j += 1

            L = self.calc_loglik(self.gamma)[2]

            if L > best_loglik:
                best_loglik = L
                best_param = (self.U, self.W, self.Beta)

            num_step_lst.append(j)

        print("Numbers of steps", num_step_lst)

        return best_loglik, best_param

class HyperParamTuning:
    K_lst = []
    gamma_lst = []

    def __init__(self, G):
        #max_K = max(G.Z, 2)
        max_K = 15
        self.K_lst = list(range(2, max_K + 1))
        self.gamma_lst = [0.1 * i for i in range(1, 10)]

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

        auc_score = np.zeros((len(self.K_lst), len(self.gamma_lst), num_runs), dtype=float)

        for k in range(0, len(self.K_lst)):
            for g in range(0, len(self.gamma_lst)):
                K, gamma = self.K_lst[k], self.gamma_lst[g]

                for r in range(0, num_runs):
                    G_train, G_test = self.construct_train_and_test_sets(G)

                    model = HyCoSBM(G_train, K, gamma)
                    best_loglik, (U, W, Beta) = model.fit()

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
                    auc_score[k][g][r] = auc

                print(K, gamma, np.mean(auc_score[k][g]), np.std(auc_score[k][g]))

        return auc_score

    def run(self, G: hypergraph.HyperGraph):
        auc_score = self.grid_search(G)

        best_auc, best_hyperparm = (-1, -1), (-1, -1)
        for k in range(0, auc_score.shape[0]):
            for g in range(0, auc_score.shape[1]):
                if np.mean(auc_score[k][g]) > best_auc[0]:
                    best_auc = (np.mean(auc_score[k][g]), np.std(auc_score[k][g]))
                    best_hyperparm = (self.K_lst[k], self.gamma_lst[g])

        return best_auc, best_hyperparm, auc_score

