
import numpy as np
import matplotlib.pyplot as plt


class LeastSquares:

    def __init__(self):
        self.iterations = 30
        self.max_samples = 100
        self.expected_theta = 5
        self.sigma = 10

    def ls(self,z,H):
        estimation=np.linalg.inv(H.T.dot(H)).dot(H.T).dot(z)
        est_error = self.expected_theta - estimation
        cost=est_error.T.dot(est_error)
        return estimation,cost[0,0]

    def I_rls(self,z_k1,theta_k,h_k1,W_k1,P_k):
        P_k1_inv = np.linalg.inv(P_k) + h_k1.dot(W_k1).dot(h_k1.T)
        P_k1=np.linalg.inv(P_k1_inv)
        k_k1=P_k1.dot(h_k1).dot(W_k1)
        theta_k1=theta_k+k_k1.dot((z_k1-h_k1.T.dot(theta_k)))

        est_error = self.expected_theta - theta_k1
        cost=est_error.T.dot(est_error)
        return theta_k1,P_k1,cost[0,0]

    def Cov_rls(self,z_k1,theta_k,h_k1,W_k1,P_k):
        k_k1=(P_k.dot(h_k1)).dot(np.linalg.inv(((h_k1.T).dot(P_k)).dot(h_k1)+np.linalg.inv(W_k1)))
        P_k1=((np.eye(P_k.shape[0])-k_k1.dot(h_k1.T))).dot(P_k)
        theta_k1=theta_k+k_k1.dot((z_k1-h_k1.T.dot(theta_k)))

        est_error = self.expected_theta - theta_k1
        cost=est_error.T.dot(est_error)
        return theta_k1,P_k1,cost[0,0]

    def ComputeExpectedCostValues(self,H_all):
        exp_cost_vals=[]
        for i in range(self.max_samples):
            H = H_all[0:i + 1, :]
            expected_cost = self.sigma * self.sigma / (H.T.dot(H)[0, 0])
            exp_cost_vals.append(expected_cost)
        plt.plot(exp_cost_vals, linewidth=2, color="blue", label="exp_cost")
        return exp_cost_vals

    def ComputeExampleLS(self, H_all):
        errors = np.zeros((self.iterations, self.max_samples))
        for j in range(self.iterations):
            cost_vals = []
            v_all = np.random.normal(0, self.sigma, [self.max_samples + 5, 1])
            z_all = H_all * self.expected_theta + v_all
            z = z_all[0:1, :]
            H = H_all[0:1, :]
            est, cost = self.ls(z, H)
            cost_vals.append(cost)
            for i in range(self.max_samples):
                z = z_all[0:i + 2, :]
                H = H_all[0:i + 2, :]
                est, cost = self.ls(z, H)
                errors[j, i] = cost
                cost_vals.append(cost)

        cost_av = []
        for j in errors.T:
            cost_av.append(j.mean())
        return cost_av

    def ComputeExampleInfoRLS(self,H_all):
        errors = np.zeros((self.iterations, self.max_samples))
        for j in range(self.iterations):
            cost_vals = []
            v_all = np.random.normal(0, self.sigma, [self.max_samples + 5, 1])
            z_all = H_all * self.expected_theta + v_all

            z_k1 = z_all[0:1, :]
            h_k1 = H_all[0:1, :]
            theta_k = np.zeros((1, 1))
            P_k = np.eye(1)
            W_k1 = np.eye(h_k1.shape[0])

            theta_k1, P_k1, cost = self.I_rls(z_k1, theta_k, h_k1, W_k1, P_k)
            cost_vals.append(cost)
            for i in range(self.max_samples):
                z_k1 = z_all[i + 1:i + 2, :]
                h_k1 = H_all[i + 1:i + 2, :]
                theta_k = theta_k1.copy()
                P_k = P_k1.copy()
                W_k1 = np.eye(h_k1.shape[0])
                theta_k1, P_k1, cost = self.I_rls(z_k1, theta_k, h_k1, W_k1, P_k)
                errors[j, i] = cost
                cost_vals.append(cost)
            if j==0:
                plt.plot(cost_vals, linewidth=1,label="cost", color="grey")
            else:
                plt.plot(cost_vals, linewidth=1, color="grey")

    def ComputeExampleCovRLS(self,H_all):
        errors = np.zeros((self.iterations, self.max_samples))
        for j in range(self.iterations):
            cost_vals = []
            v_all = np.random.normal(0, self.sigma, [self.max_samples + 5, 1])
            z_all = H_all * self.expected_theta + v_all

            z_k1 = z_all[0:1, :]
            h_k1 = H_all[0:1, :]
            theta_k = np.zeros((1, 1))
            P_k = np.eye(1)
            W_k1 = np.eye(h_k1.shape[0])

            theta_k1, P_k1, cost = self.Cov_rls(z_k1, theta_k, h_k1, W_k1, P_k)
            cost_vals.append(cost)
            for i in range(self.max_samples):
                z_k1 = z_all[i + 1:i + 2, :]
                h_k1 = H_all[i + 1:i + 2, :]
                theta_k = theta_k1.copy()
                P_k = P_k1.copy()
                W_k1 = np.eye(h_k1.shape[0])
                theta_k1, P_k1, cost = self.Cov_rls(z_k1, theta_k, h_k1, W_k1, P_k)
                errors[j, i] = cost
                cost_vals.append(cost)
            if j==0:
                plt.plot(cost_vals, linewidth=1,label="cost", color="grey")
            else:
                plt.plot(cost_vals, linewidth=1, color="grey")