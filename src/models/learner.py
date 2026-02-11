from __future__ import division

from multiprocessing import Pool

import numpy as np


def parallelized_rewards(F, R, S, theta, beta):
    return F(R, S, theta, beta)


class Learner(object):
    """
    This class supports various observation models for human demonstrations, as
    well as inference for the posterior of parameters theta and beta.
    """

    def __init__(self, Phi_R, Phi_S, thetas, betas, f_method, s_method, ss=None, kde=None):
        """
        Params:
            Phi_R [array] -- Reward features for all demonstration options.
            Phi_S [array] -- Similarity features for all demonstration options.
            thetas [list] -- Hypothesis space of theta preferences.
            betas [list] -- Possible rationality coefficients.
            f_method [string] -- Reward function parameterization.
                Currently supports: "boltzmann"
            s_method [string] -- Similarity function method.
                Currently supports: "luce", "max", "gaussian", "kde".
            ss [float] -- Variance. Must provide if s_method is "max" or "gaussian".
            kde [kde] -- KernerlDensity object from scipy.
        """
        self.Phi_xibar_R = Phi_R
        self.Phi_xibar_S = Phi_S
        self.thetas = thetas
        self.betas = betas
        self.num_betas = len(self.betas)
        self.num_thetas = len(self.thetas)
        self.prior = np.ones((self.num_betas, self.num_thetas)) / (self.num_betas * self.num_thetas)

        self.f_method = f_method
        self.s_method = s_method
        self.kde = kde
        if self.s_method == "max" or self.s_method == "gaussian":
            assert ss is not None, "Cannot run gaussian inference without specifying variance."
            self.ss = ss

    def inference(self, Phi_xiall_R, Phi_xiall_S):
        """
        Performs inference from given demonstrated features, using initialized model.
        Params:
            Phi_xiall_R [array] -- The reward features for all demonstrations.
            Phi_xiall_S [array] -- The similarity features for all demonstrations.
        """
        # Calculate P(xi | theta, beta)
        P_xi_bt = np.zeros((self.num_betas, self.num_thetas))
        for b, beta in enumerate(self.betas):
            for t, theta in enumerate(self.thetas):
                P_xi_bt[b][t] = self.observation_model(Phi_xiall_R, Phi_xiall_S, theta, beta)
        # Calculate b(theta, beta) -> posterior for theta and beta after demonstration
        P_bt = np.multiply(P_xi_bt, self.prior)
        P_bt = P_bt / sum(sum(P_bt))
        return P_bt, P_xi_bt

    def observation_model(self, Phi_xiall_R, Phi_xiall_S, theta, beta):
        """
        Finds observation model for given demonstrated features, using initialized model.
        Params:
            Phi_xiall_R [array] -- The reward features for all demonstrations.
            Phi_xiall_S [array] -- The similarity features for all demonstrations.
            theta [list] -- The preference parameter.
            beta [float] -- The rationality coefficient.
        """
        F_xiall = self.reward_model(Phi_xiall_R, Phi_xiall_S, theta, beta)
        F_xibar = self.reward_model(self.Phi_xibar_R, self.Phi_xibar_S, theta, beta)
        numerator = sum(F_xiall)
        A_max = max(F_xibar)
        denominator = A_max + np.log(sum(np.exp(F_xibar - A_max)))
        return np.exp(numerator - denominator * len(Phi_xiall_R))

    def F(self, Phi_xi_R, Phi_xi_S, theta, beta):
        # Compute F value for one demonstration
        F_val = 0

        # First the f metric
        if self.f_method == "boltzmann":
            F_val += self.f_boltzmann(Phi_xi_R, theta, beta)
        else:
            raise NotImplementedError

        # Second, the similarity metric
        if self.s_method == "luce":
            F_val -= self.s_luce()
        elif self.s_method == "max":
            F_val -= self.s_max(Phi_xi_S, theta, beta, self.ss)
        elif self.s_method == "gaussian":
            F_val -= self.s_gaussian(Phi_xi_S, self.ss)
        elif self.s_method == "kde":
            F_val -= self.s_kde(Phi_xi_S)
        else:
            raise NotImplementedError
        return F_val

    def reward_model(self, Phi_R, Phi_S, theta, beta):
        return [self.F(R, S, theta, beta) for R, S in zip(Phi_R, Phi_S)]
        # else:
        #     with Pool(processes=num_cores) as pool:
        #         return pool.starmap(parallelized_rewards, [(self.F, R, S, theta, beta) for R, S in zip(Phi_R, Phi_S)])

    def f_boltzmann(self, Phi, theta, beta):
        """
        Compute log scale f using Boltzmann.
        Params:
            Phi_xi [list] -- A feature vectors for one trajectory.
            theta [list] -- The weight vector.
            beta [float] -- The rationality coefficient.
        Returns:
            R [float] -- The f exponent according to Boltzmann.
        """
        return - beta * np.dot(Phi, theta)

    def s_luce(self):
        """
        Explicitly compute the exponent of the luce similarity.
        Luce does not use a similarity metric, so the exponent is 0.
        """
        return 0

    def s_gaussian(self, Phi_xi, ss):
        """
        Compute the similarity exponent using the gaussian kernel similarity rule.
        Params:
            Phi_xi [list] -- A feature vector for one trajectory.
            ss [float] -- The variance of the Gaussian distribution.
        Returns:
            S [float] -- Exponent of similarity metric.
        """
        R_xibar = [- 0.5 / ss * np.linalg.norm(Phi_xi - Phi) ** 2 for Phi in self.Phi_xibar_S]
        A_max = max(R_xibar)
        S = A_max + np.log(sum(np.exp(R_xibar - A_max)))
        return S

    def s_max(self, Phi_xi, theta, beta, ss):
        """
        Compute the similarity exponent using the max rule with Gaussian similarity.
        Params:
            Phi_xi [list] -- A feature vector for one trajectory.
            theta [list] -- The weight vector.
            beta [float] -- The rationality coefficient.
            ss [float] -- The variance of the Gaussian distribution.
        Returns:
            S [float] -- Exponent of similarity metric.
        """
        R_scaled_all = [-beta * np.dot(Phi_p, theta) - 0.5 / ss * np.linalg.norm(Phi_p - Phi_xi) ** 2 for Phi_p in
                        self.Phi_xibar_S]
        A_max = max(R_scaled_all)
        logsum = A_max + np.log(sum(np.exp(R_scaled_all - A_max)))
        S = logsum - A_max
        return S

    def s_kde(self, Phi_xi):
        """
        Compute the similarity exponent using the KDE density.
        Params:
            Phi_xi [list] -- A feature vector for one trajectory.
        Returns:
            S [float] -- Exponent of similarity metric.
        """
        # kde.score_samples returns log(density), which is what we want since
        # the return value of this function will be treated as an exponent
        return self.kde.score_samples(Phi_xi.reshape(1, -1))[0]
