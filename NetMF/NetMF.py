import numpy as np
import scipy.sparse as sp
import networkx as nx
import torch

class NetMF(torch.nn.Module):

    # By Yifei Jin

    def __init__(self, dimension, window_size, rank, negative, is_large=False):
        self.dimension = dimension
        self.window_size = window_size
        self.rank = rank
        self.negative = negative
        self.is_large = is_large

    def train(self, G, Graph_input=False):
        # G :nx.graph: For directed graphs, entry i,j in G corresponds to an edge from i to j.
        """
              G
              |
             (0)             G_dense = np.array([[0, 2, 1],[2, 0, 0],[1, 0, 0]])
            /   \
           1     2
          /       \
        (2)       (1)
        """
        if Graph_input:
            A = sp.csr_matrix(nx.adjacency_matrix(G))
        else:# For test the input of the .mat in the publicationx
            A = G
        if not self.is_large:
            print("Running NetMF for a small window size...")
            deepwalk_mat = self._compute_deepwalk_mat(A, window=self.window_size, b=self.negative)
        else:
            print("Running NetMF for a large window size...")
            vol = float(A.sum())
            evals, D_rt_invU = self._approximate_normalized_laplacian(A, rank=self.rank, which="LA")
            deepwalk_mat = self._approximate_deepwalk_mat(
                evals, D_rt_invU, window=self.window_size, vol=vol, b=self.negative
            )
        # factorize deepwalk matrix with SVD logM′ = Ud Σd Vd⊤
        u, s, _ = sp.linalg.svds(deepwalk_mat, self.dimension)
        self.embeddings = sp.diags(np.sqrt(s)).dot(u.T).T
        return self.embeddings

    def _compute_deepwalk_mat(self, A, window, b):
        # directly compute deepwalk matrix
        # number of nodes
        n = A.shape[0]
        # vol(G)
        vol = float(A.sum())

        # sp.csgraph.laplacian: Take the input of Adjancy matrix, output the L matrix
        L, d_rt = sp.csgraph.laplacian(A, normed=True, return_diag=True)

        # X = D^{-1/2} A D^{-1/2}
        X = sp.identity(n) - L

        S = np.zeros_like(X)
        X_power = sp.identity(n)

        # Compute P^1, · · · , P^T
        for i in range(window):
            print("Compute matrix %d-th power", i + 1)
            X_power = X_power.dot(X)
            # \sum P^r
            S += X_power

        S *= vol / window / b
        # D^-1, The length-N diagonal of the Laplacian matrix. For the normalized Laplacian, this is the array of square roots of vertex degrees or 1 if the degree is zero.
        D_rt_inv = sp.diags(d_rt ** -1)

        M = D_rt_inv.dot(D_rt_inv.dot(S).T).todense()
        # M′ = max(M, 1);
        M[M <= 1] = 1
        # logM′
        Y = np.log(M)
        return sp.csr_matrix(Y)

    def _approximate_normalized_laplacian(self, A, rank, which="LA"):
        # perform eigen-decomposition of D^{-1/2} A D^{-1/2} and keep top rank eigenpairs
        n = A.shape[0]
        L, d_rt = sp.csgraph.laplacian(A, normed=True, return_diag=True)
        # X = D^{-1/2} W D^{-1/2}
        X = sp.identity(n) - L
        print("Eigen decomposition...")
        evals, evecs = sp.linalg.eigsh(X, rank, which=which)
        print("Maximum eigenvalue %f, minimum eigenvalue %f", np.max(evals), np.min(evals))
        print("Computing D^{-1/2}U..")
        D_rt_inv = sp.diags(d_rt ** -1)
        D_rt_invU = D_rt_inv.dot(evecs)
        return evals, D_rt_invU

    def _deepwalk_filter(self, evals, window):
        for i in range(len(evals)):
            x = evals[i]
            evals[i] = 1.0 if x >= 1 else x * (1 - x ** window) / (1 - x) / window
        evals = np.maximum(evals, 0)
        print(
            "After filtering, max eigenvalue=%f, min eigenvalue=%f",
            np.max(evals),
            np.min(evals),
        )
        return evals

    def _approximate_deepwalk_mat(self, evals, D_rt_invU, window, vol, b):
        # approximate deepwalk matrix
        evals = self._deepwalk_filter(evals, window=window)
        X = sp.diags(np.sqrt(evals)).dot(D_rt_invU.T).T
        M = X.dot(X.T) * vol / b
        M[M <= 1] = 1
        Y = np.log(M)
        print("Computed DeepWalk matrix with %d non-zero elements", np.count_nonzero(Y))
        return sp.csr_matrix(Y)