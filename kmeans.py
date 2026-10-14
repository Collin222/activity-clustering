# k-means clustering
# with a gaussian mixture model to model p(x)
# the density of each cluster

import random
import math
import csv
from helpers import inverse, det

random.seed(0)

X = []

with open('data.csv', 'r') as f:
    reader = csv.reader(f)
    headers = next(reader)
    for row in reader:
        features = row[1:]
        features[0] = int(features[0])
        features[1] = float(features[1])

        X.append(features)

class KMeans:
    def __init__(self, k, X):
        self.k = k
        self.X = X
        self.m = len(X)
        self.n = len(X[0])

        print(f"Initializing K-Means with {k} clusters, {self.m} examples, and {self.n} features")

    def l2_norm(self, x): 
        return math.sqrt(sum([a**2 for a in x]))

    def init_mus(self):
        mus = []
        used = set()
        while len(mus) < self.k:
            idx = random.randint(0, self.m - 1)
            if idx in used:
                continue

            used.add(idx)
            mus.append(self.X[idx])
        self.mus = mus

    def add_vectors(self, a, b):
        return [a[i] + b[i] for i in range(len(a))]

    def subtract_vectors(self, a, b):
        return [a[i] - b[i] for i in range(len(a))]
    
    def scale_vector(self, a, c):
        return [a[i] * c for i in range(len(a))]

    def estep(self):
        c = [self.predict(x) for x in self.X]
        return c

    def mstep(self, c):
        mus = []
        for j in range(self.k):
            total = [0] * self.n
            count = 0
            for i in range(self.m):
                if c[i] != j:
                    continue
                count += 1
                total = self.add_vectors(total, self.X[i])
            mu = self.scale_vector(total, 1/count)
            mus.append(mu)
        self.mus = mus
        return mus

    def predict(self, x):
        minj = None
        mindiff = None
        for j in range(self.k):
            diff = self.l2_norm(self.subtract_vectors(x, self.mus[j]))
            if mindiff is None or diff < mindiff:
                minj = j
                mindiff = diff
        if minj is None:
            raise Exception("failed to predict x")
        return minj


    def print_clusters(self):
        counts = [0] * self.k
        for x in self.X:
            j = self.predict(x)
            counts[j] += 1
        for j in range(self.k):
            print(f"cluster {j} : {counts[j]} points")

    def train(self, epochs=100):
        self.init_mus()

        for e in range(epochs):
            if e % 10 == 0 and e != 0:
                print(f"Epoch {e}")
                self.print_clusters()

            c = self.estep()
            self.mstep(c)

        print(f"Finished training k-means after {epochs} epochs")

    def inertia(self):
        total = 0
        for x in self.X:
            j = self.predict(x)
            diff = self.subtract_vectors(x, self.mus[j])
            total += sum(a ** 2 for a in diff)
        return total

class GMM:
    def __init__(self, k, X):
        self.k = k
        self.X = X
        self.m = len(X)
        self.n = len(X[0])

    def train_k_means(self, times=10):
        models = [KMeans(self.k, self.X) for _ in range(times)]
        scores = []
        for m in models:
            m.train()
            scores.append(m.inertia())
        print(f"Scores : {scores}")
        best = scores.index(min(scores))
        self.kmodel = models[best]
        return self.kmodel

    def transpose(self, A):
        return [list(row) for row in zip(*A)]

    def subtract_vectors(self, a, b):
        return [a[i] - b[i] for i in range(len(a))]

    def dot_product(self, a, b):
        return sum([a[i] * b[i] for i in range(len(a))])

    def matrix_multiplication(self, A, B):
        C = []
        BT = self.transpose(B)
        for r in range(len(A)):
            row = []
            for c in range(len(BT)):
                row.append(self.dot_product(A[r], BT[c]))
            C.append(row)
        return C

    def scale_matrix(self, A, c):
        return [[c * a for a in row] for row in A]

    def dimensions(self, A):
        return len(A), len(A[0])

    def extract_scalar(self, A):
        r, c = self.dimensions(A)
        if r != 1 or c != 1:
            raise Exception(f"invalid dimensions {r}x{c} for extract_scalar()")
        return A[0][0]

    def scale_vector(self, x, c):
        return [c * a for a in x]

    def matrix_addition(self, A, B):
        return [[A[r][c] + B[r][c] for c in range(len(A[r]))] for r in range(len(A))]

    def initialized_parameters(self):
        self.mu = [list(m) for m in self.kmodel.mus]
        counts = [0] * self.k
        for x in self.X:
            counts[self.kmodel.predict(x)] += 1
        self.phi = [c / self.m for c in counts]

        self.cov = [None] * self.k
        for j in range(self.k):
            cov = [[0.0] * self.n for _ in range(self.n)]
            for x in self.X:
                if self.kmodel.predict(x) != j:
                    continue
                diff = self.subtract_vectors(x, self.mu[j])
                for r in range(self.n):
                    for c in range(self.n):
                        cov[r][c] += diff[r] * diff[c]

            for r in range(self.n):
                for c in range(self.n):
                    cov[r][c] /= counts[j]
                cov[r][r] += 1e-6

            self.cov[j] = cov

        print("Initialized parameters for GMM")

    def gaussian_pdf(self, x, cov, mu):
        diff = self.subtract_vectors(x, mu)
        difft = self.transpose([diff])
        covinv = inverse(cov)
        inner = self.matrix_multiplication(self.matrix_multiplication([diff], covinv), difft)
        return (1/((math.pi * 2) ** (self.n/2) * (det(cov) ** 0.5))) * math.exp(self.extract_scalar(self.scale_matrix(inner, -1/2)))

    def add_vectors(self, a, b):
        return [a[i] + b[i] for i in range(len(a))]

    def compute_responsibilities(self, x):
        weights = []
        for j in range(self.k):
            p_xz = self.gaussian_pdf(x, self.cov[j], self.mu[j])
            weight = p_xz * self.phi[j]
            weights.append(weight)
        summed = sum(weights)
        return [w / summed for w in weights]

    def estep(self):
        self.weights = [self.compute_responsibilities(x) for x in self.X]
        return self.weights

    def mstep(self):
        for j in range(self.k):
            summed = sum([self.weights[i][j] for i in range(self.m)])
            phij = (1/self.m) * summed
            muj = [0.0] * self.n
            for i in range(self.m):
                add = self.scale_vector(self.X[i], self.weights[i][j] / summed)
                muj = self.add_vectors(muj, add)

            covj = None
            for i in range(self.m):
                diff = [self.subtract_vectors(self.X[i], muj)]
                difft = self.transpose(diff)
                c = self.scale_matrix(self.matrix_multiplication(difft, diff), self.weights[i][j])
                if covj is None:
                    covj = c
                else:
                    covj = self.matrix_addition(covj, c)

            covj = self.scale_matrix(covj, 1/summed)
            # regularize so its invertable
            for r in range(len(covj)):
                covj[r][r] += 1e-6

            self.phi[j] = phij
            self.mu[j] = muj
            self.cov[j] = covj

    def compute_likelihood(self):
        total = 0.0
        for x in self.X:
            p = sum([self.phi[j] * self.gaussian_pdf(x, self.cov[j], self.mu[j]) for j in range(self.k)])
            total += math.log(p)
        return total / self.m

    def train(self, epochs=10):
        self.train_k_means()
        self.initialized_parameters()

        for e in range(epochs):
            likelihood = self.compute_likelihood()
            print(f"Epoch {e} with likelihood : {likelihood}")

            self.estep()
            self.mstep()

        
        print(f"Done training GMM after {epochs} epochs")
        likelihood = self.compute_likelihood()
        print(f"Likelihood : {likelihood}")

    def predict(self, x):
        weights = self.compute_responsibilities(x)
        confidence = max(weights)
        # cluster index, confidence it belongs to j cluster
        return weights.index(confidence), confidence

model = GMM(5, X)
model.train()
