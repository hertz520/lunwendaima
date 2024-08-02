import numpy as np
from scipy.io import loadmat
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from metric_learn import SCML_Supervised
from sklearn.mixture import GaussianMixture
from sklearn.svm import SVC
from scipy.linalg import expm

class MDaML_Simulator:
    def __init__(self, n_clusters=3, eta=1.0, lambda1=0.1, lambda2=0.1):
        self.n_clusters = n_clusters
        self.eta = eta
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.metric_learner = SCML_Supervised()
        self.CK = None
        self.W = None

    def fit(self, X, y):
        self.CK, self.W = self.initialize_clusters(X)
        self.metric_learner.fit(X, y)

    def transform(self, X):
        return self.metric_learner.transform(X)

    def initialize_clusters(self, X):
        gmm = GaussianMixture(n_components=self.n_clusters)
        gmm.fit(X)
        CK = gmm.means_
        W = gmm.predict_proba(X)
        return CK, W

    def orthogonal_projection(self, W, grad):
        return W @ (grad + grad.T) @ W

    def retraction(self, W, Z):
        return W @ expm(np.linalg.inv(W) @ Z @ np.linalg.inv(W)) @ W

    def update_CK(self, X):
        for k in range(self.n_clusters):
            num = np.sum([self.W[i, k] ** self.eta * X[i] for i in range(len(X))], axis=0)
            den = np.sum([self.W[i, k] ** self.eta for i in range(len(X))])
            self.CK[k] = num / den

    def update_W(self, X, M):
        for i in range(len(X)):
            Fi = [self.calculate_Fi(X[i], k, M) for k in range(self.n_clusters)]
            self.W[i] = self.calculate_W(Fi)

    def calculate_Fi(self, xi, k, M):
        term1 = np.dot(np.dot((xi - self.CK[k]).T, M), (xi - self.CK[k]))
        term2 = self.lambda1 * np.sum([self.W[j, k] ** self.eta for j in range(len(self.W))])
        return term1 + term2

    def calculate_W(self, Fi):
        W_new = [(1 / f) ** (1 / (self.eta - 1)) for f in Fi]
        return W_new / np.sum(W_new)

    def update_M(self, X, y):
        gradient_M = self.calculate_gradient(X, y)
        M_new = self.metric_learner.metric() - self.lambda2 * gradient_M
        return self.project_to_psd_cone(M_new)

    def calculate_gradient(self, X, y):
        grad1 = np.sum(
            [self.W[i, k] ** self.eta * np.outer((X[i] - self.CK[k]), (X[i] - self.CK[k])) for i in range(len(X)) for k
             in range(self.n_clusters)], axis=0)
        grad2 = np.sum(
            [self.W[i, k] ** self.eta * self.calculate_triplet_loss(X, i, k, y) for i in range(len(X)) for k in
             range(self.n_clusters)], axis=0)
        return grad1 + self.lambda1 * grad2

    def calculate_triplet_loss(self, X, i, k, y):
        loss = 0
        for t in range(len(y)):
            if y[t] == y[i]:
                continue
            dist_pos = np.dot(np.dot((X[i] - X[t]).T, self.metric_learner.metric()), (X[i] - X[t]))
            dist_neg = np.dot(np.dot((X[i] - self.CK[k]).T, self.metric_learner.metric()), (X[i] - self.CK[k]))
            loss += max(0, 1 - dist_neg + dist_pos)
        return loss

    def project_to_psd_cone(self, M):
        eigvals, eigvecs = np.linalg.eigh(M)
        eigvals[eigvals < 0] = 0
        return np.dot(eigvecs, np.dot(np.diag(eigvals), eigvecs.T))


# 加载数据集
mat_data = loadmat('!cancer.mat')
mu = np.array(mat_data['mu'], dtype='float')
zi = np.array(mat_data['zi'], dtype='float')
pos_label = np.ones(len(mu))
neg_label = np.zeros(len(zi))
X = np.concatenate((mu, zi))
y = np.concatenate((pos_label, neg_label))
scaler = StandardScaler()
X = scaler.fit_transform(X)

kf = KFold(n_splits=10, shuffle=True, random_state=42)
accuracies = []

for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    mdaml_simulator = MDaML_Simulator(n_clusters=3, eta=1.0, lambda1=0.1, lambda2=0.1)
    mdaml_simulator.fit(X_train, y_train)

    X_train_transformed = mdaml_simulator.transform(X_train)
    X_test_transformed = mdaml_simulator.transform(X_test)

    clf = SVC()
    clf.fit(X_train_transformed, y_train)
    y_pred = clf.predict(X_test_transformed)

    accuracy = accuracy_score(y_test, y_pred)
    accuracies.append(accuracy)

print(f'十折交叉验证的平均分类准确率: {np.mean(accuracies):.4f}')
