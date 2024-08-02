import numpy as np
from scipy.io import loadmat
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from metric_learn import MLKR

# 加载数据集
mat_data = loadmat('!cancer.mat')
mu = np.array(mat_data['mu'], dtype='float')
zi = np.array(mat_data['zi'], dtype='float')
pos_label = np.ones(len(mu))
neg_label = np.zeros(len(zi))
X = np.concatenate((mu, zi))
y = np.concatenate((pos_label, neg_label))

def initialize_parameters(d):
    M = np.eye(d)
    alpha = 0.1
    beta = 0.1
    gamma1 = 1.0
    gamma2 = 1.0
    return M, alpha, beta, gamma1, gamma2


def mahalanobis_distance(x, y, M):
    diff = x - y
    return np.dot(np.dot(diff.T, M), diff)


def update_weights(T, alpha, q, p):
    u = np.zeros(len(T))
    for i, t in enumerate(T):
        if t < alpha * (1 / q - 1):
            u[i] = 1
        elif t > alpha / q:
            u[i] = 0
        else:
            u[i] = (1 / q - t / alpha) ** (1 / (p - 1))
    return u


def update_metric_matrix(X, S, D, u, M, beta, gamma1, gamma2, eta):
    G = np.zeros_like(M)
    for i in range(len(X)):
        if u[i] > 0:
            sum_S = np.sum([(X[i] - x).reshape(-1, 1).dot((X[i] - x).reshape(1, -1)) * np.exp(
                -gamma1 * mahalanobis_distance(X[i], x, M)) for x in S[i]], axis=0)
            sum_D = np.sum([(X[i] - x).reshape(-1, 1).dot((X[i] - x).reshape(1, -1)) * np.exp(
                -gamma2 * mahalanobis_distance(X[i], x, M)) for x in D[i]], axis=0)
            G += u[i] * (sum_S / len(S[i]) - sum_D / len(D[i]))
    G += 2 * beta * (M - np.eye(M.shape[0]))
    M = M - eta * G
    return np.linalg.multi_dot([M, np.linalg.pinv(M.T), M])


def construct_triplets(X, y, M, k=10):
    S = []
    D = []
    for i in range(len(X)):
        pos_indices = np.where(y == y[i])[0]
        neg_indices = np.where(y != y[i])[0]
        pos_dists = [mahalanobis_distance(X[i], X[j], M) for j in pos_indices]
        neg_dists = [mahalanobis_distance(X[i], X[j], M) for j in neg_indices]
        pos_sorted = np.argsort(pos_dists)[:k]
        neg_sorted = np.argsort(neg_dists)[:k]
        S.append(X[pos_indices[pos_sorted]])
        D.append(X[neg_indices[neg_sorted]])
    return S, D


def self_paced_learning(X, y, alpha, beta, gamma1, gamma2, p, q, eta, epochs):
    M, alpha, beta, gamma1, gamma2 = initialize_parameters(X.shape[1])
    for epoch in range(epochs):
        S, D = construct_triplets(X, y, M)
        T = [np.sum([mahalanobis_distance(X[i], x, M) for x in S[i]]) - np.sum(
            [mahalanobis_distance(X[i], x, M) for x in D[i]]) for i in range(len(X))]
        u = update_weights(T, alpha, q, p)
        M = update_metric_matrix(X, S, D, u, M, beta, gamma1, gamma2, eta)
    return M


class BMLFSP_MLKR:
    def __init__(self):
        self.model = MLKR()

    def fit(self, X, y):
        self.model.fit(X, y)

    def transform(self, X):
        return self.model.transform(X)

    def predict(self, X_train, y_train, X_test):
        X_train_transformed = self.transform(X_train)
        X_test_transformed = self.transform(X_test)
        y_pred = []
        for x in X_test_transformed:
            distances = np.linalg.norm(X_train_transformed - x, axis=1)
            nearest_index = np.argmin(distances)
            y_pred.append(y_train[nearest_index])
        return np.array(y_pred)


kf = KFold(n_splits=10, shuffle=True, random_state=42)
accuracies = []

for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    model = BMLFSP_MLKR()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_train, y_train, X_test)
    acc = accuracy_score(y_test, y_pred)
    accuracies.append(acc)

print("十折交叉验证的平均分类准确率: {:.2f}%".format(np.mean(accuracies) * 100))


def calculate_loss(T, u, alpha):
    loss = 0
    for i in range(len(T)):
        loss += u[i] * T[i] + alpha * (1 / len(T) ** 2) * sum(u ** 2)
    return loss


def bi_level_metric_learning(X, y, alpha, beta, gamma1, gamma2, p, q, eta, epochs):
    M, alpha, beta, gamma1, gamma2 = initialize_parameters(X.shape[1])
    for epoch in range(epochs):
        S, D = construct_triplets(X, y, M)
        T = [np.sum([mahalanobis_distance(X[i], x, M) for x in S[i]]) - np.sum(
            [mahalanobis_distance(X[i], x, M) for x in D[i]]) for i in range(len(X))]
        u = update_weights(T, alpha, q, p)
        M = update_metric_matrix(X, S, D, u, M, beta, gamma1, gamma2, eta)
        loss = calculate_loss(T, u, alpha)
        print(f"Epoch {epoch + 1}, Loss: {loss}")
    return M


if __name__ == "__main__":
    epochs = 10
    alpha = 0.1
    beta = 0.1
    gamma1 = 1.0
    gamma2 = 1.0
    p = 2
    q = 0.5
    eta = 0.01

    print("开始双层度量学习框架的训练...")
    M = bi_level_metric_learning(X, y, alpha, beta, gamma1, gamma2, p, q, eta, epochs)
    print("训练完成！")
