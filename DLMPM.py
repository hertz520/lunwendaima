import numpy as np
from scipy.io import loadmat
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from scipy.linalg import inv

mat_data = loadmat('!cancer.mat')
mu = np.array(mat_data['mu'], dtype='float')
zi = np.array(mat_data['zi'], dtype='float')
pos_label = np.ones(len(mu))
neg_label = np.zeros(len(zi))
X = np.concatenate((mu, zi))
y = np.concatenate((pos_label, neg_label))
scaler = StandardScaler()
X = scaler.fit_transform(X)

def rbf_kernel(X1, X2, gamma=0.1):
    sq_dists = np.sum(X1**2, axis=1).reshape(-1, 1) + np.sum(X2**2, axis=1) - 2 * np.dot(X1, X2.T)
    K = np.exp(-gamma * sq_dists)
    return K

def smw_inv(H, v):# 实现Sherman-Morrison-Woodbury（SMW）恒等式
    m = H.shape[1]
    A_inv = np.eye(m) / v
    temp = inv(np.eye(m) + H.T @ A_inv @ H)
    return A_inv - A_inv @ H @ temp @ H.T @ A_inv

def solve_dual_problem(K, y_train, v=0.5):# 解决对偶问题，求解对偶变量𝛾
    pos_indices = np.where(y_train == 1)[0]
    neg_indices = np.where(y_train == 0)[0]
    l_D = np.mean(K[pos_indices], axis=0)
    l_S = np.mean(K[neg_indices], axis=0)
    H = np.hstack((K[:, pos_indices], K[:, neg_indices]))
    Q_inv = smw_inv(H, v)
    gamma = v * Q_inv @ (l_D - l_S)
    return gamma

# 网格搜索寻找最优参数
def grid_search(X, y):
    kf = KFold(n_splits=10)
    gamma_values = [0.01, 0.1, 1, 10]
    v_values = [0.1, 0.5, 1, 2]
    best_params = None
    best_score = 0

    for gamma in gamma_values:
        for v in v_values:
            accuracies = []
            for train_index, test_index in kf.split(X):
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]
                K_train = rbf_kernel(X_train, X_train, gamma)
                K_test = rbf_kernel(X_test, X_train, gamma)
                gamma_ = solve_dual_problem(K_train, y_train, v)
                decision_values = K_test @ gamma_
                y_pred = np.where(decision_values >= 0, 1, 0)
                acc = accuracy_score(y_test, y_pred)
                accuracies.append(acc)
            mean_accuracy = np.mean(accuracies)
            if mean_accuracy > best_score:
                best_score = mean_accuracy
                best_params = {'gamma': gamma, 'v': v}

    return best_params, best_score

best_params, best_score = grid_search(X, y)
print("最佳参数:", best_params)
print("最佳平均准确率:", best_score)

kf = KFold(n_splits=10)
accuracies = []
all_y_true = []
all_y_pred = []

for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    K_train = rbf_kernel(X_train, X_train, best_params['gamma'])
    K_test = rbf_kernel(X_test, X_train, best_params['gamma'])
    gamma_ = solve_dual_problem(K_train, y_train, best_params['v'])
    decision_values = K_test @ gamma_
    y_pred = np.where(decision_values >= 0, 1, 0)
    accuracies.append(accuracy_score(y_test, y_pred))
    all_y_true.extend(y_test)
    all_y_pred.extend(y_pred)

print("平均准确率:", np.mean(accuracies))
