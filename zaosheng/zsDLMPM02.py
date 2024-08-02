import numpy as np
from scipy.io import loadmat
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from scipy.linalg import inv

# 加载数据
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

def smw_inv(H, v):
    m = H.shape[1]
    A_inv = np.eye(m) / v
    temp = inv(np.eye(m) + H.T @ A_inv @ H)
    return A_inv - A_inv @ H @ temp @ H.T @ A_inv

def solve_dual_problem(K, y_train, v=0.5):
    pos_indices = np.where(y_train == 1)[0]
    neg_indices = np.where(y_train == 0)[0]
    l_D = np.mean(K[pos_indices], axis=0)
    l_S = np.mean(K[neg_indices], axis=0)
    H = np.hstack((K[:, pos_indices], K[:, neg_indices]))
    Q_inv = smw_inv(H, v)
    gamma = v * Q_inv @ (l_D - l_S)
    return gamma

def add_gaussian_noise(X, noise_level):
    noise = np.random.normal(0, noise_level, X.shape)
    return X + noise

def run_experiment_with_noise_factor(X, y, noise_level, gamma, v, rho_values):
    kf = KFold(n_splits=10, shuffle=True, random_state=42)

    for rho in rho_values:
        accuracies = []

        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            X_train_noisy = add_gaussian_noise(X_train, noise_level)
            X_test_noisy = add_gaussian_noise(X_test, noise_level)
            train_perturbations = np.random.normal(0, rho, X_train_noisy.shape)
            test_perturbations = np.random.normal(0, rho, X_test_noisy.shape)
            X_train_noisy = X_train_noisy + train_perturbations
            X_test_noisy = X_test_noisy + test_perturbations
            K_train = rbf_kernel(X_train_noisy, X_train_noisy, gamma)
            K_test = rbf_kernel(X_test_noisy, X_train_noisy, gamma)
            gamma_ = solve_dual_problem(K_train, y_train, v)
            decision_values = K_test @ gamma_
            y_pred = np.where(decision_values >= 0, 1, 0)
            acc = accuracy_score(y_test, y_pred)
            accuracies.append(acc)

        print(f'添加20%高斯噪声并设置噪声因子ρ={rho}后的平均分类准确率: {np.mean(accuracies):.4f}')

fixed_gamma = 0.1
fixed_v = 0.5
rho_values = [0.5, 1.0, 1.5, 2.0]
run_experiment_with_noise_factor(X, y, 0.2, fixed_gamma, fixed_v, rho_values)
