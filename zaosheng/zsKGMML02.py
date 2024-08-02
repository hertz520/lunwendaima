import numpy as np
from scipy.io import loadmat
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from metric_learn import LSML_Supervised

# 加载数据集
def load_data():
    mat_data = loadmat('!cancer.mat')
    mu = np.array(mat_data['mu'], dtype='float')
    zi = np.array(mat_data['zi'], dtype='float')
    pos_label = np.ones(len(mu))
    neg_label = np.zeros(len(zi))
    X = np.concatenate((mu, zi))
    y = np.concatenate((pos_label, neg_label))
    return X, y

def standardize_data(X_train, X_test):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test

def apply_lsml(X_train, y_train):
    lsml = LSML_Supervised(max_iter=100)
    lsml.fit(X_train, y_train)
    return lsml

def transform_data(lsml, X_train, X_test):
    X_train_lsml = lsml.transform(X_train)
    X_test_lsml = lsml.transform(X_test)
    return X_train_lsml, X_test_lsml

def train_and_evaluate_svm(X_train_lsml, y_train, X_test_lsml, y_test):
    clf = SVC(kernel='linear')
    clf.fit(X_train_lsml, y_train)
    y_pred = clf.predict(X_test_lsml)
    return accuracy_score(y_test, y_pred)

def add_gaussian_noise(X, noise_level):
    noise = np.random.normal(0, noise_level, X.shape)
    return X + noise

def add_noise_factor(X, noise_factor):
    perturbations = np.random.normal(0, noise_factor, X.shape)
    return X + perturbations

def perform_cross_validation_with_noise_factor(X, y, noise_level, noise_factors):
    kf = KFold(n_splits=10, shuffle=True, random_state=42)

    for noise_factor in noise_factors:
        accuracies = []

        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            X_train_noisy = add_gaussian_noise(X_train, noise_level)
            X_test_noisy = add_gaussian_noise(X_test, noise_level)
            X_train_noisy = add_noise_factor(X_train_noisy, noise_factor)
            X_test_noisy = add_noise_factor(X_test_noisy, noise_factor)
            X_train_noisy, X_test_noisy = standardize_data(X_train_noisy, X_test_noisy)
            lsml = apply_lsml(X_train_noisy, y_train)
            X_train_lsml, X_test_lsml = transform_data(lsml, X_train_noisy, X_test_noisy)
            accuracy = train_and_evaluate_svm(X_train_lsml, y_train, X_test_lsml, y_test)
            accuracies.append(accuracy)

        print(f"添加20%高斯噪声并设置噪声因子ρ={noise_factor}后的十折交叉验证的平均准确率: {np.mean(accuracies):.4f}")

def main():
    X, y = load_data()
    noise_level = 0.2
    noise_factors = [0.5, 1.0, 1.5, 2.0]
    perform_cross_validation_with_noise_factor(X, y, noise_level, noise_factors)

if __name__ == "__main__":
    main()
