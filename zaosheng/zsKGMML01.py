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

def perform_cross_validation(X, y, noise_level):
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    accuracies = []

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        X_train_noisy = add_gaussian_noise(X_train, noise_level)
        X_test_noisy = add_gaussian_noise(X_test, noise_level)
        X_train_noisy, X_test_noisy = standardize_data(X_train_noisy, X_test_noisy)
        lsml = apply_lsml(X_train_noisy, y_train)
        X_train_lsml, X_test_lsml = transform_data(lsml, X_train_noisy, X_test_noisy)
        accuracy = train_and_evaluate_svm(X_train_lsml, y_train, X_test_lsml, y_test)
        accuracies.append(accuracy)

    return np.mean(accuracies)

def main():
    X, y = load_data()
    noise_levels = [0.2, 0.25]
    for noise_level in noise_levels:
        average_accuracy = perform_cross_validation(X, y, noise_level)
        print(f"添加{int(noise_level*100)}%高斯噪声后的十折交叉验证的平均准确率: {average_accuracy:.4f}")


if __name__ == "__main__":
    main()
