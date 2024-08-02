import numpy as np
from scipy.io import loadmat
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from metric_learn import MLKR

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

def add_gaussian_noise(X, noise_level):
    noise = np.random.normal(0, noise_level, X.shape)
    return X + noise

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

def cross_validation_with_noise(X, y, noise_level):
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    accuracies = []

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        X_train_noisy = add_gaussian_noise(X_train, noise_level)
        X_test_noisy = add_gaussian_noise(X_test, noise_level)
        model = BMLFSP_MLKR()
        model.fit(X_train_noisy, y_train)
        y_pred = model.predict(X_train_noisy, y_train, X_test_noisy)
        acc = accuracy_score(y_test, y_pred)
        accuracies.append(acc)

    return np.mean(accuracies)

def main():
    X, y = load_data()
    noise_levels = [0.2, 0.25]
    for noise_level in noise_levels:
        average_accuracy = cross_validation_with_noise(X, y, noise_level)
        print(f"添加{int(noise_level*100)}%高斯噪声后的十折交叉验证的平均准确率: {average_accuracy:.2f}%")

if __name__ == "__main__":
    main()
