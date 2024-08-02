import numpy as np
from scipy.io import loadmat
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from metric_learn import LSML_Supervised
from sklearn.neighbors import KNeighborsClassifier

# 加载数据
mat_data = loadmat('!cancer.mat')
mu = np.array(mat_data['mu'], dtype='float')
zi = np.array(mat_data['zi'], dtype='float')
pos_label = np.ones(len(mu))
neg_label = np.zeros(len(zi))
X = np.concatenate((mu, zi))
y = np.concatenate((pos_label, neg_label))

def add_gaussian_noise(X, noise_level):
    noise = np.random.normal(0, noise_level, X.shape)
    return X + noise

def optimize_metric_learning(X_train, y_train):
    lsml = LSML_Supervised(num_constraints=200)
    lsml.fit(X_train, y_train)
    return lsml

def cross_validation_with_noise(X, y, noise_level):
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    accuracies = []

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        X_train_noisy = add_gaussian_noise(X_train, noise_level)
        X_test_noisy = add_gaussian_noise(X_test, noise_level)
        lsml = optimize_metric_learning(X_train_noisy, y_train)
        X_train_transformed = lsml.transform(X_train_noisy)
        X_test_transformed = lsml.transform(X_test_noisy)
        knn = KNeighborsClassifier(n_neighbors=3)
        knn.fit(X_train_transformed, y_train)
        y_pred = knn.predict(X_test_transformed)
        acc = accuracy_score(y_test, y_pred)
        accuracies.append(acc)

    return np.mean(accuracies)

# 主函数
def main():
    noise_levels = [0.2, 0.25]
    for noise_level in noise_levels:
        average_accuracy = cross_validation_with_noise(X, y, noise_level)
        print(f"添加{int(noise_level*100)}%高斯噪声后的十折交叉验证的平均分类准确率: {average_accuracy:.4f}")

if __name__ == "__main__":
    main()
