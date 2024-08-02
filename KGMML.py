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

def perform_cross_validation(X, y):
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    accuracies = []

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        X_train, X_test = standardize_data(X_train, X_test)
        lsml = apply_lsml(X_train, y_train)
        X_train_lsml, X_test_lsml = transform_data(lsml, X_train, X_test)
        accuracy = train_and_evaluate_svm(X_train_lsml, y_train, X_test_lsml, y_test)
        accuracies.append(accuracy)

    return np.mean(accuracies)

def main():
    X, y = load_data()
    average_accuracy = perform_cross_validation(X, y)
    print(f"十折交叉验证的平均准确率: {average_accuracy:.4f}")

def kernel_function(x1, x2):
    return np.dot(x1, x2.T)

def compute_kernel_matrix(X):
    n_samples = X.shape[0]
    K = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        for j in range(n_samples):
            K[i, j] = kernel_function(X[i], X[j])
    return K
if __name__ == "__main__":
    main()

class KernelGeometricMeanMetricLearning:
    def __init__(self, t=0.5):
        self.t = t

    def fit(self, S, D):
        self.S = S
        self.D = D
        self.M = self.geometric_mean(S, D)

    def geometric_mean(self, S, D):
        S_inv = np.linalg.inv(S)
        D_inv = np.linalg.inv(D)
        return S_inv @ D_inv

    def transform(self, X):
        return X @ self.M

def detailed_theoretical_methods():
    S = np.random.rand(5, 5)
    D = np.random.rand(5, 5)
    kgmml = KernelGeometricMeanMetricLearning()
    kgmml.fit(S, D)
    X_transformed = kgmml.transform(np.random.rand(10, 5))
    return X_transformed

transformed_data = detailed_theoretical_methods()
print("Transformed data using theoretical methods: ", transformed_data)
