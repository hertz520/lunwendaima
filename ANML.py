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

def perturbation_loss(X, y, transform_fn):
    perturbations = np.random.randn(*X.shape) * 0.01
    X_perturbed = X + perturbations
    X_transformed = transform_fn(X)
    X_perturbed_transformed = transform_fn(X_perturbed)
    loss = np.sum((X_perturbed_transformed - X_transformed) ** 2)
    return loss

def optimize_metric_learning(X_train, y_train):
    lsml = LSML_Supervised(num_constraints=200)
    lsml.fit(X_train, y_train)
    return lsml

kf = KFold(n_splits=10, shuffle=True, random_state=42)
accuracies = []

for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    lsml = optimize_metric_learning(X_train, y_train)
    X_train_transformed = lsml.transform(X_train)
    X_test_transformed = lsml.transform(X_test)

    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train_transformed, y_train)
    y_pred = knn.predict(X_test_transformed)
    acc = accuracy_score(y_test, y_pred)
    accuracies.append(acc)

print(f'十折交叉验证的平均分类准确率: {np.mean(accuracies):.4f}')

def compute_perturbation_direction(X, transform_fn, r=1.0):
    directions = []
    for xi in X:
        perturbation = np.random.randn(xi.shape[0]) * r
        xi_transformed = transform_fn(xi.reshape(1, -1))
        directions.append(perturbation @ xi_transformed.T)
    return np.array(directions)

def adjust_metric(M, directions, step_size=0.01):
    for direction in directions:
        M += step_size * (direction @ direction.T)
    return M

def train_with_adversarial_examples(X, y, transform_fn, num_iterations=10, step_size=0.01):
    M = np.eye(X.shape[1])
    for _ in range(num_iterations):
        directions = compute_perturbation_direction(X, transform_fn)
        M = adjust_metric(M, directions, step_size)
    return M

def extended_optimize_metric_learning(X_train, y_train):
    lsml = optimize_metric_learning(X_train, y_train)
    M = train_with_adversarial_examples(X_train, y_train, lsml.transform)
    return lsml, M

kf = KFold(n_splits=10, shuffle=True, random_state=42)
accuracies = []

for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    lsml, M = extended_optimize_metric_learning(X_train, y_train)
    X_train_transformed = lsml.transform(X_train)
    X_test_transformed = lsml.transform(X_test)

    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train_transformed, y_train)
    y_pred = knn.predict(X_test_transformed)
    acc = accuracy_score(y_test, y_pred)
    accuracies.append(acc)

print(f'十折交叉验证的平均分类准确率: {np.mean(accuracies):.4f}')

def generate_adversarial_examples(X, transform_fn, epsilon=0.01):
    perturbations = np.random.randn(*X.shape) * epsilon
    X_adv = X + perturbations
    return transform_fn(X_adv)

def evaluate_robustness(X, y, transform_fn, epsilon=0.01):
    X_adv = generate_adversarial_examples(X, transform_fn, epsilon)
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X, y)
    y_pred = knn.predict(X_adv)
    accuracy = accuracy_score(y, y_pred)
    return accuracy

robustness_score = evaluate_robustness(X, y, lsml.transform)
print(f'对抗样本的鲁棒性评分: {robustness_score:.4f}')

def train_final_model(X, y, num_iterations=10):
    lsml = optimize_metric_learning(X, y)
    for _ in range(num_iterations):
        M = train_with_adversarial_examples(X, y, lsml.transform)
    return lsml

final_model = train_final_model(X, y)
print(f'最终模型的度量矩阵: \n{final_model}')
