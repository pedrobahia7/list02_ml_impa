# %%
# Question 1) g
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

number_of_folds = 10
data = pd.read_csv("../data/bodyfat.csv")
X = data.drop(columns=["BodyFat", "Density"])
y = data["BodyFat"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0
)

# %%
# (i) Qual é o erro médio quadrático de uma regressão linear nos dados de teste?
from sklearn.metrics import mean_squared_error

linear_regression_model = LinearRegression(fit_intercept=True)
linear_regression_model.fit(X_train, y_train)
y_hat = linear_regression_model.predict(X_test)
mse_linear = mean_squared_error(y_true=y_test, y_pred=y_hat)

y_hat_train = linear_regression_model.predict(X_train)
mse_linear_train = mean_squared_error(y_true=y_train, y_pred=y_hat_train)


print(
    "O erro médio quadrático da regressão linear para o conjunto de teste é",
    mse_linear,
)
print(
    "O erro médio quadrático da regressão linear para o conjunto de treino é",
    mse_linear_train,
)


# %%
# (ii) Qual é o erro médio quadrático de uma regressão ridge nos dados de teste?
alphas = 10 ** np.linspace(3, -2, 100)
ridge_cv_mse = []
for alpha in tqdm(alphas):
    cv_MSE = []
    folds = KFold(
        n_splits=number_of_folds, shuffle=True, random_state=2023
    ).split(X_train, y_train)

    for train_idx, val_idx in folds:
        ridge_pipeline = make_pipeline(
            StandardScaler(), Ridge(alpha=alpha)
        )
        ridge_pipeline[1].fit(
            X_train.iloc[train_idx], y_train.iloc[train_idx]
        )  # treine seu modelo nos folds de treino
        y_hat = ridge_pipeline[1].predict(
            X_train.iloc[val_idx]
        )  # faça suas previsões no fold de validação
        cv_MSE.append(
            mean_squared_error(y_true=y_train.iloc[val_idx], y_pred=y_hat)
        )  # salve o MSE encontrado no fold de validação

    ridge_cv_mse.append(np.mean(cv_MSE))

assert len(ridge_cv_mse) == len(alphas)
optimal_alpha = alphas[
    np.argmin(ridge_cv_mse)
]  # encontre o melhor valor de alpha
print("O melhor alpha encontrado é:", optimal_alpha)
print("Com ele, o MSE médio nos 10 folds foi:", np.min(ridge_cv_mse))
ridge_pipeline = make_pipeline(
    StandardScaler(), Ridge(alpha=optimal_alpha)
)
ridge_pipeline[1].fit(
    X_train, y_train
)  # treine seu modelo em todo o conjunto de treino
y_hat_ridge = ridge_pipeline[1].predict(
    X_test
)  # faça as previsões em todo o conjunto de teste
ridge_test_mse = mean_squared_error(
    y_true=y_test, y_pred=y_hat_ridge
)  # salve o MSE encontrado no conjunto de teste

y_hat_ridge_train = ridge_pipeline[1].predict(X_train)
ridge_train_mse = mean_squared_error(
    y_true=y_train, y_pred=y_hat_ridge_train
)


print(
    "O erro quadrático médio da regressão ridge para o conjunto de teste é:",
    ridge_test_mse,
)
print(
    "O erro quadrático médio da regressão ridge para o conjunto de treino é:",
    ridge_train_mse,
)

# %%
# (iii) Finalmente, vamos usar kernel ridge regression para tentar melhorar esse resultado. Para isso, vamos olhar para os seguintes hiperparâmetros:
kernels = ["linear", "polynomial", "rbf", "laplacian"]
gammas = [10**-3]
alphas = 10 ** np.linspace(3, -2, 100)
hyperparams = [
    (kernel, gamma, alpha)
    for kernel in kernels
    for gamma in gammas
    for alpha in alphas
]

kernel_ridge_cv_mse = []
for hyperparam in tqdm(hyperparams):
    kernel_fold, gamma_fold, alpha_fold = hyperparam
    cv_MSE = []
    folds = KFold(
        n_splits=number_of_folds, shuffle=True, random_state=2023
    ).split(X_train, y_train)

    for train_idx, val_idx in folds:
        kernel_ridge_pipeline = make_pipeline(
            StandardScaler(),
            KernelRidge(
                kernel=kernel_fold, alpha=alpha_fold, gamma=gamma_fold
            ),
        )
        kernel_ridge_pipeline[1].fit(
            X_train.iloc[train_idx], y_train.iloc[train_idx]
        )
        y_hat = kernel_ridge_pipeline[1].predict(X_train.iloc[val_idx])
        cv_MSE.append(
            mean_squared_error(y_true=y_train.iloc[val_idx], y_pred=y_hat)
        )

    kernel_ridge_cv_mse.append(np.mean(cv_MSE))

assert len(kernel_ridge_cv_mse) == len(hyperparams)
optimal_hyperparam = hyperparams[np.argmin(kernel_ridge_cv_mse)]
optimal_kernel, optimal_gamma, optimal_alpha = optimal_hyperparam

print(
    f"O melhor hyperparam encontrado é:\n alpha: {optimal_alpha},\n gamma: {optimal_gamma},\n kernel: {optimal_kernel}"
)
print(
    "Com ele, o MSE médio nos 10 folds foi:", np.min(kernel_ridge_cv_mse)
)

kernel_ridge_pipeline = make_pipeline(
    StandardScaler(),
    KernelRidge(
        kernel=optimal_kernel, alpha=optimal_alpha, gamma=optimal_gamma
    ),
)
kernel_ridge_pipeline[1].fit(X_train, y_train)
kernel_y_hat_ridge = kernel_ridge_pipeline[1].predict(X_test)
kernel_ridge_test_mse = mean_squared_error(
    y_true=y_test, y_pred=kernel_y_hat_ridge
)

kernel_y_hat_ridge_train = kernel_ridge_pipeline[1].predict(X_train)
kernel_ridge_train_mse = mean_squared_error(
    y_true=y_train, y_pred=kernel_y_hat_ridge_train
)

print(
    "O erro quadrático médio da regressão ridge com kernel para o conjunto de teste é:",
    kernel_ridge_test_mse,
)
print(
    "O erro quadrático médio da regressão ridge com kernel para o conjunto de treino é:",
    kernel_ridge_train_mse,
)


# %%
# (iv) Encontre o erro de teste do modelo de kernel ridge com o hiperparâmetro encontrado no item anterior. Compare a performance de regressão linear, ridge e kernel ridge.

# Encontrar erro in-sample para avaliação do ajuste do modelo


print("Treino:")
print(f"Erro médio quadrático da regressão linear: {mse_linear_train}")
print(f"Erro médio quadrático da regressão ridge: {ridge_train_mse}")
print(
    f"Erro médio quadrático da regressão ridge com kernel: {kernel_ridge_train_mse}"
)

print("\n")
print("Teste:")
print(f"Erro médio quadrático da regressão linear: {mse_linear}")
print(f"Erro médio quadrático da regressão ridge: {ridge_test_mse}")
print(
    f"Erro médio quadrático da regressão ridge com kernel: {kernel_ridge_test_mse}"
)

print(
    "Dos três modelos analisados, o erro de teste da Regressão Linear foi o menor dele."
    "Mesmo com o a divisão em k-folds, aparentemente houve overfitting na regressão Ridge com Kernel, já que seu erro de treino é consideravelmente menor que o erro de teste"
    "e o erro de treino é o menor dentre os demais modelos. Este comportamento não é anormal, dado que estamos projetando os dados em um espaço vetorial de maior dimensão "
    "Já a regressão ridge teve o pior dentre os erros de treino, o que é esperado dada a restrição introduzida pelo método. Seu erro de teste foi maior que o do regressor linear"
    "o que também não é anormal"
)
