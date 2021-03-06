#%%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import itertools
from pyk_st.math import functional

heaviside = lambda x:1*(x>0)
#_seikaku_na_kansuu = lambda x: np.sin(np.pi * x) / np.pi * x + 0.1 * x

_seikaku_na_kansuu = lambda x:np.sin(x)+0.1*x-0.5*heaviside(x)

def generate_suitably(n:int = 100) -> (np.ndarray, np.ndarray): # 1 x n, 1 x n
    teacher_x = np.array([[6 * np.random.rand() - 3] for _ in range(n)])
    teacher_y = _seikaku_na_kansuu(teacher_x) + 0.1 * np.random.randn(n).reshape(-1, 1)

    return teacher_x, teacher_y

def generate_suitably2D(n:int = 100) -> (pd.core.frame.DataFrame, pd.core.series.Series): # 2 x n, 1 x n
    teacher_X = pd.DataFrame([[6 * np.random.rand() - 3, np.random.randn()] for _ in range(n)], columns=['main', 'noise'])
    teacher_y = _seikaku_na_kansuu(teacher_X['main'].values) + 0.1*teacher_X['noise'].values + 0.1*np.random.randn(n).reshape(-1, 1)

    return teacher_X, teacher_y

def generate_suitably3D(n:int = 100) -> (pd.core.frame.DataFrame, pd.core.series.Series): # 3 x n, 1 x n
    teacher_X = pd.DataFrame([[6 * np.random.rand() - 3, np.random.randn(), np.random.randn()] for _ in range(n)], columns=['main', 'noise', 'random'])
    teacher_y = _seikaku_na_kansuu(teacher_X['main'].values) + 0.1*teacher_X['noise'].values + 0.1*np.random.randn(n).reshape(-1, 1)

    return teacher_X, teacher_y
#%%
# predictor: sklearn KernelRidge, RandomForest, etc... , whitch has attribute `predict`
def predictor_parameter_test(predictor, params) -> None:
    keys = list(params.keys())
    train_x, train_y = generate_suitably()

    N = 1000
    X = np.linspace(-3, 3, N).reshape(-1, 1)
    #P = predictor.predict(X)

    plt.scatter(train_x, train_y, color='black', marker='.', label='sample point')

    for prms in itertools.product(*params.values()):
        prm = {keys[i]:prms[i] for i in range(len(prms))}
        print(prm)
        for k,v in prm.items():
            setattr(predictor, k, v) 
        predictor.fit(train_x, train_y)
        P = predictor.predict(X)
        plt.plot(X,P,label=str(prm))
        print(prms)
    plt.plot(X, _seikaku_na_kansuu(X), color='black', linewidth=0.5, label='exact')
    plt.legend()

if __name__ == "__main__":
    #内部インポートテスト
    #plt.plot(np.linspace(-1,1,100), functional.f(np.linspace(-1,1,100)))
    #plt.show()

    from sklearn.kernel_ridge import KernelRidge

    rgr = KernelRidge(kernel='rbf', alpha=1.0, gamma=1.0)
    predictor_parameter_test(predictor=rgr, params={'alpha':[0, 1.0, 10.0], 'gamma':[1.0]})
    plt.show()