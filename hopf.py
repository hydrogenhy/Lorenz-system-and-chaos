import numpy as np
import matplotlib.pyplot as plt

# 定义一个函数，该函数返回矩阵 A 特征值
def compute_eigenvalues(A):
    eigenvalues, _ = np.linalg.eig(A)
    return eigenvalues

# 定义参数范围
param_range = np.linspace(-2, 2, 100)

# 初始化特征值存储数组
eigenvalues_array = np.zeros((len(param_range), 2), dtype=complex)

# 循环计算不同参数下的特征值
for i, param in enumerate(param_range):
    # 在这里根据需要设置矩阵 A 的参数
    A = np.array([[1, param], [0, 1]])

    # 计算特征值并存储
    eigenvalues_array[i, :] = compute_eigenvalues(A)

# 绘制特征值随参数变化的图像
for i in range(A.shape[0]):
    plt.plot(param_range, eigenvalues_array[:, i].real, label=f'Real part of eigenvalue {i+1}')
    plt.plot(param_range, eigenvalues_array[:, i].imag, label=f'Imaginary part of eigenvalue {i+1}', linestyle='--')

plt.xlabel('Parameter Value')
plt.ylabel('Eigenvalue')
plt.title('Eigenvalues of Matrix A vs Parameter')
plt.legend()
plt.grid(True)
plt.show()
