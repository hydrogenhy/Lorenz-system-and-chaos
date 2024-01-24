import numpy as np
import matplotlib.pyplot as plt


def solve(sigma, r, b):
    coefficients = [1, (sigma + b + 1), (r + sigma) * b, 2 * b * sigma * (r - 1)]
    roots = np.roots(coefficients)
    return roots


def get_lambda(sigma, b):
    if sigma <= b + 1:
        ValueError('参数不合格')
        return
    rh = sigma * ((sigma + b + 3) / (sigma - b - 1))
    t = []
    re = []
    im = []
    for r in np.logspace(np.log10(max(1.3, rh - 100)), np.log10(rh + 500), 100):
        # r = 10 ** r
        solution = solve(sigma, r, b)
        t.append(r)
        re.append(solution[0].real)
        im.append(solution[0].imag)
        t.append(r)
        re.append(solution[1].real)
        im.append(solution[1].imag)
        t.append(r)
        re.append(solution[2].real)
        im.append(solution[2].imag)
        # if solution[0].real > 0:
        #     print(solution)
    return t, re, im


def plots(t, re, im):
    plt.close('all')
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    plt.plot([0, 0], [-35, 35], '--', color='gray')
    plt.scatter(re, im, c=t, cmap='viridis', marker='o')
    ax.set_title('$(\\sigma, r_H, b)=(5, 25, 2)$')
    ax.set_xlabel('real')
    ax.set_ylabel('imag')
    plt.colorbar(label='r')
    plt.show()


if __name__ == '__main__':
    t0, re0, im0 = get_lambda(5, 2)
    plots(t0, re0, im0)
