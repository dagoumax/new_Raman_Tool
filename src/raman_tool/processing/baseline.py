"""基线校正算法.

默认使用 arPLS (asymmetrically reweighted Penalized Least Squares) 算法。
也保留多项式拟合作为备选。
"""

import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from raman_tool.models import Spectrum
from raman_tool.safety import check_baseline_points
from raman_tool.validation import integer_parameter, numeric_vector, positive_float, validate_spectrum_arrays


def arPLS(y: np.ndarray, lam: float = 1e5, max_iter: int = 50, tol: float = 1e-6) -> np.ndarray:
    """arPLS 基线估计 (不对称重加权惩罚最小二乘).

    参考: Baek et al., Analyst 140(1), 250-257 (2015).

    Args:
        y: 输入信号 (1D array)
        lam: 平滑参数，越大基线越平滑 (默认 1e5)
        max_iter: 最大迭代次数
        tol: 收敛容差

    Returns:
        baseline: 估计的基线 (与 y 同形状)
    """
    y = numeric_vector(y, "arPLS signal", minimum=3)
    lam = positive_float(lam, "lam")
    max_iter = integer_parameter(max_iter, "max_iter", minimum=1)
    tol = positive_float(tol, "tol")
    if tol >= 1:
        raise ValueError("tol must be less than 1")
    N = len(y)
    check_baseline_points(N)

    # 稀疏二阶差分矩阵
    D = diags([1, -2, 1], [0, 1, 2], shape=(N - 2, N), format="csc")
    DTD = D.T @ D

    w = np.ones(N, dtype=np.float64)

    for _ in range(max_iter):
        W = diags(w, 0, shape=(N, N), format="csc")
        A = W + lam * DTD
        b = w * y
        z = spsolve(A, b)
        if not np.all(np.isfinite(z)):
            raise ValueError("arPLS failed to produce a finite baseline; check signal scale and lam")

        d = y - z
        idx = d < 0  # y < z 的区域 (信号)

        if np.any(idx):
            dn = d[idx]
            m = np.mean(dn)
            s = np.std(dn)
            s = max(s, np.finfo(np.float64).eps)
        else:
            m, s = 0.0, 1.0

        arg = 2 * (d - (-m + 2 * s)) / s
        arg = np.clip(arg, -60, 60)
        w_new = np.where(d >= 0, 1.0 / (1.0 + np.exp(arg)), 1.0)

        norm_w = np.linalg.norm(w)
        norm_w = max(norm_w, np.finfo(np.float64).eps)
        if np.linalg.norm(w_new - w) / norm_w < tol:
            w = w_new
            W = diags(w, 0, shape=(N, N), format="csc")
            z = spsolve(W + lam * DTD, w * y)
            break

        w = w_new

    if not np.all(np.isfinite(z)):
        raise ValueError("arPLS failed to produce a finite baseline; check signal scale and lam")
    return z


def poly_baseline(spectrum: Spectrum, degree: int = 3) -> np.ndarray:
    """使用多项式拟合估计基线.

    Args:
        spectrum: 输入光谱
        degree: 多项式阶数

    Returns:
        baseline: 基线数组
    """
    x, y = validate_spectrum_arrays(spectrum.raman_shift, spectrum.intensity)
    degree = integer_parameter(degree, "degree")
    if degree >= len(x):
        raise ValueError("degree must be less than the number of data points")
    if degree == 0:
        return np.full_like(y, np.mean(y))
    # Scale the x-domain before fitting, avoiding ill-conditioning at large offsets.
    fitted = np.polynomial.Polynomial.fit(x, y, degree)
    return fitted(x)


def subtract_baseline(
    spectrum: Spectrum,
    method: str = "arPLS",
    lam: float = 1e5,
    degree: int = 3,
    max_iter: int = 50,
    tol: float = 1e-6,
) -> Spectrum:
    """对光谱进行基线校正.

    Args:
        spectrum: 输入光谱
        method: 基线方法 ("arPLS" 默认, 或 "poly")
        lam: arPLS 平滑参数 (默认 1e5)
        degree: poly 多项式阶数 (默认 3)
        max_iter: arPLS 最大迭代次数

    Returns:
        baseline-corrected Spectrum 对象
    """
    validate_spectrum_arrays(spectrum.raman_shift, spectrum.intensity)
    if method == "arPLS":
        baseline = arPLS(spectrum.intensity, lam=lam, max_iter=max_iter, tol=tol)
    elif method == "poly":
        baseline = poly_baseline(spectrum, degree)
    else:
        raise ValueError(f"未知的基线方法: {method}。支持: arPLS, poly")

    corrected_intensity = spectrum.intensity - baseline

    return Spectrum(
        raman_shift=spectrum.raman_shift.copy(),
        intensity=corrected_intensity,
        filename=spectrum.filename,
        metadata={
            **spectrum.metadata,
            "baseline_corrected": True,
            "baseline_method": method,
        },
    )
