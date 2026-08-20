# fit X,Y,Z to a 2D gaussian
from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc


# def gaussian_2d(x: float, y: float, mu, cov) -> float:
#     inv_cov = np.linalg.inv(cov)
#     det_cov = float(np.linalg.det(cov))
#     r = np.array([x, y]).T - mu
#     z = np.exp(-0.5 * (r @ inv_cov @ r.T))
#     coeff = 1 / (2 * np.pi * np.sqrt(det_cov))
#     return coeff * z


def gaussian_1d(x, mu, sigma):
    return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


# making it meshgrid compatible
def gaussian_2d(x, y, mu, cov):
    inv_cov = np.linalg.inv(cov)
    det_cov = np.linalg.det(cov)
    # Compute the differences
    diff_x = x - mu[0]
    diff_y = y - mu[1]
    # Compute the exponent for the Gaussian
    exponent = -0.5 * (
        inv_cov[0, 0] * diff_x**2
        + 2 * inv_cov[0, 1] * diff_x * diff_y
        + inv_cov[1, 1] * diff_y**2
    )
    coeff = 1 / (2 * np.pi * np.sqrt(det_cov))
    return coeff * np.exp(exponent)


def gaussian_2d_exponent(x, y, mu, cov):
    inv_cov = np.linalg.inv(cov)
    # Compute the differences
    diff_x = x - mu[0]
    diff_y = y - mu[1]
    # Compute the exponent for the Gaussian
    exponent = -0.5 * (
        inv_cov[0, 0] * diff_x**2
        + 2 * inv_cov[0, 1] * diff_x * diff_y
        + inv_cov[1, 1] * diff_y**2
    )
    return exponent


def gaussian_1d_offset(x, mu, sigma, scale, offset) -> float:
    return gaussian_1d(x, mu, sigma) * (scale * (sigma * np.sqrt(2 * np.pi))) + offset


def gaussian_2d_offset(x, y, mu, cov, scale, offset) -> float:
    return (
        gaussian_2d(x, y, mu, cov) * (scale * (2 * np.pi * np.sqrt(np.linalg.det(cov))))
        + offset
    )


def gaussian_2d_offset_saturated(
    x: float, y: float, mu, cov, scale, offset, saturation
) -> float:
    return np.clip(gaussian_2d(x, y, mu, cov) * scale + offset, 0, saturation)


def gaussian_2d_smooth_heaviside(
    x: float, y: float, mu, cov, amp, transition_width
) -> float:
    inv_cov = np.linalg.inv(cov)
    r = np.array([x, y]) - mu
    quadratic_form = r.T @ inv_cov @ r

    # Define a smooth transition using a sigmoid-like function
    # smooth_transition = 1 / (1 + np.exp((quadratic_form - 1) / transition_width))
    # using scipy.erfc
    smooth_transition = amp * erfc((quadratic_form - 1) / transition_width)

    return smooth_transition


def statistics_for_gaussian_1d(xdata, Idata):
    # Flatten the arrays in case they are not 1D
    xdata = xdata.flatten()
    Idata = Idata.flatten()

    # Calculate the sum of the data values
    sum_I = np.sum(Idata)

    # Calculate the weighted mean (mu)
    mu = np.sum(xdata * Idata) / sum_I

    # Calculate the weighted variance (sigma^2)
    sigma_squared = np.sum(Idata * (xdata - mu) ** 2) / sum_I
    sigma = np.sqrt(sigma_squared)

    return mu, sigma


def statistics_for_gaussian2d(xdata, ydata, Idata):
    # Flatten the arrays in case they are not 1D
    xdata = xdata.flatten()
    ydata = ydata.flatten()
    Idata = Idata.flatten()

    # Calculate the sum of the data values
    sum_I = np.sum(Idata)

    # Calculate the weighted mean (mu)
    mu_x = np.sum(xdata * Idata) / sum_I
    mu_y = np.sum(ydata * Idata) / sum_I
    mu = np.array([mu_x, mu_y])

    # Center the coordinates by subtracting the mean
    x_centered = xdata - mu_x
    y_centered = ydata - mu_y

    # Calculate the elements of the covariance matrix
    sigma_xx = np.sum(Idata * x_centered * x_centered) / sum_I
    sigma_xy = np.sum(Idata * x_centered * y_centered) / sum_I
    sigma_yy = np.sum(Idata * y_centered * y_centered) / sum_I

    # Assemble the covariance matrix (cov)
    cov = np.array([[sigma_xx, sigma_xy], [sigma_xy, sigma_yy]])

    return mu, cov


def popt_get_mu_cov_1d(popt):
    mu = popt[0]
    sigma = popt[1]
    return mu, sigma


def popt_get_mu_cov_2d(popt):
    mu = popt[:2]
    cov = np.array([[popt[2], popt[3]], [popt[3], popt[4]]])
    return mu, cov


def fit_gaussian_1d(X, Z, p0=None, offset=False):
    X = np.array(X)
    Z = np.array(Z)
    if offset == False:
        # fit to gaussian_1d using least square
        def _residuals(p, x, z):
            z_fit = gaussian_1d(x, p[0], p[1])
            return z - z_fit

    else:

        def _residuals(p, x, z):
            z_fit = gaussian_1d_offset(x, p[0], p[1], p[2], p[3])
            return z - z_fit

    xdata = np.array(X).flatten()
    zdata = np.array(Z).flatten()
    if p0 is None:
        mu, sigma = statistics_for_gaussian_1d(X, Z)
        print("Statistics for Gaussian 1D: ", mu, sigma)
        if offset == False:
            p0 = np.array([mu, sigma])
        else:
            # get the value at mu
            xidx = np.unravel_index(np.argmin(np.abs(X - mu)), X.shape)[0]
            Z_center = Z[xidx]
            print("X,Z_center: ", X[xidx], Z_center)
            if np.abs(Z_center - np.min(Z)) > np.abs(Z_center - np.max(Z)):
                print("Z_center is closer to max(Z)")
                scale = +(np.max(Z) - np.min(Z))
                offset = np.min(Z)
            else:
                print("Z_center is closer to min(Z)")
                scale = -(np.max(Z) - np.min(Z))
                offset = np.max(Z)
            p0 = np.array([mu, sigma, scale, offset])
        print("Initial guess for p0: ", p0)
        print("Initial residuals: ", np.sum(_residuals(p0, xdata, zdata) ** 2))
    from scipy.optimize import least_squares

    res = least_squares(_residuals, p0, args=(xdata, zdata), ftol=1e-10, xtol=1e-10)
    popt = res.x
    print("Final residuals: ", np.sum(_residuals(popt, xdata, zdata) ** 2))
    return popt


def fit_gaussian_2d(X, Y, Z, p0=None, offset=False, saturation=None):
    X = np.array(X)
    Y = np.array(Y)
    Z = np.array(Z)

    if offset == False:
        # fit to gaussian_2d_cov using least square
        def _residuals(p, x, y, z):
            mu, cov = popt_get_mu_cov_2d(p)
            z_fit = np.array([gaussian_2d(x_, y_, mu, cov) for x_, y_ in zip(x, y)])
            return z - z_fit

    else:

        def _residuals(p, x, y, z):
            mu, cov = popt_get_mu_cov_2d(p)
            z_fit = np.array(
                [
                    gaussian_2d_offset(x_, y_, mu, cov, p[5], p[6])
                    for x_, y_ in zip(x, y)
                ]
            )
            return z - z_fit

    if saturation is not None:

        def _residuals(p, x, y, z):
            mu, cov = popt_get_mu_cov_2d(p)
            z_fit = np.array(
                [
                    gaussian_2d_offset_saturated(
                        x_, y_, mu, cov, p[5], p[6], saturation
                    )
                    for x_, y_ in zip(x, y)
                ]
            )
            return z - z_fit

    xdata = np.array(X).flatten()
    ydata = np.array(Y).flatten()
    zdata = np.array(Z).flatten()

    if p0 is None:
        mu, cov = statistics_for_gaussian2d(X, Y, Z)
        print("Statistics for Gaussian 2D: ", mu, cov)
        sigma = np.sqrt(np.linalg.eigvals(cov))
        print("Sigma: ", sigma)
        if offset == False:
            p0 = np.array([mu[0], mu[1], cov[0, 0], cov[0, 1], cov[1, 1]])
        else:
            # get the value at (mu[0],mu[1])
            xidx = np.unravel_index(np.argmin(np.abs(X - mu[0])), X.shape)[1]
            yidx = np.unravel_index(np.argmin(np.abs(Y - mu[1])), Y.shape)[0]
            Z_center = Z[yidx, xidx]
            print("X,Y,Z_center: ", X[yidx, xidx], Y[yidx, xidx], Z_center)
            if np.abs(Z_center - np.min(Z)) > np.abs(Z_center - np.max(Z)):
                print("Z_center is closer to max(Z)")
                scale = +(np.max(Z) - np.min(Z))
                offset = np.min(Z)
            else:
                print("Z_center is closer to min(Z)")
                scale = -(np.max(Z) - np.min(Z))
                offset = np.max(Z)
            p0 = np.array(
                [mu[0], mu[1], cov[0, 0], cov[0, 1], cov[1, 1], scale, offset]
            )
            if saturation is not None:
                p0 = np.append(p0, saturation)
        print("Initial guess for p0: ", p0)

    from scipy.optimize import least_squares

    res = least_squares(_residuals, p0, args=(xdata, ydata, zdata))
    popt = res.x
    return popt


def fit_gaussian_2d_smooth_heaviside(X, Y, Z, p0=None):
    X = np.array(X)
    Y = np.array(Y)
    Z = np.array(Z)

    # fit to gaussian_2d_cov using least square
    def _residuals(p, x, y, z):
        mu, cov = popt_get_mu_cov_2d(p)
        z_fit = np.array(
            [
                gaussian_2d_smooth_heaviside(x_, y_, mu, cov, p[5], p[6])
                for x_, y_ in zip(x, y)
            ]
        )
        return z - z_fit

    xdata = np.array(X).flatten()
    ydata = np.array(Y).flatten()
    zdata = np.array(Z).flatten()

    if p0 is None:
        mu, cov = statistics_for_gaussian2d(X, Y, Z)
        p0 = np.array(
            [mu[0], mu[1], cov[0, 0], cov[0, 1], cov[1, 1], np.max(Z) / 2, 0.1]
        )
        print("Initial guess for p0: ", p0)

    from scipy.optimize import least_squares

    res = least_squares(_residuals, p0, args=(xdata, ydata, zdata))
    popt = res.x
    return popt


def fit_and_plot_gaussian_2d(X, Y, Z, p0=None, ax=None, offset=False, saturation=None):
    popt = fit_gaussian_2d(X, Y, Z, p0=p0, offset=offset, saturation=saturation)
    bounds_x = (np.min(X), np.max(X))
    bounds_y = (np.min(Y), np.max(Y))
    X_new = np.linspace(*bounds_x, 100)
    Y_new = np.linspace(*bounds_y, 100)
    X_new, Y_new = np.meshgrid(X_new, Y_new)
    mu, cov = popt_get_mu_cov_2d(popt)
    Z_new = np.array(
        [gaussian_2d(x_, y_, mu, cov) for x_, y_ in zip(X_new.ravel(), Y_new.ravel())]
    )
    #
    if ax is None:
        fig, ax = plt.subplots()
    ax.imshow(
        Z.reshape(X.shape) / np.max(Z),
        origin="lower",
        extent=[bounds_x[0], bounds_x[1], bounds_y[0], bounds_y[1]],
    )
    ax.contour(X_new, Y_new, Z_new.reshape(X_new.shape), cmap="jet")
    return popt


def fit_and_plot_smooth_heaviside(X, Y, Z, p0=None, ax=None):
    popt = fit_gaussian_2d_smooth_heaviside(X, Y, Z, p0=p0)
    bounds_x = (np.min(X), np.max(X))
    bounds_y = (np.min(Y), np.max(Y))
    X_new = np.linspace(*bounds_x, 100)
    Y_new = np.linspace(*bounds_y, 100)
    X_new, Y_new = np.meshgrid(X_new, Y_new)
    mu, cov = popt_get_mu_cov_2d(popt)
    Z_new = np.array(
        [
            gaussian_2d_smooth_heaviside(x_, y_, mu, cov, popt[5], popt[6])
            for x_, y_ in zip(X_new.ravel(), Y_new.ravel())
        ]
    )
    #
    if ax is None:
        fig, ax = plt.subplots()
    ax.imshow(
        Z.reshape(X.shape) / np.max(Z),
        origin="lower",
        extent=[bounds_x[0], bounds_x[1], bounds_y[0], bounds_y[1]],
    )
    ax.contour(X_new, Y_new, Z_new.reshape(X_new.shape), cmap="jet")
    return popt


def ZX(X0, Z_row):
    """average X with weights Z_row"""
    return np.dot(X0, Z_row) / np.sum(Z_row)


def ZZX(X0, Z_row):
    """average (X0-mu)^2 with weights Z_row"""
    mu = ZX(X0, Z_row)
    return np.dot((X0 - mu) ** 2, Z_row) / np.sum(Z_row)


def ZZZX(X0, Z_row):
    """average (X0-mu)^3 with weights Z_row"""
    mu = ZX(X0, Z_row)
    return np.dot((X0 - mu) ** 3, Z_row) / np.sum(Z_row)


def statistics_skewness(X0, Z_row):
    return ZZZX(X0, Z_row) / ZZX(X0, Z_row) ** 1.5


if __name__ == "__main__":
    # test 1D gaussian fitting
    x = np.linspace(-5, 5, 100)
    mu_true = 0.5
    sigma_true = 1.0
    scale_true = 2.0
    offset_true = 0.5
    y = gaussian_1d_offset(x, mu_true, sigma_true, scale_true, offset_true)
    popt = fit_gaussian_1d(x, y, offset=True)
    print("Fitted parameters: ", popt)
    plt.plot(x, y, label="Data")
    y_fit = gaussian_1d_offset(x, *popt)
    plt.plot(x, y_fit, "--", label="Gaussian fit")
    plt.legend()
    plt.title("1D Gaussian Fit Test")
    plt.xlabel("x")
    plt.ylabel("Intensity")
    plt.show()
    # test 2D gaussian fitting
    x = np.linspace(-1, 1, 20)
    y = np.linspace(-1, 1, 20)
    X, Y = np.meshgrid(x, y)
    cov = [[1, 0], [0, 1]]
    sigma = np.sqrt(np.linalg.eigvals(cov))
    print("Sigma: ", sigma)
    Z = gaussian_2d(X, Y, mu=[0, 0], cov=cov)
    popt = fit_and_plot_gaussian_2d(X, Y, Z)
    print(popt)
    plt.show()
