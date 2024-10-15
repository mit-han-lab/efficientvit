import math

import numpy as np

__all__ = ["OneEuroFilter", "GazeSmoother", "LandmarkSmoother"]


class OneEuroFilter:
    def __init__(self, t0=-1.0, x0=0.0, dx0=0.0, min_cutoff=1.0, beta=0.01, d_cutoff=1.0):
        """Initialize the one euro filter."""
        # The parameters.
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        # Previous values.
        self.x_prev = float(x0)
        self.dx_prev = float(dx0)
        self.t_prev = float(t0)

    def __call__(self, x, t):
        if self.t_prev < 0.0:
            self.t_prev = t
            self.x_prev = x
            return x
        """Compute the filtered signal."""
        t_e = t - self.t_prev

        # The filtered derivative of the signal.
        a_d = self.smoothing_factor(t_e, self.d_cutoff)
        dx = (x - self.x_prev) / t_e
        dx_hat = self.exponential_smoothing(a_d, dx, self.dx_prev)

        # The filtered signal.
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self.smoothing_factor(t_e, cutoff)
        x_hat = self.exponential_smoothing(a, x, self.x_prev)

        # Memorize the previous values.
        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t

        return x_hat

    def smoothing_factor(self, t_e, cutoff):
        r = 2 * math.pi * cutoff * t_e
        return r / (r + 1)

    def exponential_smoothing(self, a, x, x_prev):
        return a * x + (1 - a) * x_prev


class GazeSmoother:
    def __init__(self, filter, **kwargs):
        self.yaw_smoother = filter(**kwargs)
        self.pitch_smoother = filter(**kwargs)

    def __call__(self, yawpitch, **kwargs):
        return self.yaw_smoother(yawpitch[0], **kwargs), self.pitch_smoother(yawpitch[1], **kwargs)


class LandmarkSmoother:
    def __init__(self, filter, pt_num, **kwargs):
        self.pt_num = pt_num
        self.filters = []
        for i in range(pt_num):
            self.filters.append([filter(**kwargs), filter(**kwargs)])

    def __call__(self, landmark, **kwargs):
        dtype = np.array(landmark).dtype
        landmark = np.array(landmark).reshape(self.pt_num, 2)
        ret = [
            [
                self.filters[i][0](landmark[i][0], **kwargs),
                self.filters[i][1](landmark[i][1], **kwargs),
            ]
            for i in range(self.pt_num)
        ]
        return np.array(ret).astype(dtype)
