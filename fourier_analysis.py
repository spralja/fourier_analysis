import numpy as np


class FourierAnalysis:
    _mu = 0.75 / np.pi ** 3  # equation (1.1.1)
    _a = -np.pi / 4.0  # equation(2.3.1)
    _b = np.pi / 4.0  # equation(2.3.2)

    @staticmethod
    def get_mu():
        return FourierAnalysis._mu

    @staticmethod
    def get_a():
        return FourierAnalysis._a

    @staticmethod
    def get_b():
        return FourierAnalysis._b

    @staticmethod
    def d_beta(n_beta):  # equation (2.3.4)
        a = FourierAnalysis.get_a()
        b = FourierAnalysis.get_b()
        return (b - a) / n_beta

    @staticmethod
    def beta(n_beta, index):
        a = FourierAnalysis.get_a()
        d_beta = FourierAnalysis.d_beta(n_beta)
        return a + d_beta * (index + 0.5)

    @staticmethod
    def z(n_sigma):
        return np.arange(-n_sigma, n_sigma + 1)

    @staticmethod
    def z_0(n_sigma):
        z_0 = []
        for sigma in np.arange(-n_sigma, n_sigma + 1):
            if sigma == 0:
                continue

            z_0.append(sigma)

        z_0 = np.array(z_0)
        return z_0

    @staticmethod
    def z_plus(n_sigma):
        z_plus = []
        for sigma in np.arange(1, n_sigma + 1):
            z_plus.append(sigma)

        z_plus = np.array(z_plus)
        return z_plus

    @staticmethod
    def generate_trigs(n_beta):
        sin = open("data/n_beta_%d/trigs/sin.txt" % n_beta, "w+")
        sin.truncate(0)

        cos = open("data/n_beta_%d/trigs/cos.txt" % n_beta, "w+")
        cos.truncate(0)

        tan = open("data/n_beta_%d/trigs/tan.txt" % n_beta, "w+")
        tan.truncate(0)

        for i in np.arange(n_beta):
            beta = FourierAnalysis.beta(n_beta, i)
            sin.write(str(np.sin(beta)) + " ")
            cos.write(str(np.cos(beta)) + " ")
            tan.write(str(np.tan(beta)) + " ")

    @staticmethod
    def generate_alpha(n_beta, n_sigma):
        fourier_analysis = FourierAnalysis(n_beta, n_sigma, _load_alpha=False, _load_c=False)

        z_0 = fourier_analysis.get_z_0()
        z_plus = fourier_analysis.get_z_plus()

        fourier_analysis._alpha = Alpha(n_beta, n_sigma)
        for n in z_0:
            for m in z_plus:
                for phi in np.arange(n_beta):
                    for theta in np.arange(n_beta):
                        alpha = fourier_analysis.alpha(n, m, phi, theta)
                        fourier_analysis._alpha.set(n, m, phi, theta, alpha)

        for n in z_plus:
            for phi in np.arange(n_beta):
                for theta in np.arange(n_beta):
                    alpha = fourier_analysis.alpha(n, 0, phi, theta)
                    fourier_analysis._alpha.set(n, 0, phi, theta, alpha)

        fourier_analysis._alpha.unload()

    @staticmethod
    def generate_c(n_beta, n_sigma):
        fourier_analysis = FourierAnalysis(n_beta, n_sigma, _load_c=False)
        fourier_analysis._c = C(n_beta, n_sigma)
        z_0 = fourier_analysis.get_z_0()
        z_plus = fourier_analysis.get_z_plus()
        for k in z_0:
            for n in z_0:
                for m in z_plus:
                    c = fourier_analysis.numerically_integrate(k, n, m)
                    fourier_analysis._c.set(k, n, m, c)

        for k in z_0:
            for n in z_plus:
                c = fourier_analysis.numerically_integrate(k, n, 0)
                fourier_analysis._c.set(k, n, 0, c)

        for k in z_plus:
            c = fourier_analysis.numerically_integrate(k, 0, 0)
            fourier_analysis._c.set(k, 0, 0, c)

        c = C.get_zero_zero_zero()
        fourier_analysis._c.set(0, 0, 0, c)
        fourier_analysis._c.unload()

    def __init__(self, n_beta, n_sigma, _load_trigs=True, _load_alpha=True, _load_c=True):
        self._n_beta = n_beta  # equation(2.3.3)
        self._d_beta = FourierAnalysis.d_beta(n_beta)  # equation (2.3.4)

        self._n_sigma = n_sigma
        self._z = FourierAnalysis.z(n_sigma)
        self._z_0 = FourierAnalysis.z_0(n_sigma)
        self._z_plus = FourierAnalysis.z_plus(n_sigma)

        if _load_trigs:
            self._load_trigs()

        if _load_alpha:
            self._load_alpha()

        if _load_c:
            self._load_c()
            self._c.print()


    def _load_trigs(self):
        n_beta = self.get_n_beta()

        sin = open("data/n_beta_%d/trigs/sin.txt" % n_beta).read().split(" ")
        cos = open("data/n_beta_%d/trigs/cos.txt" % n_beta).read().split(" ")
        tan = open("data/n_beta_%d/trigs/tan.txt" % n_beta).read().split(" ")

        self._sin_beta = np.full(n_beta, 0.0)
        self._cos_beta = np.full(n_beta, 0.0)
        self._tan_beta = np.full(n_beta, 0.0)

        for i in np.arange(n_beta):
            self._sin_beta[i] = sin[i]
            self._cos_beta[i] = cos[i]
            self._tan_beta[i] = tan[i]

    def _load_alpha(self):
        n_beta = self.get_n_beta()
        n_sigma = self.get_n_sigma()
        alpha = open("data/n_beta_%d/n_sigma_%d/alpha.txt" % (n_beta, n_sigma)).read().split(" ")
        self._alpha = Alpha(n_beta, n_sigma)
        n = -n_sigma
        m = -n_sigma
        phi = 0
        theta = 0
        for _alpha in alpha:
            self._alpha.set(n, m, phi, theta, float(_alpha))
            theta += 1
            if theta == n_beta:
                theta = 0
                phi += 1
                if phi == n_beta:
                    phi = 0
                    m += 1
                    if m == n_sigma + 1:
                        m = -n_sigma
                        n += 1
                        if n == n_sigma + 1:
                            break

    def _load_c(self):
        n_beta = self.get_n_beta()
        n_sigma = self.get_n_sigma()
        f = open("data/n_beta_%d/n_sigma_%d/f.txt" % (n_beta, n_sigma)).read().split(" ")
        g = open("data/n_beta_%d/n_sigma_%d/g.txt" % (n_beta, n_sigma)).read().split(" ")
        z = self.get_z()
        self._c = C(n_beta, n_sigma)

        k = -n_sigma
        n = -n_sigma
        m = -n_sigma

        for _f in f:
            self._c.set_f(k, n, m, float(_f))
            m += 1
            if m == n_sigma + 1:
                m = -n_sigma
                n += 1
                if n == n_sigma + 1:
                    n = -n_sigma
                    k += 1
                    if k == n_sigma + 1:
                        break

        k = -n_sigma
        n = -n_sigma
        m = -n_sigma

        for _g in g:
            self._c.set_g(k, n, m, float(_g))
            m += 1
            if m == n_sigma + 1:
                m = -n_sigma
                n += 1
                if n == n_sigma + 1:
                    n = -n_sigma
                    k += 1
                    if k == n_sigma + 1:
                        break

        self._c.unload(
            path_f="data/n_beta_%d/n_sigma_%d/f2.txt" % (n_beta, n_sigma),
            path_g="data/n_beta_%d/n_sigma_%d/g2.txt" % (n_beta, n_sigma)
        )

    def numerically_integrate(self, k, n, m):
        n_beta = self.get_n_beta()  # equation (2.3.3)
        mu = FourierAnalysis.get_mu()  # equation (1.1.1)
        d_beta2 = self.get_d_beta() ** 2  # equation (2.3.4)
        kappa = self.kappa(k)  # equation (1.1.4)
        sum_f = 0.0  # equation (2.5.1)
        sum_g = 0.0  # equation (2.5.1)
        for phi in np.arange(n_beta):
            for theta in np.arange(n_beta):
                alpha = self.alpha(n=n, m=m, phi=phi, theta=theta)  # equation (1.4.1)
                t1 = np.sin(alpha)  # equation (1.10.1)
                t2 = np.cos(alpha)  # equation (1.10.2)
                p = self.p(k=k, n=n, m=m, phi=phi, theta=theta)  # equation (1.1.2)
                q = self.q(k=k, n=n, m=m, phi=phi, theta=theta)  # equation (1.1.3)
                sum_f += self.f(kappa=kappa, t1=t1, t2=t2, p=p, q=q)  # equation (2.5.1)
                sum_g += self.g(kappa=kappa, t1=t1, t2=t2, p=p, q=q)  # equation (2.5.2.)

        return mu * sum_f * d_beta2, mu * sum_g * d_beta2

    @staticmethod
    def f(kappa, t1, t2, p, q):  # equation (1.12.1)
        return -np.pi * kappa * t1 / p + (kappa * t2 - 1) / q

    @staticmethod
    def g(kappa, t1, t2, p, q):  # equation (1.12.2)
        return np.pi * kappa * t2 / p + kappa * t1 / q

    def get_beta(self, index):
        return self.get_a() + self.get_d_beta() * (index + 0.5)

    @staticmethod
    def kappa(k):  # equation (1.1.4)
        if k % 2 == 0:
            return 1

        return -1

    def alpha(self, n, m, phi, theta):  # equation (1.6.1)
        return np.pi * (m * self.tan(phi) / self.cos(theta) + n * self.tan(theta))

    def p(self, k, n, m, phi, theta):  # equation (1.1.2)
        return m * self.sin(phi) + self.cos(phi) * (k * self.cos(theta) + n * self.sin(theta))

    def q(self, k, n, m, phi, theta):  # equation (1.1.3)
        return (
                m ** 2 * self.sin(phi) * self.tan(phi) +
                2 * m * self.sin(phi) * (k * self.cos(theta) + n * self.sin(theta)) +
                self.cos(phi) * (k * self.cos(theta) + n * self.sin(theta)) ** 2
        )

    def sin(self, index):
        return self._sin_beta[index]

    def cos(self, index):
        return self._cos_beta[index]

    def tan(self, index):
        return self._tan_beta[index]

    def get_n_beta(self):  # equation (1)
        return self._n_beta

    def get_d_beta(self):  # equation (1)
        return self._d_beta

    def get_n_sigma(self):  # equation (3.3.3)
        return self._n_sigma

    def get_z(self):
        return self._z

    def get_z_0(self):
        return self._z_0

    def get_z_plus(self):
        return self._z_plus


class Alpha:
    def __init__(self, n_beta, n_sigma):
        self._n_beta = n_beta
        self._n_sigma = n_sigma
        z = FourierAnalysis.z(n_sigma)
        self._data = np.full((z.size, z.size, n_beta, n_beta), 0.0)

    def get(self, n, m, phi, theta):
        i_n = self._get_sigma_to_index_sigma(n)
        i_m = self._get_sigma_to_index_sigma(m)
        return self._data[i_n][i_m][phi][theta]

    def set(self, n, m, phi, theta, value):
        i_n = self._get_sigma_to_index_sigma(n)
        i_m = self._get_sigma_to_index_sigma(m)
        self._data[i_n][i_m][phi][theta] = value

        i_minus_n = self._get_sigma_to_index_sigma(-n)
        i_minus_m = self._get_sigma_to_index_sigma(-m)
        self._data[i_minus_n][i_minus_m][phi][theta] = value

    def get_n_beta(self):
        return self._n_beta

    def get_n_sigma(self):
        return self._n_sigma

    def _get_sigma_to_index_sigma(self, sigma):
        n_sigma = self.get_n_sigma()
        return sigma + n_sigma

    def unload(self, path=None):
        n_beta = self.get_n_beta()
        n_sigma = self.get_n_sigma()
        if path is None:
            path = "data/n_beta_%d/n_sigma_%d/alpha.txt" % (n_beta, n_sigma)

        alpha = open(path, "w+")
        alpha.truncate(0)

        for datum in self._data:
            for _datum in datum:
                for __datum in _datum:
                    for ___datum in __datum:
                        alpha.write(str(___datum) + " ")


class C:
    _zero_zero_zero = (1, 0.0)

    @staticmethod
    def get_zero_zero_zero():
        return C._zero_zero_zero

    def __init__(self, n_beta, n_sigma):
        self._n_beta = n_beta
        self._n_sigma = n_sigma
        z = FourierAnalysis.z(n_sigma)
        self._f_data = np.full((z.size, z.size, z.size), 0.0)
        self._g_data = np.full((z.size, z.size, z.size), 0.0)

    def get_n_beta(self):
        return self._n_beta

    def get_n_sigma(self):
        return self._n_sigma

    def _get_sigma_to_index_sigma(self, sigma):
        n_sigma = self.get_n_sigma()
        return sigma + n_sigma

    def set_f(self, k, n, m, value):
        i_k = self._get_sigma_to_index_sigma(k)
        i_n = self._get_sigma_to_index_sigma(n)
        i_m = self._get_sigma_to_index_sigma(m)
        self._f_data[i_k][i_n][i_m] = value

        i_minus_k = self._get_sigma_to_index_sigma(-k)
        i_minus_n = self._get_sigma_to_index_sigma(-n)
        i_minus_m = self._get_sigma_to_index_sigma(-m)
        self._f_data[i_minus_k][i_minus_n][i_minus_m] = value

    def set_g(self, k, n, m, value):
        i_k = self._get_sigma_to_index_sigma(k)
        i_n = self._get_sigma_to_index_sigma(n)
        i_m = self._get_sigma_to_index_sigma(m)
        self._g_data[i_k][i_n][i_m] = value

        i_minus_k = self._get_sigma_to_index_sigma(-k)
        i_minus_n = self._get_sigma_to_index_sigma(-n)
        i_minus_m = self._get_sigma_to_index_sigma(-m)
        self._g_data[i_minus_k][i_minus_n][i_minus_m] = -value

    def set(self, k, n, m, value):
        self.set_f(k, n, m, value[0])
        self.set_g(k, n, m, value[1])

    def get(self, k, m, n):
        i_k = self._get_sigma_to_index_sigma(k)
        i_n = self._get_sigma_to_index_sigma(n)
        i_m = self._get_sigma_to_index_sigma(m)
        return self._f_data[i_k][i_n][i_m], self._g_data[i_k][i_n][i_m]

    def unload(self, path_f=None, path_g=None):
        n_beta = self.get_n_beta()
        n_sigma = self.get_n_sigma()
        if path_f is None:
            path_f = "data/n_beta_%d/n_sigma_%d/f.txt" % (n_beta, n_sigma)

        if path_g is None:
            path_g = "data/n_beta_%d/n_sigma_%d/g.txt" % (n_beta, n_sigma)

        f = open(path_f, "w+")
        f.truncate(0)

        g = open(path_g, "w+")
        g.truncate(0)

        for _f_data in self._f_data:
            for __f_data in _f_data:
                for ___f_data in __f_data:
                    f.write(str(___f_data) + " ")

        for _g_data in self._g_data:
            for __g_data in _g_data:
                for ___g_data in __g_data:
                    g.write(str(___g_data) + " ")


    def print(self):
        n_sigma = self.get_n_sigma()
        z = np.arange(-n_sigma, n_sigma + 1)
        for k in z:
            for n in z:
                for m in z:
                    print(k, m, n, self.get(k, m, n))