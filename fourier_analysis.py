import numpy as np
import struct


class FourierAnalysis:
    _mu = 0.75 / np.pi ** 3  # equation (1.1.1)
    C000 = 0.75 / np.pi * np.log(3 + 2 * np.sqrt(2))
    _a = np.pi / 4.0  # equation(2.3.1)
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

    def __init__(self, n_beta, n_sigma):
        self._n_beta = n_beta  # equation(2.3.3)
        self._d_beta = FourierAnalysis.get_d_beta(n_beta)  # equation (2.3.4)

        self._n_sigma = n_sigma
        self._z = self._get_z(n_sigma)
        self._z_0 = self._get_z_0(n_sigma)
        self._z_plus = self._get_z_plus(n_sigma)

        self._calculate_values_of_trigonometric_functions()

        z = self.get_z()
        self._F = np.full((z.size, z.size, z.size), 0.0)
        self._G = np.full((z.size, z.size, z.size), 0.0)

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

    def get_F(self, k, n, m):  # equation (3.1.3)
        n_sigma = self.get_n_sigma()  # equation (3.3.3)
        return self._F[k + n_sigma][n + n_sigma][m + n_sigma]

    def get_G(self, k, n, m):  # equation (3.1.3)
        n_sigma = self.get_n_sigma()  # equation (3.3.3)
        return self._G[k + n_sigma][n + n_sigma][m + n_sigma]

    def _calculate_values_of_trigonometric_functions(self):
        # this method precalculates all the required values of the trigonometric functions
        n_beta = self.get_n_beta()  # from equation (1)

        self._sin_beta = np.full(n_beta, 0.0)
        for i in np.arange(n_beta):
            self._sin_beta[i] = np.sin(self.get_beta(i))

        self._cos_beta = np.full(n_beta, 0.0)
        for i in np.arange(n_beta):
            self._cos_beta[i] = np.cos(self.get_beta(i))

        self._tan_beta = np.full(n_beta, 0.0)
        for i in np.arange(n_beta):
            self._tan_beta[i] = np.tan(self.get_beta(i))

    def generate(self):  # equation (1)
        # this method precalculates all the required values of the C coefficient
        n_sigma = self.get_n_sigma()  # equation (3.3.4)
        z_0 = self.get_z_0()  # equation (3.3.2)
        z_plus = self.get_z_plus()  # equation (3.3.1)
        self._F = np.full((2 * n_sigma + 1, 2 * n_sigma + 1, 2 * n_sigma + 1), 0.0)  # equation (3.1.3)
        self._G = np.full((2 * n_sigma + 1, 2 * n_sigma + 1, 2 * n_sigma + 1), 0.0)  # equation (3.1.3)
        for k in z_0:
            for n in z_0:
                for m in z_plus:
                    self._set_C(k, n, m, self.numerically_integrate(k, n, m))

        for k in z_0:
            for n in z_plus:
                self._set_C(k, n, 0, self.numerically_integrate(k, n, 0))

        for k in z_0:
            self._set_C(k, 0, 0, self.numerically_integrate(k, 0, 0))

        self._set_C(0, 0, 0, (self.C000, 0.0))

        return self

    def _set_C(self, k, n, m, values):  # equation (1)
        n_sigma = self.get_n_sigma()
        self._F[k + n_sigma][n + n_sigma][m + n_sigma] = values[0]
        self._G[k + n_sigma][n + n_sigma][m + n_sigma] = values[1]
        self._F[-k + n_sigma][-n + n_sigma][-m + n_sigma] = values[0]
        self._G[-k + n_sigma][-n + n_sigma][-m + n_sigma] = -values[1]

    def _set_F(self, k, n, m, value):
        n_sigma = self.get_n_sigma()
        self._F[k + n_sigma][n + n_sigma][m + n_sigma] = value

    def _set_G(self, k, n, m, value):
        n_sigma = self.get_n_sigma()
        self._G[k + n_sigma][n + n_sigma][m + n_sigma] = value

    @staticmethod
    def _float_to_binary(num):
        return bin(struct.unpack('!I', struct.pack('!f', num))[0])[2:].zfill(32)

    @staticmethod
    def _binary_to_float(binary):
        return struct.unpack('!f', struct.pack('!I', int(binary, 2)))[0]

    def unload(self, path):
        n_beta = self.get_n_beta()
        n_sigma = self.get_n_sigma()
        file = open(path, "w+")
        file.truncate(0)
        file.write(str(n_beta))
        file.write(" ")
        file.write(str(n_sigma))
        file.write(" ")
        for _f in self._F:
            for _ff in _f:
                for _fff in _ff:
                    file.write(self._float_to_binary(_fff))
                    file.write(" ")

        for _g in self._G:
            for _gg in _g:
                for _ggg in _gg:
                    file.write(self._float_to_binary(_ggg))
                    file.write(" ")

    def _load(self, data):
        n_beta = self.get_n_beta()
        n_sigma = self.get_n_sigma()
        z = self.get_z()
        i = 2
        for k in z:
            for n in z:
                for m in z:
                    print(type(self._binary_to_float(data[i])))
                    self._set_F(k, n, m, self._binary_to_float(data[i]))
                    i += 1

        for k in z:
            for n in z:
                for m in z:
                    self._set_G(k, n, m, self._binary_to_float(data[i]))
                    i += 1

        return self

    @staticmethod
    def load(path):
        file = open(path)
        data = file.read().split(" ")
        n_beta = int(data[0])
        n_sigma = int(data[1])
        return FourierAnalysis(n_beta, n_sigma)._load(data)

    @staticmethod
    def _get_z(n_sigma):
        return np.arange(-n_sigma, n_sigma + 1)

    @staticmethod
    def _get_z_0(n_sigma):
        z_0 = []
        for sigma in np.arange(-n_sigma, n_sigma + 1):
            if sigma == 0:
                continue

            z_0.append(sigma)

        z_0 = np.array(z_0)
        return z_0

    @staticmethod
    def _get_z_plus(n_sigma):
        z_plus = []
        for sigma in np.arange(1, n_sigma + 1):
            z_plus.append(sigma)

        z_plus = np.array(z_plus)
        return z_plus