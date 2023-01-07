import numpy as np
from scipy import constants as const


class C:
    def __init__(self):
        self.nAir = 1.0003
        self.nGe = 3.979
        self.nSi = 3.425
        self.sigmaSi = 1.67E-2
        #self.nSiO2 = 1.4780
        self.nSiO2 = 1.4780
        self.sigmaSiO2 = 1.0E-6
        self.Wavelength = 4250  # [nm]
        self.c0 = const.speed_of_light

    def gridRes(self, fmax, Nlambda=40):
        nmax = max([self.nAir, self.nGe, self.nSi])
        lambdaMin = self.c0 / fmax * nmax
        deltaLambda = lambdaMin / Nlambda
        return deltaLambda

    def minFeat(self, dmin, Nd=4):
        deltad = dmin / Nd
        return deltad

    def deltamin(self, deltax, deltay, deltaz):
        deltamin = min([deltax, deltay, deltaz])
        return deltamin

    def deltat(self, deltax, deltay, deltaz, nmin):
        deltat = nmin / (np.sqrt(1 / deltax + 1 / deltay + 1 / deltaz) * 2 * self.c0)
        return deltat

    def refIdx(self, ur, er):
        n = np.sqrt(ur * const.mu_0 * er * const.epsilon_0)
        return n

    def epislon_r(self, n):
        epsilon_r = n ** 2
        return epsilon_r

    @staticmethod
    def Impedance(er, ur):
        Imp = np.sqrt(ur / er) * np.sqrt(const.mu_0 / const.epsilon_0)
        return Imp

    def WaveVelocity(self, n):
        v = self.c0 / n
        return v
