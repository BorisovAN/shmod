import numpy as np
from scipy.fft import fft2, ifft2
from scipy.signal import convolve2d
import cv2
from matching.detectors.rift.lowpassfilter import lowpassfilter

def phasecong3(im, nscale=4, norient=6, minWaveLength=3, mult=2.1, sigmaOnf=0.55,
               k=2.0, cutOff=0.5, g=10, noiseMethod=-1):
    if len(im.shape) > 2:
        im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
        im = im.astype(np.float32)

    epsilon = .0001
    rows, cols = im.shape
    imagefft = fft2(im)


    zero = np.zeros((rows, cols), dtype=np.float32)
    EO = [[None for _ in range(norient)] for _ in range(nscale)]
    PC = [None for _ in range(norient)]
    covx2 = zero.copy()
    covy2 = zero.copy()
    covxy = zero.copy()

    EnergyV = np.zeros((rows, cols, 3), dtype=np.float32)
    pcSum = zero.copy()

    # Set up X and Y matrices with ranges normalized to +/- 0.5
    if cols % 2:
        xrange = np.arange(-(cols - 1) / 2, (cols - 1) / 2 + 1, dtype=np.float32) / (cols - 1)
    else:
        xrange = np.arange(-cols / 2, cols / 2, dtype=np.float32) / (cols)

    if rows % 2:
        yrange = np.arange(-(rows - 1) / 2, (rows - 1) / 2 + 1, dtype=np.float32) / (rows - 1)
    else:
        yrange = np.arange(-rows / 2, rows / 2, dtype=np.float32) / (rows)

    x, y = np.meshgrid(xrange, yrange)

    radius = np.sqrt(x ** 2 + y ** 2)
    theta = np.arctan2(-y, x)

    radius = np.fft.ifftshift(radius)
    theta = np.fft.ifftshift(theta)
    radius[0, 0] = 1

    sintheta = np.sin(theta)
    costheta = np.cos(theta)

    # Construct the radial filter components
    lp = lowpassfilter((rows, cols), .45, 15)

    logGabor = [None] * nscale

    for s in range(nscale):
        wavelength = minWaveLength * mult ** (s)
        fo = 1.0 / wavelength
        logGabor[s] = np.exp((-(np.log(radius / fo)) ** 2) / (2 * np.log(sigmaOnf) ** 2))
        logGabor[s] = logGabor[s] * lp
        logGabor[s][0, 0] = 0

    # The main loop...
    for o in range(norient):
        angl = o * np.pi / norient
        ds = sintheta * np.cos(angl) - costheta * np.sin(angl)
        dc = costheta * np.cos(angl) + sintheta * np.sin(angl)
        dtheta = np.abs(np.arctan2(ds, dc))
        dtheta = np.minimum(dtheta * norient / 2, np.pi)
        spread = (np.cos(dtheta) + 1) / 2

        sumE_ThisOrient = zero.copy()
        sumO_ThisOrient = zero.copy()
        sumAn_ThisOrient = zero.copy()
        Energy = zero.copy()

        for s in range(nscale):
            filter = logGabor[s] * spread
            EO[s][o] = ifft2(imagefft * filter)

            An = np.abs(EO[s][o])
            sumAn_ThisOrient += An
            sumE_ThisOrient += np.real(EO[s][o])
            sumO_ThisOrient += np.imag(EO[s][o])

            if s == 0:
                if noiseMethod == -1:
                    tau = np.median(sumAn_ThisOrient) / np.sqrt(np.log(4))
                elif noiseMethod == -2:
                    tau = rayleighmode(sumAn_ThisOrient.flatten())
                maxAn = An
            else:
                maxAn = np.maximum(maxAn, An)

        EnergyV[:, :, 0] += sumE_ThisOrient
        EnergyV[:, :, 1] += np.cos(angl) * sumO_ThisOrient
        EnergyV[:, :, 2] += np.sin(angl) * sumO_ThisOrient

        XEnergy = np.sqrt(sumE_ThisOrient ** 2 + sumO_ThisOrient ** 2) + epsilon
        MeanE = sumE_ThisOrient / XEnergy
        MeanO = sumO_ThisOrient / XEnergy

        for s in range(nscale):
            E = np.real(EO[s][o])
            O = np.imag(EO[s][o])
            Energy += E * MeanE + O * MeanO - np.abs(E * MeanO - O * MeanE)

        if noiseMethod >= 0:
            T = noiseMethod
        else:
            totalTau = tau * (1 - (1 / mult) ** nscale) / (1 - (1 / mult))
            EstNoiseEnergyMean = totalTau * np.sqrt(np.pi / 2)
            EstNoiseEnergySigma = totalTau * np.sqrt((4 - np.pi) / 2)
            T = EstNoiseEnergyMean + k * EstNoiseEnergySigma

        Energy = np.maximum(Energy - T, 0)
        width = (sumAn_ThisOrient / (maxAn + epsilon) - 1) / (nscale - 1)
        weight = 1.0 / (1 + np.exp((cutOff - width) * g))

        PC[o] = weight * Energy / sumAn_ThisOrient

        pcSum += PC[o]

        covx = PC[o] * np.cos(angl)
        covy = PC[o] * np.sin(angl)
        covx2 += covx ** 2
        covy2 += covy ** 2
        covxy += covx * covy

    covx2 = covx2 / (norient / 2)
    covy2 = covy2 / (norient / 2)
    covxy = 4 * covxy / norient
    denom = np.sqrt(covxy ** 2 + (covx2 - covy2) ** 2) + epsilon
    M = (covy2 + covx2 + denom) / 2
    m = (covy2 + covx2 - denom) / 2

    or_ = np.arctan2(EnergyV[:, :, 2], EnergyV[:, :, 1])
    or_[or_ < 0] += np.pi
    or_ = or_ * 180 / np.pi

    OddV = np.sqrt(EnergyV[:, :, 1] ** 2 + EnergyV[:, :, 2] ** 2)
    featType = np.arctan2(EnergyV[:, :, 0], OddV)

    return M, m, or_, featType, PC, EO, T, pcSum


def rayleighmode(data, nbins=50):
    mx = np.max(data)
    edges = np.linspace(0, mx, nbins + 1, dtype=np.float32)
    n, _ = np.histogram(data, edges)
    ind = np.argmax(n)
    return (edges[ind] + edges[ind + 1]) / 2
