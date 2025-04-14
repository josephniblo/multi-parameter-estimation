import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tkinter import Tk
from tkinter.filedialog import askopenfilename

# Hide the main Tkinter window
Tk().withdraw()

# Prompt user to select a CSV file
file_path = askopenfilename(filetypes=[("CSV files", "*.csv")])
if not file_path:
    raise ValueError("No file selected. Please select a valid CSV file.")

dfFOUR = pd.read_csv(file_path)


MCITER = 2000
state = "Plus"
# dfFOUR = pd.read_csv('Sep_S2_polHOM_FG1_2025-03-15--17h-40m_2025-03-15--17h-44m_100mW.csv')

df = dfFOUR

# Columns to sum for coincidences

coincidences_columns = ["TT"]

# Check if all specified columns exist in the data
missing_columns = [col for col in coincidences_columns if col not in df.columns]
if missing_columns:
    raise ValueError(
        f"The following columns are missing in the data: {missing_columns}"
    )

# Calculate coincidences by summing the specified columns
df["0dd_coincidences"] = df[coincidences_columns].sum(axis=1)

fold = 4
angles = np.linspace(-30, 30, len(df))
columnsFOUR = list(dfFOUR.columns.values)
sampleArr = []
for k in range(0, MCITER):
    e = []
    for i in range(0, len(df["angle"])):
        s = np.random.poisson(int(df["0dd_coincidences"][i]), 1)
        e.append(s.tolist()[0])
    sampleArr.append(pd.Series(e, index=angles))
MONTE_Err = []
for j in angles:
    p = []
    for ITER in range(0, MCITER):
        p.append(sampleArr[ITER][j])
    MONTE_Err.append(np.std(p))
# plt.scatter(dfFOUREdit['angle'],dfFOUREdit['Summed'])
# plt.errorbar(dfFOUREdit['angle'],dfFOUREdit['Summed'], yerr=MONTE_Err)

testset = sum(sampleArr) / len(sampleArr)
import numpy, scipy.optimize


def fit_sin(tt, yy):
    '''Fit sin to the input time sequence, and return fitting parameters "amp", "omega", "phase", "offset", "freq", "period" and "fitfunc"'''
    tt = numpy.array(tt)
    yy = numpy.array(yy)
    ff = numpy.fft.fftfreq(len(tt), (tt[1] - tt[0]))  # assume uniform spacing
    Fyy = abs(numpy.fft.fft(yy))
    guess_freq = abs(
        ff[numpy.argmax(Fyy[1:]) + 1]
    )  # excluding the zero frequency "peak", which is related to offset
    guess_amp = numpy.std(yy) * 2.0**0.5
    guess_offset = numpy.mean(yy)
    guess = numpy.array([guess_amp, 2.0 * numpy.pi * guess_freq, 0.0, guess_offset])

    def sinfunc(t, A, w, p, c):
        return A * numpy.sin(w * t + p) + c

    popt, pcov = scipy.optimize.curve_fit(sinfunc, tt, yy, p0=guess)
    A, w, p, c = popt
    f = w / (2.0 * numpy.pi)
    fitfunc = lambda t: A * numpy.sin(w * t + p) + c

    res_s = np.dot((yy - sinfunc(tt, *popt)), (yy - sinfunc(tt, *popt)))
    yMean = np.mean(yy)
    tot = np.dot((yy - yMean), (yy - yMean))

    # calculate r-squared = 1 - (residual sum of squares) / (total sum of squares)
    r_sqr_s = 1 - res_s / tot
    return {
        "amp": A,
        "omega": w,
        "phase": p,
        "offset": c,
        "freq": f,
        "period": 1.0 / f,
        "fitfunc": fitfunc,
        "maxcov": numpy.max(pcov),
        "rawres": (guess, popt, pcov),
        "res_s": res_s,
        "r_sqr_s": r_sqr_s,
    }


def plot_cfit(xData, yData, cfitData, nPoints):
    # parse raw data structures
    x = xData
    y = yData
    # parse curve-fit data structures
    cfit = cfitData
    n = nPoints

    # plot raw data
    # datPlot, = plt.plot(x, y, 'b.')

    # plot curvefit
    (cfitPlot,) = plt.plot(np.linspace(min(x), max(x), n), cfit, "r-")

    # draw confidence region around curvefit defined by one-sigma assuming Poissonian statistics
    plt.fill_between(
        np.linspace(min(x), max(x), n),
        cfit - np.sqrt(cfit),
        cfit + np.sqrt(cfit),
        alpha=0.3,
        color="r",
    )

    return cfitPlot


def plot_cfit2(xData, yData, cfitData, nPoints, monte):
    # parse raw data structures
    x = xData
    y = yData
    # parse curve-fit data structures
    cfit = cfitData
    n = nPoints

    # plot raw data
    # datPlot, = plt.plot(x, y, 'b.')

    # plot curvefit
    (cfitPlot,) = plt.plot(np.linspace(min(x), max(x), n), cfit, "r-")

    # draw confidence region around curvefit defined by one-sigma assuming Poissonian statistics
    plt.fill_between(
        np.linspace(min(x), max(x), n), cfit - monte, cfit + monte, alpha=0.3, color="y"
    )

    return cfitPlot


def plot_cfit3(xData, yData, cfitData, nPoints, monte):
    # parse raw data structures
    x = xData
    y = yData
    # parse curve-fit data structures
    cfit = cfitData
    n = nPoints

    # plot raw data
    # datPlot, = plt.plot(x, y, 'b.')

    # plot curvefit
    (cfitPlot,) = plt.plot(np.linspace(min(x), max(x), n), cfit, "r-")

    # draw confidence region around curvefit defined by one-sigma assuming Poissonian statistics
    plt.fill_between(
        np.linspace(min(x), max(x), n), cfit - monte, cfit + monte, alpha=0.3, color="y"
    )

    return cfit


# EDIT THIS TO CHANGE FIT
import pylab as plt

# N, amp, omega, phase, offset, noise = 1, 1., 1., 1., 0, 0
# tt = dfFOUREdit['angle']
# tt2 = dfFOUREdit['angle']*N
# yy=testset

# REAL DATA
N, amp, omega, phase, offset, noise = 1, 1.0, 1.0, 1.0, 0, 0
tt = df["angle"]
tt2 = df["angle"] * N
yy = df["0dd_coincidences"]


res = fit_sin(tt, yy)
print(
    "Amplitude=%(amp)s, Angular freq.=%(omega)s, phase=%(phase)s, offset=%(offset)s, Max. Cov.=%(maxcov)s"
    % res
)
"""
calculate visibilities based on optimal parameters (popt) of each cfit
working definitions (because min is the unknown parameter to be found):
Formally, vis = (max - min) / (max)
max = bkgd
ampl = bkgd - min
vis* = ampl / bkgd
"""
if state == "Plus":
    visi = -1 * res["amp"] / res["rawres"][1][-1]
if state == "Minus":
    visi = 1 * res["amp"] / res["rawres"][1][-1]
perr_s = np.sqrt(np.diag(res["rawres"][2]))
stdErr_vis_s = (
    (perr_s[0] / res["amp"]) ** 2 + (perr_s[3] / res["rawres"][1][-1]) ** 2
) ** 0.5 * visi

plt.errorbar(
    dfFOUR["angle"], df["0dd_coincidences"], yerr=MONTE_Err, ls="None", fmt="None"
)
plt.scatter(tt, df["0dd_coincidences"], label="Data", s=4)
# plt.plot(tt*N, res["fitfunc"](tt*N), "r-", label="curve fit", linewidth=2)
plot_cfit(tt, yy, res["fitfunc"](tt), len(dfFOUR))
# plot_cfit2(tt, yy, res["fitfunc"](tt), 31, MONTE_Err)
# plt.legend(loc="best")
plt.xlabel("angle")
plt.ylabel("Fourfolds")
plt.title(
    "Visi (\%): " + str(round(visi * 100, 4)) + "+-" + str(round(stdErr_vis_s * 100, 4))
)
plt.show()

import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt


# Define a sine function for fitting
def sin_func(x, A, omega, phase, C):
    return A * np.sin(omega * x + phase) + C


# Fit the sine function to the data
def fit_sin(tt, yy):
    """Fit sine function to data and return fitting parameters"""
    tt = np.array(tt)
    yy = np.array(yy)

    # Initial parameter guesses
    guess_freq = 2 * np.pi / (max(tt) - min(tt))  # Rough estimate based on range
    guess_amp = (max(yy) - min(yy)) / 2
    guess_offset = np.mean(yy)
    guess_phase = 0  # Initial phase guess

    # Fit the function using curve_fit
    popt, pcov = scipy.optimize.curve_fit(
        sin_func, tt, yy, p0=[guess_amp, guess_freq, guess_phase, guess_offset]
    )

    A, omega, phase, C = popt  # Extract fitted parameters
    return A, omega, phase, C, popt


# Find the minimum using SciPy's minimization method
def find_minimum(fit_params, x_range):
    """Find the minimum of the fitted sine wave"""
    A, omega, phase, C = fit_params

    # Define function to minimize
    def neg_sin(x):
        return sin_func(x, A, omega, phase, C)

    # Minimize the function over the given x_range
    result = scipy.optimize.minimize_scalar(
        neg_sin, bounds=(min(x_range), max(x_range)), method="bounded"
    )
    return result.x  # Minimum x value


# Load data
tt = df["angle"]
yy = df["0dd_coincidences"]

# Perform the fit
A, omega, phase, C, popt = fit_sin(tt, yy)

# Find the minimum of the fitted sine wave
dip_position = find_minimum(popt, tt)
print(f"Minimum position (dip): {dip_position:.4f} degrees")

# Plot the data and fitted curve
plt.errorbar(tt, yy, yerr=MONTE_Err, ls="None", fmt="o", label="Data")
plt.plot(tt, sin_func(tt, A, omega, phase, C), "r-", label="Fit", linewidth=2)

# Mark the dip position on the plot
plt.axvline(
    dip_position, color="blue", linestyle="--", label=f"Dip at {dip_position:.2f}°"
)
plt.text(
    dip_position,
    np.min(yy),
    f"  {dip_position:.2f}°",
    color="blue",
    verticalalignment="bottom",
)

# Labels and title
plt.xlabel("Angle (degrees)")
plt.ylabel("Fourfold Coincidences")
plt.title("Fitted Sinusoidal Curve with Dip Location")
plt.legend()
plt.show()
