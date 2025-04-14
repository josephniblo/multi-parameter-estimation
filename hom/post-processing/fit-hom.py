# -*- coding: utf-8 -*-
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pandas as pd


class fitHOM:
    def __init__(self, position, coincidences):
        xData = position
        yData = coincidences

        if len(xData) != len(yData):
            print("X and Y datasets have unmatched lengths")

        # guess starting parameters from input data for Levenberg-Marquardt algorithm
        bkgd_guess = np.mean(
            np.sort(yData)[int(round(0.90 * len(yData))) : :]
        )  # average 90 percentile largest values
        xpos_guess = xData[
            np.argmin(yData)
        ]  # use min(coincidences) index to guess dip position
        sigm_guess = 0.8  # typical value, no obvious way to guess from dataset
        ampl_guess = bkgd_guess - min(yData)  # basically max-min
        # additional fit function parameters to account for drifts in count rates
        slope_guess = (yData[0] - yData[-1]) / (
            xData[0] - xData[-1]
        )  # slope from rise over run, first and final data point
        yint_guess = (
            yData[0] - slope_guess * xData[0]
        )  # y-intercept extrapolated from guessed slope and first data point

        # scipy.optimize.curve_fit for each 'static' function defined after main code
        # standard Gaussian: _g
        param_guess_g = (ampl_guess, xpos_guess, sigm_guess, bkgd_guess)
        popt_g, pcov_g = curve_fit(self.func_gaus, xData, yData, p0=param_guess_g)
        perr_g = np.sqrt(np.diag(pcov_g))

        # Gaussian+linear: _g1
        param_guess_g1 = (ampl_guess, xpos_guess, sigm_guess, slope_guess, yint_guess)
        popt_g1, pcov_g1 = curve_fit(self.func_gaus1, xData, yData, p0=param_guess_g1)
        perr_g1 = np.sqrt(np.diag(pcov_g1))

        # stndard triangular: _t
        param_guess_t = (ampl_guess, xpos_guess, sigm_guess, sigm_guess, bkgd_guess)
        popt_t, pcov_t = curve_fit(self.func_tri, xData, yData, p0=param_guess_t)
        perr_t = np.sqrt(np.diag(pcov_t))

        # triangle+linear: _t1
        param_guess_t1 = (
            ampl_guess,
            xpos_guess,
            sigm_guess,
            sigm_guess,
            slope_guess,
            yint_guess,
        )
        popt_t1, pcov_t1 = curve_fit(self.func_tri1, xData, yData, p0=param_guess_t1)

        # calculate visibilities based on optimal parameters (popt) of each cfit
        # working definitions (because min is the unknown parameter to be found):
        # Formally, vis = (max - min) / (max)
        # max = bkgd
        # ampl = bkgd - min
        # vis* = ampl / bkgd
        vis_g = popt_g[0] / popt_g[3]
        y0max_g1 = (
            popt_g1[3] * popt_g1[1] + popt_g1[4]
        )  # this parameter will be used multiple times
        vis_g1 = popt_g1[0] / y0max_g1  # vis calculated from the slope at x(yMin)
        vis_t = popt_t[0] / popt_t[4]
        y0max_t1 = popt_t1[4] * popt_t1[1] + popt_t1[5]
        vis_t1 = popt_t1[0] / y0max_t1

        # calculate uncertainty in visibilities from covariance matrix of fits
        # diagonal elements correspond to standard error of the optimised fit parameter

        # calculate residual sum of squares (from c-fit function)
        res_g = np.dot(
            (yData - self.func_gaus(xData, *popt_g)),
            (yData - self.func_gaus(xData, *popt_g)),
        )
        res_g1 = np.dot(
            (yData - self.func_gaus1(xData, *popt_g1)),
            (yData - self.func_gaus1(xData, *popt_g1)),
        )
        res_t = np.dot(
            (yData - self.func_tri(xData, *popt_t)),
            (yData - self.func_tri(xData, *popt_t)),
        )
        res_t1 = np.dot(
            (yData - self.func_tri1(xData, *popt_t1)),
            (yData - self.func_tri1(xData, *popt_t1)),
        )
        # calculate total sum of squares (from raw dataset, same for all curvefits)
        yMean = np.mean(yData)
        tot = np.dot((yData - yMean), (yData - yMean))
        # calculate r-squared = 1 - (residual sum of squares) / (total sum of squares)
        r_sqr_t = 1 - res_t / tot
        r_sqr_g = 1 - res_g / tot
        r_sqr_g1 = 1 - res_g1 / tot
        r_sqr_t1 = 1 - res_t1 / tot
        # print a message in terminal with r-squared coefficients using each fit function
        print(
            "R^2 for Gaussian: %.4f and Triangle: %.4f and Gaussian+Linear: %.4f and Triangle+Linear: %.4f"
            % (float(r_sqr_g), float(r_sqr_t), float(r_sqr_g1), float(r_sqr_t1))
        )

        # generate curvefit datapoints
        nPoints = 500
        cfit_g = self.func_gaus(np.linspace(min(xData), max(xData), nPoints), *popt_g)
        cfit_g1 = self.func_gaus1(
            np.linspace(min(xData), max(xData), nPoints), *popt_g1
        )
        cfit_t = self.func_tri(np.linspace(min(xData), max(xData), nPoints), *popt_t)
        cfit_t1 = self.func_tri1(np.linspace(min(xData), max(xData), nPoints), *popt_t1)

        # plot each curve-fit with raw data
        plt.figure(
            "Gaussian fit", figsize=(8, 6), dpi=200, facecolor="w", edgecolor="k"
        )
        plt.ion()  # enables interactive mode to prevent terminal block after each plot, renable to hold before final plot
        dat_g, cfit_g = self.plot_cfit(xData, yData, cfit_g, nPoints)
        plt.legend(
            (dat_g, cfit_g),
            ("data", "fit: ampl=%.0f, xpos=%.4f, sigm=%.4f, bkgd=%.0f" % tuple(popt_g)),
        )
        self.format_cfit(r_sqr_g, vis_g)

        plt.figure(
            "Gaussian+Linear fit", figsize=(8, 6), dpi=200, facecolor="w", edgecolor="k"
        )
        plt.savefig("Gaussian+Linear.png")

    def plot_cfit(self, xData, yData, cfitData, nPoints):
        # parse raw data structures
        x = xData
        y = yData
        # parse curve-fit data structures
        cfit = cfitData
        n = nPoints

        # plot raw data
        (datPlot,) = plt.plot(x, y, "b.")

        # plot curvefit
        (cfitPlot,) = plt.plot(np.linspace(min(x), max(x), n), cfit, "r-")

        # draw psudo-confidence region around curvefit defined by Poissonian statistics
        plt.fill_between(
            np.linspace(min(x), max(x), n),
            cfit - np.sqrt(cfit),
            cfit + np.sqrt(cfit),
            alpha=0.3,
            color="r",
        )

        return datPlot, cfitPlot

    def format_cfit(self, r_sqr, vis_calc):
        R_sqr = r_sqr
        vis = vis_calc
        # general formatting
        # plt.figure(figsize=(8, 6), dpi=200, facecolor='w', edgecolor='k')
        plt.suptitle(
            "R^2: %4f, Fitted vis: %.4f" % (float(R_sqr), float(vis)), fontsize=20
        )
        plt.xlabel("position", fontsize=16)
        plt.ylabel("coincidences", fontsize=16)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.show()

    @staticmethod
    def func_gaus(x, ampl, xpos, sigm, bkgd):
        # define standard Gaussian function
        return -1 * ampl * np.exp(-((x - xpos) ** 2) / (2.0 * sigm**2)) + bkgd

    @staticmethod
    def func_tri(x, ampl, xpos, sigm_L, sigm_R, bkgd):
        # define triangular function
        ret = []
        for xx in x:
            if xx < (xpos - sigm_L):
                ret.append(bkgd)
            elif (xpos - sigm_L) <= xx < xpos:
                ret.append(
                    bkgd - ((xx - (xpos - sigm_L)) / (xpos - (xpos - sigm_L)) * ampl)
                )
            elif xpos <= xx < (xpos + sigm_R):
                ret.append(
                    bkgd - (((xpos + sigm_R) - xx) / ((xpos + sigm_R) - xpos)) * ampl
                )
            elif xx >= (xpos + sigm_R):
                ret.append(bkgd)
        return np.array(ret)

    @staticmethod
    def func_gaus1(x, ampl, xpos, sigm, slope, yint):
        # define a Gaussian function with a slope function
        return (-1 * ampl * np.exp(-((x - xpos) ** 2) / (2.0 * sigm**2))) + (
            slope * x + yint
        )

    @staticmethod
    def func_tri1(x, ampl, xpos, sigL, sigR, slope, yint):
        # define triangular function
        ret = []
        for xx in x:
            if xx < (xpos - sigL):
                ret.append(slope * xx + yint)
            elif (xpos - sigL) <= xx < xpos:
                ret.append(
                    (slope * xx + yint)
                    - ((xx - (xpos - sigL)) / (xpos - (xpos - sigL)) * ampl)
                )
            elif xpos <= xx < (xpos + sigR):
                ret.append(
                    (slope * xx + yint)
                    - (((xpos + sigR) - xx) / ((xpos + sigR) - xpos)) * ampl
                )
            elif xx >= (xpos + sigR):
                ret.append(slope * xx + yint)
        return np.array(ret)

    @staticmethod
    def func_linear(x, slope, yint):
        # define standard linear gradient function
        return slope * x + yint


if __name__ == "__main__":
    from tkinter import Tk
    from tkinter.filedialog import askopenfilename

    Tk().withdraw()
    # await user to select a file
    print("Select a .csv file to load")
    f_name = ""
    while f_name == "":
        f_name = askopenfilename()

    # check file is csv, later on can include more file types for loading
    fName = f_name
    fName = str(fName)
    if str(fName)[-4::] == ".csv":
        print("File selected: %s" % fName)
    else:
        print("Select a valid .csv file")
        sys.exit()

    df = pd.read_csv(fName)

    # check csv file contains column headers, 'position' and multiple potential coincidences columns
    try:
        position = df["position"]
    except KeyError:
        while True:
            try:
                xString = input(
                    "Specify scan position column name %s: " % str(tuple(df.columns))
                )
                position = df[xString]
                break
            except SyntaxError:
                print("Enter the string with quotation marks")

    # List of potential coincidences column names
    coincidences_columns = ["TT"]  # Sep source ["cc"]

    # Check which of the specified columns exist in the dataframe
    valid_columns = [col for col in coincidences_columns if col in df.columns]

    if (
        not valid_columns
    ):  # If none of the columns are found, prompt the user to specify
        while True:
            try:
                yString = input(
                    "Specify coincidences column name(s), separated by commas: %s: "
                    % str(tuple(df.columns))
                )
                valid_columns = [
                    col.strip()
                    for col in yString.split(",")
                    if col.strip() in df.columns
                ]
                if valid_columns:
                    break
                else:
                    print("No valid columns specified. Try again.")
            except SyntaxError:
                print("Enter the string with quotation marks.")

    # Sum the data from the selected columns to create a new coincidences column
    if valid_columns:
        df["coincidences"] = df[valid_columns].sum(axis=1)
        print(f"Summed columns: {valid_columns} into 'coincidences'.")
    else:
        raise ValueError("No valid columns for coincidences were found or specified.")

    # for easier handling of data will format into numpy.array
    position = np.array(position)
    coincidences = np.array(df["coincidences"])

    # run main script
    fitHOM(position, coincidences)

    # hold final plot open until closed by user
    plt.ioff()  # disable interactive mode to hold final plot open
