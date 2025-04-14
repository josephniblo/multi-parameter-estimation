# -*- coding: utf-8 -*-
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pandas as pd
import argparse  # Import argparse for command-line argument parsing


class FitHOM:
    @staticmethod
    def fit_and_plot(position, coincidences, output_dir):
        x_data = position
        y_data = coincidences

        if len(x_data) != len(y_data):
            raise ValueError("X and Y datasets have unmatched lengths")

        # Guess starting parameters for the Levenberg-Marquardt algorithm
        bkgd_guess = np.mean(np.sort(y_data)[int(round(0.90 * len(y_data)))::])
        xpos_guess = x_data[np.argmin(y_data)]
        sigm_guess = 0.8
        ampl_guess = bkgd_guess - min(y_data)
        slope_guess = (y_data[0] - y_data[-1]) / (x_data[0] - x_data[-1])
        yint_guess = y_data[0] - slope_guess * x_data[0]

        # Fit Gaussian
        param_guess_g = (ampl_guess, xpos_guess, sigm_guess, bkgd_guess)
        popt_g, pcov_g = curve_fit(FitHOM.func_gaussian, x_data, y_data, p0=param_guess_g)
        r_squared_g = FitHOM.calculate_r_squared(y_data, FitHOM.func_gaussian(x_data, *popt_g))
        vis_g = popt_g[0] / popt_g[3]

        # Fit Gaussian+Linear
        param_guess_g1 = (ampl_guess, xpos_guess, sigm_guess, slope_guess, yint_guess)
        popt_g1, pcov_g1 = curve_fit(FitHOM.func_gaussian_linear, x_data, y_data, p0=param_guess_g1)
        r_squared_g1 = FitHOM.calculate_r_squared(y_data, FitHOM.func_gaussian_linear(x_data, *popt_g1))
        y0max_g1 = popt_g1[3] * popt_g1[1] + popt_g1[4]
        vis_g1 = popt_g1[0] / y0max_g1

        # Generate and save plots
        os.makedirs(output_dir, exist_ok=True)  # Ensure the output directory exists
        FitHOM.plot_fit(
            x_data, y_data, FitHOM.func_gaussian, popt_g, r_squared_g, vis_g,
            "Gaussian Fit", os.path.join(output_dir, "Gaussian_fit.png")
        )
        FitHOM.plot_fit(
            x_data, y_data, FitHOM.func_gaussian_linear, popt_g1, r_squared_g1, vis_g1,
            "Gaussian+Linear Fit", os.path.join(output_dir, "Gaussian_Linear_fit.png")
        )

    @staticmethod
    def plot_fit( x_data, y_data, func, popt, r_squared, visibility, title, filename):
        """Generate and save a plot for a given fit."""
        n_points = 500
        x_fit = np.linspace(min(x_data), max(x_data), n_points)
        y_fit = func(x_fit, *popt)

        plt.figure(title, figsize=(8, 6), dpi=200, facecolor="w", edgecolor="k")
        plt.plot(x_data, y_data, "b.", label="Data")
        plt.plot(x_fit, y_fit, "r-", label="Fit")
        plt.fill_between(
            x_fit, y_fit - np.sqrt(y_fit), y_fit + np.sqrt(y_fit), alpha=0.3, color="r"
        )
        plt.title(f"{title}\nR²: {r_squared:.4f}, Visibility: {visibility:.4f}", fontsize=16)
        plt.xlabel("Position", fontsize=14)
        plt.ylabel("Coincidences", fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True)
        plt.savefig(filename, dpi=200, bbox_inches="tight")
        plt.close()

    @staticmethod
    def calculate_r_squared(y_data, y_fit):
        """Calculate the R² value for a fit."""
        ss_res = np.sum((y_data - y_fit) ** 2)
        ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
        return 1 - (ss_res / ss_tot)

    @staticmethod
    def func_gaussian(x, ampl, xpos, sigm, bkgd):
        """Define a standard Gaussian function."""
        return -1 * ampl * np.exp(-((x - xpos) ** 2) / (2.0 * sigm**2)) + bkgd

    @staticmethod
    def func_gaussian_linear(x, ampl, xpos, sigm, slope, yint):
        """Define a Gaussian function with a linear component."""
        return (-1 * ampl * np.exp(-((x - xpos) ** 2) / (2.0 * sigm**2))) + (slope * x + yint)


if __name__ == "__main__":
    # Use argparse to handle command-line arguments
    parser = argparse.ArgumentParser(description="Fit and plot data from a CSV file.")
    parser.add_argument(
        "file", nargs="?", default=None, help="Path to the CSV file containing the data."
    )
    args = parser.parse_args()

    # If no file is provided, prompt the user to select one
    if args.file:
        file_name = args.file
    else:
        from tkinter import Tk
        from tkinter.filedialog import askopenfilename

        print("Select a .csv file to load")
        Tk().withdraw()
        file_name = askopenfilename()

    if not file_name.endswith(".csv"):
        raise ValueError("Please select a valid .csv file")

    # Determine the output directory based on the input file name
    base_name = os.path.splitext(os.path.basename(file_name))[0]
    output_dir = os.path.join(os.path.dirname(file_name), "../plots", base_name)

    df = pd.read_csv(file_name)

    # Extract position and coincidences columns
    try:
        position = df["position"].to_numpy()
    except KeyError:
        raise KeyError("The CSV file must contain a 'position' column")

    coincidences_columns = ["TT"]
    valid_columns = [col for col in coincidences_columns if col in df.columns]

    if not valid_columns:
        raise ValueError("No valid coincidences columns found in the CSV file")

    coincidences = df[valid_columns].sum(axis=1).to_numpy()

    # Run the fitting and plotting
    FitHOM().fit_and_plot(position, coincidences, output_dir)
