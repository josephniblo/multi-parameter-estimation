# -*- coding: utf-8 -*-
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pandas as pd


class FitHOM:
    def __init__(self, position, coincidences):
        x_data = position
        y_data = coincidences

        if len(x_data) != len(y_data):
            print('X and Y datasets have unmatched lengths')
        
        # guess starting parameters from input data for Levenberg-Marquardt algorithm
        bkgd_guess = np.mean(np.sort(y_data)[int(round(0.90 * len(y_data)))::])  # average 90 percentile largest values
        xpos_guess = x_data[np.argmin(y_data)]  # use min(coincidences) index to guess dip position
        sigm_guess = 0.8  # typical value, no obvious way to guess from dataset
        ampl_guess = bkgd_guess - min(y_data)  # basically max-min
        # additional fit function parameters to account for drifts in count rates
        slope_guess = (y_data[0] - y_data[-1]) / (x_data[0] - x_data[-1])  # slope from rise over run, first and final data point
        yint_guess = y_data[0] - slope_guess * x_data[0]  # y-intercept extrapolated from guessed slope and first data point

        # scipy.optimize.curve_fit for each 'static' function defined after main code
        # standard Gaussian: _g
        param_guess_g = (ampl_guess, xpos_guess, sigm_guess, bkgd_guess)
        popt_g, pcov_g = curve_fit(self.func_gaussian, x_data, y_data, p0=param_guess_g)
        perr_g = np.sqrt(np.diag(pcov_g))

        # Gaussian+linear: _g1
        param_guess_g1 = (ampl_guess, xpos_guess, sigm_guess, slope_guess, yint_guess)
        popt_g1, pcov_g1 = curve_fit(self.func_gaussian_linear, x_data, y_data, p0=param_guess_g1)
        perr_g1 = np.sqrt(np.diag(pcov_g1))

        # standard triangular: _t
        param_guess_t = (ampl_guess, xpos_guess, sigm_guess, sigm_guess, bkgd_guess)
        popt_t, pcov_t = curve_fit(self.func_triangle, x_data, y_data, p0=param_guess_t)
        perr_t = np.sqrt(np.diag(pcov_t))

        # triangle+linear: _t1
        param_guess_t1 = (ampl_guess, xpos_guess, sigm_guess, sigm_guess, slope_guess, yint_guess)
        popt_t1, pcov_t1 = curve_fit(self.func_triangle_linear, x_data, y_data, p0=param_guess_t1)

        # calculate visibilities based on optimal parameters (popt) of each fit
        vis_g = popt_g[0] / popt_g[3]
        y0max_g1 = popt_g1[3] * popt_g1[1] + popt_g1[4]  # this parameter will be used multiple times
        vis_g1 = popt_g1[0] / y0max_g1  # vis calculated from the slope at x(y_min)
        vis_t = popt_t[0] / popt_t[4]
        y0max_t1 = popt_t1[4] * popt_t1[1] + popt_t1[5]
        vis_t1 = popt_t1[0] / y0max_t1

        # calculate residual sum of squares (from fit function)
        res_g = np.dot((y_data - self.func_gaussian(x_data, *popt_g)), (y_data - self.func_gaussian(x_data, *popt_g)))
        res_g1 = np.dot((y_data - self.func_gaussian_linear(x_data, *popt_g1)), (y_data - self.func_gaussian_linear(x_data, *popt_g1)))
        res_t = np.dot((y_data - self.func_triangle(x_data, *popt_t)), (y_data - self.func_triangle(x_data, *popt_t)))
        res_t1 = np.dot((y_data - self.func_triangle_linear(x_data, *popt_t1)), (y_data - self.func_triangle_linear(x_data, *popt_t1)))
        # calculate total sum of squares (from raw dataset, same for all curve fits)
        y_mean = np.mean(y_data)
        total = np.dot((y_data - y_mean), (y_data - y_mean))
        # calculate r-squared = 1 - (residual sum of squares) / (total sum of squares)
        r_squared_t = 1 - res_t / total
        r_squared_g = 1 - res_g / total
        r_squared_g1 = 1 - res_g1 / total
        r_squared_t1 = 1 - res_t1 / total
        # print a message in terminal with r-squared coefficients using each fit function
        print('R^2 for Gaussian: %.4f and Triangle: %.4f and Gaussian+Linear: %.4f and Triangle+Linear: %.4f' % (
            float(r_squared_g), float(r_squared_t), float(r_squared_g1), float(r_squared_t1)))

        # generate curve fit datapoints
        n_points = 500
        cfit_g = self.func_gaussian(np.linspace(min(x_data), max(x_data), n_points), *popt_g)
        cfit_g1 = self.func_gaussian_linear(np.linspace(min(x_data), max(x_data), n_points), *popt_g1)
        cfit_t = self.func_triangle(np.linspace(min(x_data), max(x_data), n_points), *popt_t)
        cfit_t1 = self.func_triangle_linear(np.linspace(min(x_data), max(x_data), n_points), *popt_t1)

        # plot each curve-fit with raw data
        plt.figure('Gaussian fit', figsize=(8, 6), dpi=200, facecolor='w', edgecolor='k')        
        plt.ion()  # enables interactive mode to prevent terminal block after each plot, re-enable to hold before final plot
        data_g, cfit_g = self.plot_curve_fit(x_data, y_data, cfit_g, n_points)
        plt.legend((data_g, cfit_g), ('data', 'fit: ampl=%.0f, xpos=%.4f, sigm=%.4f, bkgd=%.0f' % tuple(popt_g)))
        self.format_curve_fit(r_squared_g, vis_g)

        plt.figure('Gaussian+Linear fit', figsize=(8, 6), dpi=200, facecolor='w', edgecolor='k')
        plt.savefig('Gaussian+Linear.png')

    def plot_curve_fit(self, x_data, y_data, cfit_data, n_points):
        # parse raw data structures
        x = x_data
        y = y_data
        # parse curve-fit data structures
        cfit = cfit_data
        n = n_points

        # plot raw data        
        data_plot, = plt.plot(x, y, 'b.')
        
        # plot curve fit
        cfit_plot, = plt.plot(np.linspace(min(x), max(x), n), cfit, 'r-')
        
        # draw pseudo-confidence region around curve fit defined by Poissonian statistics
        plt.fill_between(np.linspace(min(x), max(x), n), cfit - np.sqrt(cfit), cfit + np.sqrt(cfit), alpha=0.3, color='r')

        return data_plot, cfit_plot

        
    def format_curve_fit(self, r_squared, vis_calc):
        r_sqr = r_squared
        vis = vis_calc
        # general formatting
        plt.suptitle('R^2: %4f, Fitted vis: %.4f' % (float(r_sqr), float(vis)), fontsize=20)
        plt.xlabel('position', fontsize=16)
        plt.ylabel('coincidences', fontsize=16)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.show()

    @staticmethod
    def func_gaussian(x, ampl, xpos, sigm, bkgd):
        # define standard Gaussian function
        return -1 * ampl * np.exp(-(x - xpos)**2 / (2. * sigm**2)) + bkgd 
    
    @staticmethod
    def func_triangle(x, ampl, xpos, sigm_l, sigm_r, bkgd):
        # define triangular function
        ret = []
        for xx in x:
            if xx < (xpos - sigm_l):
                ret.append(bkgd)
            elif (xpos - sigm_l) <= xx < xpos:
                ret.append(bkgd - ((xx - (xpos - sigm_l)) / (xpos - (xpos - sigm_l)) * ampl))
            elif xpos <= xx < (xpos + sigm_r):
                ret.append(bkgd - (((xpos + sigm_r) - xx) / ((xpos + sigm_r) - xpos)) * ampl)
            elif xx >= (xpos + sigm_r):                
                ret.append(bkgd)
        return np.array(ret)

    @staticmethod
    def func_gaussian_linear(x, ampl, xpos, sigm, slope, yint):
        # define a Gaussian function with a slope function
        return (-1 * ampl * np.exp(-(x - xpos)**2 / (2. * sigm**2))) + (slope * x + yint)

    @staticmethod
    def func_triangle_linear(x, ampl, xpos, sig_l, sig_r, slope, yint):
        # define triangular function
        ret = []
        for xx in x:
            if xx < (xpos - sig_l):
                ret.append(slope * xx + yint)
            elif (xpos - sig_l) <= xx < xpos:
                ret.append((slope * xx + yint) - ((xx - (xpos - sig_l)) / (xpos - (xpos - sig_l)) * ampl))
            elif xpos <= xx < (xpos + sig_r):
                ret.append((slope * xx + yint) - (((xpos + sig_r) - xx) / ((xpos + sig_r) - xpos)) * ampl)
            elif xx >= (xpos + sig_r):                
                ret.append(slope * xx + yint)
        return np.array(ret)

    @staticmethod
    def func_linear(x, slope, yint):
        # define standard linear gradient function
        return slope * x + yint


if __name__ == '__main__':
    from tkinter import Tk
    from tkinter.filedialog import askopenfilename
    
    Tk().withdraw()
    # await user to select a file
    print('Select a .csv file to load')
    file_name = ''
    while file_name == '':
        file_name = askopenfilename()
    
    # check file is csv, later on can include more file types for loading
    file_name = str(file_name)
    if str(file_name)[-4::] == '.csv':
        print('File selected: %s' % file_name)
    else:
        print('Select a valid .csv file')
        sys.exit()

    df = pd.read_csv(file_name)
    
    # check csv file contains column headers, 'position' and multiple potential coincidences columns
    try:
        position = df['position']
    except KeyError:
        while True:
            try:
                x_string = input('Specify scan position column name %s: ' % str(tuple(df.columns)))
                position = df[x_string]
                break
            except SyntaxError:
                print('Enter the string with quotation marks')

    # List of potential coincidences column names
    coincidences_columns = ['TT']  # Sep source ["cc"] 

    # Check which of the specified columns exist in the dataframe
    valid_columns = [col for col in coincidences_columns if col in df.columns]

    if not valid_columns:  # If none of the columns are found, prompt the user to specify
        while True:
            try:
                y_string = input('Specify coincidences column name(s), separated by commas: %s: ' % str(tuple(df.columns)))
                valid_columns = [col.strip() for col in y_string.split(',') if col.strip() in df.columns]
                if valid_columns:
                    break
                else:
                    print('No valid columns specified. Try again.')
            except SyntaxError:
                print('Enter the string with quotation marks.')

    # Sum the data from the selected columns to create a new coincidences column
    if valid_columns:
        df['coincidences'] = df[valid_columns].sum(axis=1)
        print(f"Summed columns: {valid_columns} into 'coincidences'.")
    else:
        raise ValueError("No valid columns for coincidences were found or specified.")

    # for easier handling of data will format into numpy.array
    position = np.array(position)
    coincidences = np.array(df['coincidences'])

    # run main script
    FitHOM(position, coincidences)

    # hold final plot open until closed by user
    plt.ioff()  # disable interactive mode to hold final plot open
