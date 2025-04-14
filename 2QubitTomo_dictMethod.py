# -*- coding: utf-8 -*-
"""
Created on Tue Aug  8 14:48:30 2023

@author: rb2067

Currently working 
"""
import os
import sys
sys.path.append('./tomoFunc/')
import pandas as pd
import numpy as np
from numpy.linalg import eig
import scipy.linalg as linalg
import itertools as it
import scipy.sparse as sparse
from scipy.optimize import minimize,differential_evolution,shgo
import cvxpy as cp
from numpy.random import poisson
import pandas as pd
from tomoFunc import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as colors
import matplotlib.cm as cm
import matplotlib
from mpl_toolkits.mplot3d import Axes3D

#file_path = '/home/jh115/Heriot-Watt University Team Dropbox/RES_EPS_EMQL/projects/QuantumSecretSharing/Scripts/Tomo_data_to_process'

data = pd.read_csv('2025-03-16--17h-28m_16--17h-30m_2qb_tomo_sagnac2_20mW_A.csv')
counts = data.loc[:, 'TT':'RR']

#Step 1.) convert matrix to columns
M = np.array(counts)
# Define the mapping
basis_map = {
    'Z': {'T': 'H', 'R': 'V'},
    'X': {'T': 'D', 'R': 'A'},
    'Y': {'T': 'R', 'R': 'L'}
}

def quad2QubitS(new_measurements_array):
    '''
    Translates data from 2-qubit scheme measurements (structured as tuples)
    to a dict with projector labels as keys and (non-normalized)
    data as values. This form is used by dataSolve() to return
    a density matrix.

    Arguments:
        > new_measurements_array: array of tuples containing (projector label, data value)

    Outputs:
        > tomodic: dictionary of data indexed by associated projector
    '''

    # Extract labels and data from the tuples
    lab = [t[0] for t in new_measurements_array]
    dat = np.array([t[1] for t in new_measurements_array], dtype=float).reshape(-1, 1)

    # Construct dictionary
    tomodic = {key: dat[i].item() for i, key in enumerate(lab)}

    # Basis labels for 2 qubits
    lb = list(it.product(['Z', 'X', 'Y'], repeat=2))

    ord = list()

    # Get individual projectors for 2 qubits.
    # Assuming the function pLab(k) returns projectors for 2-qubits.
    for k in lb:
        L = pLab(''.join(k))

        for i, p in enumerate(L):
            L[i] = ''.join(p)

        ord.extend(L)

    # Reorder dict
    for key in ord:
        tomodic[key] = tomodic.pop(key)

    return tomodic

# Prepare basis labels
basis_labels = [''.join(p) for p in it.product(['Z', 'X', 'Y'], repeat=2)]
detector_labels = [''.join(p) for p in it.product(['T', 'R'], repeat=2)]

new_measurements = []

for i, basis in enumerate(basis_labels):
    for j, detector in enumerate(detector_labels):
        # Decompose basis and detector to individual components
        mapped_label = ''.join([basis_map[b][d] for b, d in zip(basis, detector)])

        # Append to new measurements list
        new_measurements.append((mapped_label, M[i, j]))

# Updated to include 4 qubit combinations


tomodic = quad2QubitS(new_measurements)

rho,projs = dataSolve(tomodic)



def barplot3d(rho, title, dpi=100):
    fig = plt.figure(figsize=(12, 8), dpi=dpi)
    ax1 = fig.add_subplot(111, projection='3d')

    x = range(1, 5)
    y = range(1, 5)
    z = np.zeros(16)

    X, Y = np.meshgrid(x, y)
    X_ = np.ravel(X)
    Y_ = np.ravel(Y)

    dx = np.zeros(16) + 0.5
    dy = np.zeros(16) + 0.5
    dz = rho.reshape(16)

    cmap = cm.get_cmap('inferno')
    max_height = np.max(dz)
    min_height = np.min(dz)
    rgba = [cmap((k - min_height) / max_height) for k in dz]

    ax1.bar3d(X_, Y_, z, dx, dy, dz, color=rgba)

    tick_labels = ['HH', 'HV', 'VH', 'VV']
    ticks = np.arange(1.25, 5.25, 1)  # Adjusted so the ticks and labels match

    plt.xticks(ticks, tick_labels, rotation=45, ha='right')
    plt.yticks(ticks, tick_labels, rotation=-45, ha='right')

    ax1.set_zlim(0, 1)
    plt.tight_layout()
    plt.title(title)
    plt.show()

    return fig

barplot3d(np.real(rho),'Real(rho)')
barplot3d(np.imag(rho),'Imag(rho)')

def psi_plus_bell_state():
    state_01 = np.array([0, 1, 0, 0])
    state_10 = np.array([0, 0, 1, 0])
    psi_plus = (state_01 - state_10) / np.sqrt(2)
    rho_plus = np.outer(psi_plus, np.conj(psi_plus).T)
    return rho_plus
bell = psi_plus_bell_state()

def phi_plus_bell_state():
    state_01 = np.array([1, 0, 0, 0])
    state_10 = np.array([0, 0, 0, 1])
    psi_plus = (state_01 + state_10) / np.sqrt(2)
    rho_plus = np.outer(psi_plus, np.conj(psi_plus).T)
    return rho_plus
phi_plus = phi_plus_bell_state()

Sep_HH = np.outer(np.array([1, 0, 0, 0]),np.array([1, 0, 0, 0]).T)
Sep_VV = np.outer(np.array([0, 0, 0, 1]),np.array([0, 0, 0, 1]).T)

# Compute fidelity between two density matrices
def fidelity(rho1, rho2):
    sqrt_rho1 = np.sqrt(rho1)
    fidelity = np.trace(np.sqrt(np.dot(np.dot(sqrt_rho1, rho2), sqrt_rho1)))
    return abs(fidelity)
def purity(rho):
    return np.trace(rho@rho)
Z = np.array([[1,0],[0,-1]])
ZZ = np.kron(Z,Z)
def QBER(rho,ZZ):
    E = np.trace(rho@ZZ)
    return (1-E)/2
X = np.array([[0,1],[1,0]])
XX = np.kron(X,X)
def QX(rho,XX):
    E = np.trace(rho@XX)
    return (1-E)/2

# Calculate fidelity
Fidelity = round(fid(phi_plus, rho),4)
Fid_HH = round(fid(Sep_HH, rho), 4)
Fid_VV = round(fid(Sep_VV, rho), 4)
print('fidelity with HH',Fid_HH )
#Calculate purity
Purity = round(np.real(purity(rho)),4)

print('Fidelity with Phi+:',Fidelity)
print('Purity:', Purity)

print('QBER',QBER(rho,ZZ))
print('QX', QX(rho,XX) )



def barplot3d_error(matrix, title='Matrix'):
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection='3d')

    # Matrix dimensions
    num_rows, num_cols = matrix.shape

    # Create coordinate arrays
    xpos, ypos = np.meshgrid(np.arange(num_rows), np.arange(num_cols), indexing='ij')
    xpos = xpos.flatten()
    ypos = ypos.flatten()
    zpos = np.zeros_like(xpos)

    # Bar dimensions
    dx = dy = 0.5 * np.ones_like(zpos)
    dz = matrix.flatten()

    # Choose colors based on sign of dz
    colors = ['r' if val < 0 else 'b' for val in dz]

    # Plot bars
    ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color=colors, shade=True)

    # Adjust axis limits to maximize bar sizes
    max_x = num_rows
    max_y = num_cols
    max_z = np.max(np.abs(dz))  # Use absolute max value for symmetric scaling

    ax.set_xlim(-0.5, max_x - 0.5)
    ax.set_ylim(-0.5, max_y - 0.5)
    ax.set_zlim(-max_z, max_z)

    # Labels
    ax.set_xlabel('Row Index')
    ax.set_ylabel('Column Index')
    ax.set_zlabel('Value')
    ax.set_title(title)
    # Set custom tick labels
    ax.set_xticks(np.arange(num_rows))
    ax.set_xticklabels(['|HH⟩', '|HV⟩', '|VH⟩', '|VV⟩'])
    ax.set_yticks(np.arange(num_cols))
    ax.set_yticklabels(['⟨HH|', '⟨HV|', '⟨VH|', '⟨VV|'])
    # Normalize dz for colormap
    norm = plt.Normalize(-max_z, max_z)
    colors = cm.seismic(norm(dz))
    plt.show()

# difference = rho - phi_plus
# barplot3d_error(np.real(difference),'Real(difference with ideal Phi+ )')
# barplot3d_error(np.imag(difference),'Imag(difference with ideal Phi+)')