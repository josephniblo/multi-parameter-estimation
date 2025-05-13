# -*- coding: utf-8 -*-

import json
import os
import sys
import pandas as pd
import numpy as np
import itertools as it
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from tomo_func import *

class TomographyResult:
    def __init__(self, reconstructed_state, target_state, fidelity, purity):
        self.reconstructed_state = reconstructed_state
        self.target_state = target_state
        self.fidelity = fidelity
        self.purity = purity
    
    def _complex_array_to_json_compatible(self, array):
        return [[{'real': float(c.real), 'imag': float(c.imag)} for c in row] for row in array]

    def save_to_json(self, file_path):
        data = {
            'reconstructed_state': self._complex_array_to_json_compatible(self.reconstructed_state),
            'target_state': self._complex_array_to_json_compatible(self.target_state),
            'fidelity': self.fidelity,
            'purity': self.purity
        }
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=4)


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


def barplot3d(rho, title, dpi=100, save_path=None):
    fig = plt.figure(figsize=(8, 8), dpi=dpi)
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

    cmap = plt.get_cmap('inferno')
    max_height = np.max(dz)
    min_height = np.min(dz)
    rgba = [cmap((k - min_height) / max_height) for k in dz]

    ax1.bar3d(X_, Y_, z, dx, dy, dz, color=rgba)

    tick_labels = ['HH', 'HV', 'VH', 'VV']
    ticks = np.arange(1.25, 5.25, 1)  # Adjusted so the ticks and labels match

    plt.xticks(ticks, tick_labels, rotation=45, ha='right')
    plt.yticks(ticks, tick_labels, rotation=-45, ha='right')

    ax1.set_zlim(0, 1)
    plt.title(title)
    
    fig.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
    
    if save_path:
        plt.savefig(save_path, dpi=300)

    return fig


def fidelity(rho1, rho2):
    sqrt_rho1 = np.sqrt(rho1)
    fidelity = np.trace(np.sqrt(np.dot(np.dot(sqrt_rho1, rho2), sqrt_rho1)))
    return abs(fidelity)


def purity(rho):
    return np.trace(rho@rho)


def barplot3d_error(matrix, title='Matrix', save_path=None):
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

    if save_path:
        plt.savefig(save_path, dpi=300)

    return fig


def plot_density_matrix_from_tomo_data(data_directory, target_density_matrix):
    file_name = os.path.join(data_directory, 'tomography_data.csv')
    data = pd.read_csv(file_name, index_col=0)
    counts = data.loc[:, ['TT','RT','TR','RR']]

    #Step 1.) convert matrix to columns
    matrix = np.array(counts)
    # Define the mapping
    basis_map = {
        'Z': {'T': 'H', 'R': 'V'},
        'X': {'T': 'D', 'R': 'A'},
        'Y': {'T': 'R', 'R': 'L'}
    }

    # Prepare basis labels
    basis_labels = [''.join(p) for p in it.product(['Z', 'X', 'Y'], repeat=2)]
    detector_labels = [''.join(p) for p in it.product(['T', 'R'], repeat=2)]

    new_measurements = []

    for i, basis in enumerate(basis_labels):
        for j, detector in enumerate(detector_labels):
            # Decompose basis and detector to individual components
            mapped_label = ''.join([basis_map[b][d] for b, d in zip(basis, detector)])

            # Append to new measurements list
            new_measurements.append((mapped_label, matrix[i, j]))

    tomodic = quad2QubitS(new_measurements)
    rho,projs = dataSolve(tomodic)

    barplot3d(np.real(rho),'Real(rho)', save_path=data_directory + '/real_rho.png')
    barplot3d(np.imag(rho),'Imag(rho)', save_path=data_directory + '/imag_rho.png')

    fidelity = round(fid(target_density_matrix, rho),4)

    pur = round(np.real(purity(rho)),4)

    # Save the reconstructed density matrix and other results
    result = TomographyResult(rho, target_density_matrix, fidelity, pur)
    result.save_to_json(os.path.join(data_directory, 'tomography_results.json'))

    return rho, projs, fidelity, pur

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python analyze_two_qubit_tomo.py <data_directory> <target_density_matrix>")
        sys.exit(1)

    data_directory = sys.argv[1]
    target_density_matrix = np.array(eval(sys.argv[2]))
    
    # Ensure the target density matrix is a 4x4 matrix
    if target_density_matrix.shape != (4, 4):
        print("Error: Target density matrix must be a 4x4 matrix.")
        print("Example: [[1, 0, 0, 0], [0, 0.5, 0.5, 0], [0, 0.5, 0.5, 0], [0, 0, 0, 0]]")
        sys.exit(1)

    repo_root = os.popen('git rev-parse --show-toplevel').read().strip()
    data_plots_directory = os.path.join(repo_root, 'tomography', 'plots', )

    rho, projs, fidelity, purity = plot_density_matrix_from_tomo_data(data_directory, target_density_matrix)
    
    print("Reconstructed Density Matrix (rho):")
    print(rho)
    
    print("Fidelity:", fidelity)
    print("Purity:", purity)
