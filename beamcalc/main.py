# -*- coding: utf-8 -*-
"""
Created on Sun Oct 12 14:36:42 2025

@author: devoi
"""
import json
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from loguru import logger
from dataclasses import dataclass

from scipy.integrate import solve_bvp, simpson


import utils

settings = utils.Settings("settings.json")

def dirac_delta(z, m=1000.0, diff=0):
    if diff == 0:
        return m / np.sqrt(np.pi) * np.exp(-(m * z)**2)
    elif diff == 1:
        return -2 * m**2 * z * np.exp(-(m * z)**2)


class BeamState:
    """Индексы вектора состояния для задачи изгиба прямолинейного стержня.

    Returns:
        v: прогиб
        theta: угол поворота
        Q: поперечная сила
        M: изгибающий момент
    """
    v = 0
    theta = 1
    Q = 2
    M = 3

    @staticmethod
    def get_idx_by_bc(end_type):
        match end_type:
            case "fix":
                return [BeamState.v, BeamState.theta]
            case "slider":
                return [BeamState.theta, BeamState.Q]
            case "pin":
                return [BeamState.v, BeamState.M]
            case "free":
                return [BeamState.Q, BeamState.M]


@dataclass
class BeamData:
    z: np.ndarray
    EJ: np.ndarray
    k: np.ndarray
    q: np.ndarray
    P: np.ndarray
    M: np.ndarray
    n_pts: int
    left_bc: str
    right_bc : str
    int_bc: list

    def __post_init__(self):
        self.zi = np.linspace(0.0, self.z[-1], self.n_pts)

    @classmethod
    def from_dict(cls, data):
        return cls(np.array(data["z"]),
                   np.array(data["EJ"]),
                   np.array(data["k"]),
                   np.array(data["q"]),
                   np.array(data["P"]),
                   np.array(data["M"]),
                   data["n_pts"],
                   data["left_bc"],
                   data["right_bc"],
                   data["int_bc"])

    def distrib_load(self, z):
        Pd = np.zeros_like(z)
        for zi, Pi in zip(self.z[1:-1], self.P[1:-1]):
            Pd += Pi * dirac_delta(z - zi)
        return np.interp(z, self.z, self.q) + Pd

    def distrib_moment(self, z):
        Md = np.zeros_like(z)
        for zi, Mi in zip(self.z[1:-1], self.M[1:-1]):
            Md += Mi * dirac_delta(z - zi)
        return Md

    def distrib_slope(self, z):
        pass

    def found_stiff(self, z):
        return np.interp(z, self.z, self.k)

    def bend_stiff(self, z):
        return np.interp(z, self.z, self.EJ)


def import_data(file_path):
    with open(file_path) as file:
        data = json.load(file)
    logger.info(f"Открыт файл '{file_path}'.")
    return BeamData.from_dict(data)


def ode_beam(z, y, beam_data, q, m):
    dydz = np.zeros_like(y)
    dydz[BeamState.v] = -y[BeamState.theta]
    dydz[BeamState.theta] = y[BeamState.M] / beam_data.bend_stiff(z)
    dydz[BeamState.Q] = (-q(z) +
                         beam_data.found_stiff(z) * y[BeamState.v])
    dydz[BeamState.M] = y[BeamState.Q] - m(z)
    return dydz


def bc_bvp(ya, yb, beam_data):
    idx_left = BeamState.get_idx_by_bc(beam_data.left_bc)
    idx_right = BeamState.get_idx_by_bc(beam_data.right_bc)
    ya0 = np.array([0.0,
                    0.0,
                    -beam_data.P[0],
                    beam_data.M[0]])
    yb0 = np.array([0.0,
                    0.0,
                    -beam_data.P[-1],
                    beam_data.M[-1]])
    bc_a = ya[idx_left] - ya0[idx_left]
    bc_b = yb[idx_right] - yb0[idx_right]
    return np.hstack((bc_a, bc_b))


def get_reactions(beam_data):
    if not beam_data.int_bc:
        logger.info("Задача не содержит внутренних закреплений.")
        return None
    num_of_reactions = len(beam_data.int_bc)
    compliance = np.zeros((num_of_reactions, num_of_reactions))
    rhs = np.zeros((num_of_reactions, ))

    for i in range(num_of_reactions):
        for j in range(i, num_of_reactions):
            compliance[i, j] = get_compliance(i, j, beam_data)
        rhs[i] = -get_compliance_rhs(i, beam_data)
    compliance = np.where(compliance, compliance, compliance.T) # сборка симметричной матрицы из верхнетреугольной
    return np.linalg.solve(compliance, rhs)


def get_compliance_rhs(i, beam_data):
    logger.info(f"Расчет {i}-й компоненты правой части системы уравнений метода сил.")
    sol = solve_bvp(lambda z, y: ode_beam(z, y,
                                          beam_data,
                                          beam_data.distrib_load,
                                          beam_data.distrib_moment),
                    lambda ya, yb: bc_bvp(ya, yb, beam_data),
                    x=beam_data.zi,
                    y=np.zeros((4, beam_data.n_pts)),
                    verbose=settings.VERBOSE,
                    max_nodes=settings.MAXNODES,
                    tol=settings.TOL,
                    bc_tol=settings.BCTOL)
    logger.warning(sol.message)
    if beam_data.int_bc[i][1] == "pin":
        return np.interp(beam_data.int_bc[i][0],
                         sol.x,
                         sol.y[BeamState.v])
    else:
        return np.interp(beam_data.int_bc[i][0],
                         sol.x,
                         sol.y[BeamState.theta])


def get_compliance(i, j, beam_data):
    logger.info(f"Расчет ({i}, {j}) компоненты матрицы податливости.")
    if beam_data.int_bc[i][1] == "pin":
        distrib_load = lambda z: dirac_delta(z - beam_data.int_bc[i][0])
        distrib_moment = lambda z: np.zeros_like(z)
    else:
        distrib_load = lambda z: np.zeros_like(z)
        distrib_moment = lambda z: dirac_delta(z - beam_data.int_bc[i][0])

    sol = solve_bvp(lambda z, y: ode_beam(z, y,
                                          beam_data,
                                          distrib_load,
                                          distrib_moment),
                    lambda ya, yb: bc_bvp(ya, yb, beam_data),
                    x=beam_data.zi,
                    y=np.zeros((4, beam_data.n_pts)),
                    verbose=settings.VERBOSE,
                    max_nodes=settings.MAXNODES,
                    tol=settings.TOL,
                    bc_tol=settings.BCTOL)
    logger.warning(sol.message)
    if beam_data.int_bc[i][1] == "pin":
        return np.interp(beam_data.int_bc[j][0], sol.x, sol.y[BeamState.v])
    else:
        return np.interp(beam_data.int_bc[j][0], sol.x, sol.y[BeamState.theta])


def distrib_load_total(z, r, typ):
    q = beam_data.distrib_load(z)
    m = beam_data.distrib_moment(z)
    for i, bc in enumerate(beam_data.int_bc):
        if bc[1] == "pin":
            q += r[i] * dirac_delta(z - bc[0])
        else:
            m += r[i] * dirac_delta(z - bc[0])
    if typ == "q":
        return q
    if typ == "m":
        return m


def solve(beam_data):
    logger.info("Запуск расчета...")
    r = get_reactions(beam_data)
    sol = solve_bvp(lambda z, y: ode_beam(z, y,
                                          beam_data,
                                          beam_data.distrib_load,
                                          beam_data.distrib_moment),
                    lambda ya, yb: bc_bvp(ya, yb, beam_data),
                    x=beam_data.zi,
                    y=np.zeros((4, beam_data.n_pts)),
                    verbose=settings.VERBOSE,
                    max_nodes=settings.MAXNODES,
                    tol=settings.TOL,
                    bc_tol=settings.BCTOL)
    logger.warning(sol.message)
    if r is None:
        return sol.x, sol.y, None
    else:
        solr = solve_bvp(lambda z, y: ode_beam(z, y,
                                               beam_data,
                                               lambda z: distrib_load_total(z, r, "q"),
                                               lambda z: distrib_load_total(z, r, "m")),
                         lambda ya, yb: bc_bvp(ya, yb, beam_data),
                         x=beam_data.zi,
                         y=np.zeros((4, beam_data.n_pts)),
                         verbose=settings.VERBOSE,
                         max_nodes=settings.MAXNODES,
                         tol=settings.TOL,
                         bc_tol=settings.BCTOL)
        logger.warning(solr.message)
        return solr.x, solr.y, r
    

def calculate_energy_balance(z, y, beam_data):
    U_bend = 0.5 * simpson(y[BeamState.M]**2, z)
    U_found = 0.5 * simpson(beam_data.found_stiff(z) *
                            y[BeamState.v]**2, z)
    W_ext = 0.5 * (simpson(distrib_load_total(z, r, "q") *
                           y[BeamState.v] +
                           distrib_load_total(z, r, "m") *
                           y[BeamState.theta], z)) # по краям!!!
    return U_bend, U_found, W_ext


def print_to_file(out_file, z, y, beam_data, U_bend, U_found, W_ext):
    print_pts = int(beam_data.z[-1] / settings.DZ + 1)
    z_print = np.linspace(0.0, beam_data.z[-1], print_pts)

    with open(f"{out_file}.dat", mode="w", encoding="utf-8") as file:
        header = f"""{'№':>3s} {'z/L':^12s} {'v':^12s} {'ϑ':^12s} {'Q':^12s} {'M':^12s}
====================================================================
"""
        print(header)
        file.write(header)
        for i, zi in enumerate(z_print):
            vi = np.interp(zi, z, y[BeamState.v])
            ti = np.interp(zi, z, y[BeamState.theta])
            Qi = np.interp(zi, z, y[BeamState.Q])
            Mi = np.interp(zi, z, y[BeamState.M])
            row = f"{i:3d} {zi: 10.5E} {vi: 10.5E} {ti: 10.5E} {Qi: 10.5E} {Mi: 10.5E}"
            print(row)
            file.write(row + "\n")
        file.write("\n===============================\n")
        file.write(f"Bending energy   : {U_bend: 10.5E}\n")
        file.write(f"Foundation energy: {U_found: 10.5E}\n")
        file.write(f"Total energy     : {(U_bend + U_found): 10.5E}\n")
        file.write(f"External work    : {W_ext: 10.5E}")


def plot_to_file(out_file, z, y):
    plot_data = [{"coord": (0, 0),
                  "x_data": z,
                  "y_data": y[BeamState.v],
                  "x_name": r"$\frac{z}{L}$",
                  "y_name": r"$\hat{v} \cdot \frac{qL^4}{EJ_x}$",
                  "style": "-k"},
                 {"coord": (1, 0),
                  "x_data": z,
                  "y_data": y[BeamState.theta],
                  "x_name": r"$\frac{z}{L}$",
                  "y_name": r"$\hat{\vartheta} \cdot \frac{qL^3}{EJ_x}$",
                  "style": "-k"},
                 {"coord": (0, 1),
                  "x_data": z,
                  "y_data": y[BeamState.Q],
                  "x_name": r"$\frac{z}{L}$",
                  "y_name": r"$\hat{Q} \cdot qL$",
                  "style": "-k"},
                 {"coord": (1, 1),
                  "x_data": z,
                  "y_data": y[BeamState.M],
                  "x_name": r"$\frac{z}{L}$",
                  "y_name": r"$\hat{M} \cdot qL^2$",
                  "style": "-k"}]

    fig, ax = plt.subplots(ncols=2, nrows=2, figsize=(8, 6))
    for plot_data_i in plot_data:
        ax[*plot_data_i["coord"]].fill_between(plot_data_i["x_data"],
                                               plot_data_i["y_data"],
                                               hatch="|||",
                                               ec="k",
                                               fc="w",
                                               zorder=0)
        ax[*plot_data_i["coord"]].margins(0)
        ax[*plot_data_i["coord"]].minorticks_on()
        ax[*plot_data_i["coord"]].grid(True, which="minor", color="k",
                                       linewidth=0.5, alpha=0.25)
        ax[*plot_data_i["coord"]].set_xlabel(plot_data_i["x_name"])
        ax[*plot_data_i["coord"]].set_ylabel(plot_data_i["y_name"])

    fig.tight_layout()
    fig.savefig(f"{out_file}.png", dpi=400)


if __name__ == "__main__":
    variant = 23
    length = "long"
    in_file = Path(f"vars/var{variant}_{length}.json")
    out_file = f"out/out{variant}_{length}"
    beam_data = import_data(in_file)

    z, y, r = solve(beam_data)

    U_bend, U_found, W_ext = calculate_energy_balance(z, y, beam_data)
    print_to_file(out_file, z, y, beam_data, U_bend, U_found, W_ext)
    plot_to_file(out_file, z, y)
    
    