# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 15:58:51 2023

@author: emerson.almeida

https://docs.simpeg.xyz/content/tutorials/05-dcr/plot_inv_2_dcr2d.html#sphx-glr-content-tutorials-05-dcr-plot-inv-2-dcr2d-py


"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
 
from discretize import TreeMesh
from discretize.utils import mkvc, refine_tree_xyz, active_from_xyz, mesh_builder_xyz

from SimPEG import (maps, data_misfit, regularization, optimization,
                    inverse_problem, inversion, directives)

from SimPEG.electromagnetics.static import resistivity as dc
from SimPEG.electromagnetics.static.utils.static_utils import plot_pseudosection
from SimPEG.utils.io_utils.io_utils_electromagnetics import read_dcip2d_ubc

try:
    from pymatsolver import Pardiso as Solver
except ImportError:
    from SimPEG import SolverLU as Solver


FIGSIZE = (8, 5)
COLORMAP = mpl.cm.jet
CONDUTIVIDADE_AR = 1e-8
OCTREE_LEVELS = [4, 6, 8]


def load_dados(data_file, topo_file=None):
    dc_data = read_dcip2d_ubc(data_file, "volt", "general")
    dc_data = _assign_uncertainty(dc_data)
    
    if topo_file:
        topo_xyz = np.loadtxt(topo_file)
    else:
        topo_xyz = create_zero_topography(dc_data)
    return [topo_xyz, dc_data]


def create_zero_topography(dc_data):
    x_domain = _get_unique_electrode_locations(dc_data)[:, 0]
    xmin, xmax = [np.min(x_domain), np.max(x_domain)]
    half_x_width = (xmax - xmin) / 2.0
    xmin -= half_x_width
    xmax += half_x_width
    
    x = np.linspace(xmin, xmax, 250, endpoint=True)
    y = np.copy(x)
    
    Z = np.zeros((len(x)*len(y), 1))
    
    Y, X = np.meshgrid(x, y)
    Y = Y.reshape([-1, 1])
    X = X.reshape([-1, 1])
    
    return np.concatenate([X, Y, Z], axis=1)
    

def _assign_uncertainty(dc_data):
    # Inversion with SimPEG requires that we define the uncertainties on 
    # our data. This represents our estimate of the standard deviation of 
    # the noise in our data. For DC data, the uncertainties are 10% of the 
    # absolute value    
    dc_data.standard_deviation = 0.05 * np.abs(dc_data.dobs)
    return dc_data


def cria_malha(dc_data, topo_xyz, mesh_param):
    """
    https://discretize.simpeg.xyz/en/main/api/generated/discretize.utils.mesh_builder_xyz.html
    
    https://simpeg.discourse.group/t/using-mesh-builder-xyz/66/3
    """
   
    dh = mesh_param['delta_h']
    pad = mesh_param['padding']
    
    coordinates = _get_unique_electrode_locations(dc_data)
    mesh = mesh_builder_xyz(coordinates, [dh, dh],
                            padding_distance=[[pad, pad], [pad, 0.0]],
                            depth_core=pad,
                            mesh_type='tree')
    mesh = _refine_mesh_topography(mesh, topo_xyz)
    mesh = _refine_mesh_electrodes(mesh, dc_data)
    x_min, x_max, z_min, z_max = _calc_core_coordinates(coordinates, pad)    
    mesh = _refine_mesh_core(mesh, x_min, x_max, z_min, z_max)
    mesh.finalize()
    return mesh


def _get_unique_electrode_locations(dc_data):
    electrode_locations = np.c_[dc_data.survey.locations_a, dc_data.survey.locations_b,
                                dc_data.survey.locations_m, dc_data.survey.locations_n]
    return np.unique(np.reshape(electrode_locations, (4*dc_data.survey.nD, 2)), axis=0)


def _calc_core_coordinates(coord, pad):
    xmin, xmax = [np.min(coord[:, 0] - pad/2.0), np.max(coord[:, 0]) + pad/2.0]
    zmin, zmax = [np.min(coord[:, 1]), np.max(coord[:, 1])]
    zmin -= (xmax - xmin) / 2.0
    return [xmin, xmax, zmin, zmax]


def _define_base_mesh(domain_width_x, domain_width_z, dh):
    n_cels_x = 2**int(np.round(np.log(domain_width_x / dh) / np.log(2.0)))
    n_cels_z = 2**int(np.round(np.log(domain_width_z / dh) / np.log(2.0)))

    hx = [(dh, n_cels_x)]
    hz = [(dh, n_cels_z)]
    
    """
    {‘0’, ‘C’, ‘N’} a str specifying whether the zero coordinate along each axis 
    is the first node location (‘0’), in the center (‘C’) or the last node 
    location (‘N’)
    
    https://discretize.simpeg.xyz/en/main/api/generated/discretize.TreeMesh.html#discretize.TreeMesh
    """
    return TreeMesh([hx, hz], x0='CN')


def _refine_mesh_topography(mesh, topo_xyz):
    topo_2d = topo_xyz[:, [0, -1]]
    
    """
    https://github.com/simpeg/discretize/issues/254
    
    The discretization around that points is going to depend on the list you 
    entered for levels. Let levels = [4, 6, 8] and assume the smallest cells 
    (base cell) in your tree mesh have side length h. Because the first entry 
    in levels is 4, your mesh will use the smallest cell size within a radius 
    of 4 x h around the point your provided. The second entry in levels is 
    a 6, so within a radius of 6 x (2h), your cells will have widths of 2h. 
    And with a radius of 8 x (4h) your cells will have widths 4h. And so 
    forth until you reach the maximum allowable cells size for the mesh.
    """
    return refine_tree_xyz(mesh, topo_2d, method='surface', 
                           octree_levels=OCTREE_LEVELS, finalize=False)


def _refine_mesh_electrodes(mesh, dc_data):
    coordinates = _get_unique_electrode_locations(dc_data)
    return refine_tree_xyz(mesh, coordinates, octree_levels=OCTREE_LEVELS, 
                           method='radial', finalize=False)


def _refine_mesh_core(mesh, xi, xf, zf, zi):
    xp, zp = np.meshgrid([xi, xf], [zf, zi])
    xyz = np.c_[mkvc(xp), mkvc(zp)]
    return refine_tree_xyz(mesh, xyz, octree_levels=OCTREE_LEVELS, method='box', finalize=False)


def project_surveys_topography(dc_data, topo_xyz, mesh):
    # project surveys to discretized topography
    """
    This step is carried out to ensure all electrodes like on the discretized 
    surface.
    """
    ind_active = _get_2d_active_indices(topo_xyz, mesh)
    survey = dc_data.survey
    survey.drape_electrodes_on_topography(mesh, ind_active, option='top')
    return survey
    

def _get_2d_active_indices(topo_xyz, mesh):
    topo_2d = np.unique(topo_xyz[:, [0, -1]], axis=0)
    return active_from_xyz(mesh, topo_2d)


def set_starting_model(bg_conductivity, topo_xyz, mesh):
    bg_conductivity = np.log(bg_conductivity)
    ind_active = _get_2d_active_indices(topo_xyz, mesh)
    nC = int(ind_active.sum())
    return bg_conductivity * np.ones(nC)


def define_simulation_physics(dc_data, topo_xyz, mesh):
    ind_active = _get_2d_active_indices(topo_xyz, mesh)
    active_map = maps.InjectActiveCells(mesh, ind_active, CONDUTIVIDADE_AR)
    conductivity_map = active_map * maps.ExpMap()
    return dc.simulation_2d.Simulation2DNodal(mesh, survey=dc_data.survey, 
                                              sigmaMap=conductivity_map,
                                              solver=Solver, storeJ=True)


def define_inverse_problem(dc_data, topo_xyz, mesh, starting_model, simulation):
    ind_active = _get_2d_active_indices(topo_xyz, mesh)
    dmis = data_misfit.L2DataMisfit(data=dc_data, simulation=simulation)
    reg = regularization.Simple(malha, indActive=ind_active, 
                                mref=starting_model, alpha_s=0.01, 
                                alpha_x=1.0, alpha_y=1.0)

    reg.mrefInSmooth = True
    opt = optimization.InexactGaussNewton(maxIter=40)
    return inverse_problem.BaseInvProblem(dmis, reg, opt)


def define_inversion_directives():
    update_sensibility_weighting = directives.UpdateSensitivityWeights()

    # beta: trade-off parameter between the data misfit and the regularization
    starting_beta = directives.BetaEstimate_ByEig(beta0_ratio=1e1)
    
    # coolingFactor: rate of reduction in trade-off parameter (beta) each 
    # time the the inverse problem is solved.
    # coolingRate: number of Gauss-Newton iterations for each trade-off 
    # paramter value.
    beta_schedule = directives.BetaSchedule(coolingFactor=2, coolingRate=2)
    
    # chi: stopping criteria for the inversion
    target_misfit = directives.TargetMisfit(chifact=1)
    
    save_iteration = directives.SaveOutputEveryIteration(save_txt=False)
    update_jacobi = directives.UpdatePreconditioner()
    
    return [update_sensibility_weighting, starting_beta, beta_schedule, 
            save_iteration, target_misfit, update_jacobi]


def run_inversion(inv_prob, directives_list, starting_conductivity_model):
    dc_inversion = inversion.BaseInversion(inv_prob, 
                                           directiveList=directives_list)

    return dc_inversion.run(starting_conductivity_model)


def plota_dados(dc_data, topo_xyz):
    fig = plt.figure(figsize=FIGSIZE)
    ax1 = fig.add_axes([0.1, 0.3, 0.8, 0.5])
    plot_pseudosection(dc_data, plot_type='contourf', ax=ax1, scale='log', 
                       data_type='apparent conductivity', 
                       cbar_label=r'$\sigma$ (S/m)', mask_topography=True,
                       contourf_opts={'levels': 100, 'cmap': COLORMAP})
    ax1.plot(topo_xyz[:, 0], topo_xyz[:, -1], 'k', lw=2.0)
    ax1.set_xlim(_get_survey_limits(dc_data))
    ax1.set_ylim(-1 * _get_survey_length(dc_data) / 3.0, 
                 np.max(topo_xyz[:, -1]) + _get_survey_length(dc_data)*0.05)
    ax1.set_title('Pseudo-seção de condutividade aparente')
    ax1.set_xlabel('Distância (m)')
    ax1.set_ylabel('Pseudo-profundidade (m)')
    plt.tight_layout()


def _get_survey_limits(dc_data):
    xmin = np.min(dc_data.survey.unique_electrode_locations[:, 0])
    xmax = np.max(dc_data.survey.unique_electrode_locations[:, 0])
    return xmin, xmax

def _get_survey_length(dc_data):
    xmin, xmax = _get_survey_limits(dc_data)
    return xmax - xmin


def plota_resultados(recovered_conductivity_model, topo_xyz, mesh, dc_data):
    ind_active = _get_2d_active_indices(topo_xyz, mesh)
    active_map = maps.InjectActiveCells(mesh, ind_active, CONDUTIVIDADE_AR)
    conductivity_map = active_map * maps.ExpMap()
    recovered_conductivity = conductivity_map * recovered_conductivity_model
    recovered_conductivity[~ind_active] = np.NaN
    sigma_min, sigma_max = [np.min(recovered_conductivity_model), 
                            np.max(recovered_conductivity_model)]
    norm = Normalize(vmin=np.exp(sigma_min), vmax=np.exp(sigma_max))
    
    fig = plt.figure(figsize=FIGSIZE)
    ax1 = fig.add_axes([0.14, 0.17, 0.68, 0.7])
    mesh.plotImage(recovered_conductivity, normal='Y', ax=ax1, 
                    grid=False, pcolor_opts={'norm': LogNorm(), 'cmap': COLORMAP})
    ax1.plot(topo_xyz[:, 0], topo_xyz[:, -1], 'k', lw=2.0)
    ax1.set_xlim(_get_survey_limits(dc_data))
    ax1.set_ylim(-1 * _get_survey_length(dc_data) / 1.5, 
                 np.max(topo_xyz[:, -1]) + _get_survey_length(dc_data)*0.05)
    ax1.set_title('Modelo Invertido')
    ax1.set_xlabel('Distância (m)')
    ax1.set_ylabel('Profundidade (m)')
    
    ax2 = fig.add_axes([0.84, 0.17, 0.03, 0.7])
    cbar = mpl.colorbar.ColorbarBase(ax2, norm=norm, orientation='vertical',
                                     cmap=COLORMAP)
    cbar.set_label(r'$\sigma$ (S/m)', rotation=270, labelpad=15)
    
    plt.tight_layout()


def plota_malha(mesh, dc_data, topo_xyz):
    fig = plt.figure(figsize=(10, 4))
    ax1 = fig.add_axes([0.1, 0.1, 0.75, 0.85])
    mesh.plotGrid(ax=ax1)
    ax1.plot(topo_xyz[:, 0], topo_xyz[:, -1], 'k', lw=2.0)
    ax1.set_xlim(_get_survey_limits(dc_data))
    ax1.set_ylim(-1 * _get_survey_length(dc_data) / 1.5, 
                 np.max(topo_xyz[:, -1]) + _get_survey_length(dc_data)*0.05)
    ax1.set_title('Discretização do espaço do modelo')
    ax1.set_xlabel('Distância (m)')
    ax1.set_ylabel('Profundidade (m)')
    plt.tight_layout()
    






# ---------------------------------------------------------------------------


parametros_malha = {'delta_h': 0.2, 'padding': 10.0}
condutividade_background = 1e-2
topografia, dados = load_dados('..\dados\CE1.dat')
plota_dados(dados, topografia)

malha = cria_malha(dados, topografia, parametros_malha)
plota_malha(malha, dados, topografia)


dados.survey = project_surveys_topography(dados, topografia, malha)
modelo_inicial = set_starting_model(condutividade_background, topografia, malha)
simulacao = define_simulation_physics(dados, topografia, malha)
problema_inverso = define_inverse_problem(dados, topografia, malha, modelo_inicial, simulacao)
lista_diretivas = define_inversion_directives()
modelo_invertido = run_inversion(problema_inverso, lista_diretivas, modelo_inicial)
plota_resultados(modelo_invertido, topografia, malha, dados)


# ---------------------------------------------------------------------------

























