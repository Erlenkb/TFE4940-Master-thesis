import numpy as np
import pandas as pd
from pylab import *
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.cm
import matplotlib.colorbar
import matplotlib.colors

def _create_TLM(nx, ny, nz):
    tlm = np.zeros((nx,ny,nz,6))
    for x in range(nx):
        for y in range(ny):
            for z in range(nz):
                tlm[x,y,z] = np.zeros(6)
    scatter_grid = tlm
    pressure_grid = tlm
    return tlm, scatter_grid, pressure_grid

def _create_P_grid(Scatter, Pressure):
    list = [2, 3, 0, 1, 5, 4]
    
    for i in range(Scatter.shape[0] - 1):
        for j in range(Scatter.shape[1] - 1):
            for l in range(Scatter.shape[2] - 1):
                for p, s in enumerate(list):
                    
                    if (p == 0):
                        if (i == 0):
                            Pressure[i][j][l][p] = R_front * Scatter[i][j][l][p]
                        else:
                            Pressure[i][j][l][p] = Scatter[i - 1][j][l][s]
                            
                    elif (p == 1):
                        if (j == 0):
                            Pressure[i][j][l][p] = R_left * Scatter[i][j][l][p]
                        else:
                            Pressure[i][j][l][p] = Scatter[i][j - 1][l][s]
                            
                    elif (p == 2):
                        if (l == 0):
                            Pressure[i][j][l][p] = R_bottom * Scatter[i][j][l][p]
                        else:
                            Pressure[i][j][l][p] = Scatter[i][j][l - 1][s]
                            
                    elif (p == 3):
                        if (i == (Scatter.shape[0] - 1)):
                            Pressure[i][j][l][p] = R_back * Scatter[i][j][l][p]
                        else:
                            Pressure[i][j][l][p] = Scatter[i + 1][j][l][s]
                            
                    elif (p == 4):
                        if (j == (Scatter.shape[1] - 1)):
                            Pressure[i][j][l][p] = R_right * Scatter[i][j][l][p]
                        else:
                            Pressure[i][j][l][p] = Scatter[i][j + 1][l][s]
                            
                    elif (p == 5):
                        if (i == (Scatter.shape[2] - 1)):
                            Pressure[i][j][l][p] = R_top * Scatter[i][j][l][p]
                        else:
                            Pressure[i][j][l][p] = Scatter[i][j][l + 1][s]
                            
    return Pressure, Scatter * 0


def scatter_node(pressure):
    S = np.zeros(6)
    s_matrix = np.multiply(pressure, np.ones((len(pressure), len(pressure))) + np.eye(len(pressure)) * (-3)) * 1/3
    for i in range(len(S)):
        S[i] = np.round(np.sum(s_matrix[i]),2)
    return S

def _create_S_grid(Pressure, Scatter):
    for i in range(Pressure.shape[0]):
        for j in range(Pressure.shape[1]):
            for l in range(Pressure.shape[2]):
                Scatter[i][j][l] = scatter_node(Pressure[i][j][l])
                
    return Pressure * 0, Scatter


def sum_pressure(pressure):
    """
    :param pressure: Pressure array
    :return: return the total pressure at spesific node
    """
    return 0.3 * np.sum(pressure)

def cuboid_data(center, size=(1,1,1)):
    o = [a - b / 2 for a, b in zip(center, size)]
    
    l, w, h = size
    x = [[o[0], o[0] + l, o[0] + l, o[0], o[0]],  # x coordinate of points in bottom surface
         [o[0], o[0] + l, o[0] + l, o[0], o[0]],  # x coordinate of points in upper surface
         [o[0], o[0] + l, o[0] + l, o[0], o[0]],  # x coordinate of points in outside surface
         [o[0], o[0] + l, o[0] + l, o[0], o[0]]]  # x coordinate of points in inside surface
    y = [[o[1], o[1], o[1] + w, o[1] + w, o[1]],  # y coordinate of points in bottom surface
         [o[1], o[1], o[1] + w, o[1] + w, o[1]],  # y coordinate of points in upper surface
         [o[1], o[1], o[1], o[1], o[1]],          # y coordinate of points in outside surface
         [o[1] + w, o[1] + w, o[1] + w, o[1] + w, o[1] + w]]    # y coordinate of points in inside surface
    z = [[o[2], o[2], o[2], o[2], o[2]],                        # z coordinate of points in bottom surface
         [o[2] + h, o[2] + h, o[2] + h, o[2] + h, o[2] + h],    # z coordinate of points in upper surface
         [o[2], o[2], o[2] + h, o[2] + h, o[2]],                # z coordinate of points in outside surface
         [o[2], o[2], o[2] + h, o[2] + h, o[2]]]                # z coordinate of points in inside surface
    return x, y, z

def plotCubeAt(pos=(0,0,0), c="b", alpha=0.1, ax=None):
    
    X, Y, Z = cuboid_data( (pos[0],pos[1],pos[2]) )
    ax.plot_surface(X, Y, Z, color=c, rstride=1, cstride=1, alpha=0.1)

def plotMatrix(ax, x, y, z, data, cmap="jet", cax=None, alpha=0.1): 
    norm = matplotlib.colors.Normalize(vmin=data.min(), vmax=data.max())
    colors = lambda i,j,k : matplotlib.cm.ScalarMappable(norm=norm,cmap = cmap).to_rgba(data[i,j,k]) 
    for i, xi in enumerate(x):
            for j, yi in enumerate(y):
                for k, zi, in enumerate(z):
                    plotCubeAt(pos=(xi, yi, zi), c=colors(i,j,k), alpha=alpha,  ax=ax)



    if cax !=None:
        cbar = matplotlib.colorbar.ColorbarBase(cax, cmap=cmap,
                                norm=norm,
                                orientation='vertical')  
        cbar.set_ticks(np.unique(data))
        
        cbar.solids.set(alpha=alpha) 




if __name__ == '__main__':
    print("Master thesis\t TFE4940")
    R_top = 1
    R_bottom = 1
    R_front = 1
    R_back = 1
    R_left = 1
    R_right = 1
    
    nx = 5
    ny = 3
    nz = 3
    
    tlm, P_grid, S_grid = _create_TLM(nx,ny,nz)
    
    S_grid[1][1][1] = [1,1,1,1,1,1]
    
    heatmap = np.zeros((tlm.shape[0], tlm.shape[1], tlm.shape[2]))
    
    for i in range(5):
        
        P_grid, S_grid = _create_S_grid(P_grid, S_grid)
        P_grid, S_grid = _create_P_grid(S_grid, P_grid)
        
        
        for i in range(P_grid.shape[0]):
            for j in range(P_grid.shape[1]):
                for l in range(P_grid.shape[2]):
                    heatmap[i][j][l] = sum_pressure(P_grid[i][j][l])
        print(heatmap)
        
        
        
    
    
""" 
Checking with the diffuse field theory - Sabines equation - Transfer function - Modal shapes

Check one point - Define special cases, wall

Oria - Trevor Cox

Create big mesh - Compensate for sound speed
"""
