import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm

def mkplot(function, override=False, cmap="viridis", id=999):

    plt.rcParams.update({
        "font.family": "Linux Biolinum"
    })

    fig, ax = plt.subplots(1, 1, figsize=(2.5, 2), dpi=300)
    min = -3.1
    max = 3.1
    step = (max - min) / (26+1)
    x_0 = x_1 = x_2 = np.arange(min, max, step)
    X0, X1= np.meshgrid(x_0, x_1)
    # X0 = np.reshape(X0, -1)
    # X1 = np.reshape(X1, -1)


    Y = 2 * np.pi * X0 /(X1)
    #Y_Res = Y
    # Y_Res = Y  / 9.6
    #Y_Res = Y * X1/9.6
    Y_Res = function(Y, X1)
    norm = cm.colors.Normalize(vmax=abs(Y_Res).max(), vmin=-abs(Y_Res).max())


    #Y_Res = 0*Y_Res +1
    if override:
        norm = cm.colors.Normalize(vmax=1.9, vmin=-1.9)



    #cmap = cm.PRGn
    # levels = np.arange(-2.0, 1.601, 0.4)
    #print(Y_Res)
    #print(Y_Res.shape)

    # cset1 = ax.contourf(X0, X1, Y_Res, norm=norm)
    asdf = ax.imshow(Y_Res, norm=norm, origin="lower", extent=(min, max, min, max), cmap=cmap, interpolation="bicubic")

    cbar = fig.colorbar(asdf)

    # cbar = fig.colorbar(cset1)
    ticks = cbar.get_ticks()
    print(ticks)
    delta = abs(Y_Res).max()
    #cbar.set_ticks(np.arange(-delta, 1.5*delta, delta))

    cbar.ax.set_xlabel('$y$')
    ax.set_xlabel('$x_0$')
    ax.set_ylabel('$x_1$')

    # ax.contour(X0, X1, Y, zdir='y', offset=40, cmap='coolwarm')

    # ax.set(xlim=(-4, 4), ylim=(-4, 4), zlim=(-4, 4),
    #        xlabel='$x_0$', ylabel='$x_1$', zlabel='y')
    fig.tight_layout(rect=(0, 0, 1, 1))
    plt.savefig("plot_residual_" + str(id) + ".png")
    plt.show()

cmaps = ["RdYlGn"]
for cmap in cmaps:

    mkplot(lambda Y, X1: Y, cmap=cmap, id=0)
    mkplot(lambda Y, X1: Y  / 9.6, cmap=cmap, id=1)
    mkplot(lambda Y, X1: Y * X1/9.6, cmap=cmap, id=2)
    mkplot(lambda Y, X1: Y * 0 + 1, override=True, cmap=cmap, id=3)
