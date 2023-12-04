import numpy as np
import pylab as pl
import pandas as pd

def main():

    df = read()

    

    #gri = df[df["mech"] == "gri"]
    #h2 = df[df["mech"] == "h2"]
    
    fig, ax = pl.subplots()
    plotMechanism(ax,df, "gri")
    pl.legend(loc="best")
    pl.show()

    fix, ax = pl.subplots()
    plotMechanism(ax,df, "h2")
    pl.legend(loc="best")
    pl.show()


def plotMechanism(ax, df, mech_name):

    mech = df[df["mech"] == mech_name]

    procs = np.array(mech["proc"],dtype=int)
    tgpu = np.array(mech["tgpu"], dtype=float)
    tcpu = np.array(mech["tcpu"], dtype=float)

    idx = np.argsort(procs)
    procs = procs[idx]
    tgpu = tgpu[idx]
    tcpu = tcpu[idx]

    ax.plot(procs, tgpu, marker="x", label="gpu", color="black")
    ax.plot(procs, tcpu, marker="x", label="cpu", color="red")


    if (mech_name=="gri"):
        ax.set_title("Diesel combustion, 53 species, 324 reactions")

    if (mech_name=="h2"):
        ax.set_title("H2 combustion, 10 species, 26 reactions")


    ax.set_xlabel("CPU process count")
    ax.set_ylabel("Runtime")
    #ax.set_ylim(0, 1000)

    #print(procs)



def read():

    f = open("results.dat", "r")
    
    proc = []
    tcpu = []
    tgpu = []
    mech = []
    ode = []
    ncell = []
    for line in f:
        if (not line.startswith("#")):
            
            ll = line.split()
            proc.append(ll[0])
            tcpu.append(ll[1])
            tgpu.append(ll[2])
            mech.append(ll[3])
            ode.append(ll[4])
            ncell.append(ll[5])

        

    d = {"proc":proc, "tcpu":tcpu, "tgpu":tgpu, "mech":mech, "ode":ode, "ncell":ncell}

    return pd.DataFrame(d)


    #return 0

    #df = pd.read_csv("results.dat", sep=" ")
    #dat = np.loadtxt("results.dat")
    #return df
    #return dat
    
main()

