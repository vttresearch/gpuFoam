import numpy as np
import pylab as pl
import pandas as pd

def main():

    dat = read()
    print(dat)

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

