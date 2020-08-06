import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numpy import array
from scipy import special
from sys import exit

# assoc_leg(degree,xvec) computes the associated Legendre functions of degree 
# 'degree' and order m = 0, 1, ..., 'degree' evaluated for each element in 'xvec'
def assoc_leg(degree, xvec):
    P_lm = np.zeros((degree+1, np.size(xvec)))
    for i in range(0, degree+1):
        P_lm[i,:] = special.lpmv(i, degree, xvec)

    return P_lm

# normalizing and applying color map to a given ndarray
def sufacecolor(angpart):
    cmap = plt.cm.jet
    norm = plt.Normalize(vmin=angpart.min(), vmax=angpart.max())
    surfang = cmap(norm(angpart))

    return surfang


def hydrogen():
    print("Remember, the Quantum Numbers MUST be integers.")
    n = input("Principal Quantum Number n = ")
    l = input("Angular Quantum Number l = ")
    m = input("Magnetic Quantum Number m = ")

    n = int(n)
    l = int(l)
    m = int(m)
    
    # checking for correct input values and valid quantum nunbers
    if n > 0 and (n % 1) == 0:
        print("n is valid.")
    else:
        print("n must be greather than zero.")
        exit(0)
    
    if (l % 1) == 0 and l >= 0:
        print("l is valid.")
    else:
        print("l must be positive or zero.")
        exit(0)
    
    if (m % 1) == 0:
        print("m is valid.")
    else:
        print("m must to be an integer.")
        exit(0)
    
    if n - 1 >= l and l >= abs(m):
        print("n, l and m are ok.")
    else:
        print("l can take values between 0 and n-1.\nm can take values between -l and l.\n check your input :)")
        exit(0)

    # setting the variables theta and phi as ndarrays
    N = max([30, l * 6])
    theta = (np.arange(N+1).reshape(-1,1)) * (np.pi/N)
    phi = np.arange(-N,N+1).reshape(1,-1) * (np.pi/N)
    mm = abs(m)


    # calculation of the espherical harmonics Ylm using the associated Legendre functions 
    gamma = 1/(l+1)
    alf = assoc_leg(l, np.cos(theta.T))
    Ylm = np.multiply(alf[mm, :].reshape(1, -1).T, np.cos(mm*phi))
    
    # calculation of the radial part ra using the generalized Laguerre polynomials
    [A,B] = np.shape(Ylm)
    xx = np.linspace(-50e-10,50e-10,B*A).reshape(A,B)
    ra = np.array(special.eval_genlaguerre(n, 0, xx))

    # finally getting the wave function (non-complex form)
    Y_lm = np.array(np.power(np.abs(Ylm), gamma))
    wf = ra * Y_lm

    # changing to cartesian coordinates
    X = np.multiply(wf, (np.dot(np.sin(theta), np.cos(phi))))
    Y = np.multiply(wf, (np.dot(np.sin(theta), np.sin(phi))))
    Z = np.multiply(wf, (np.multiply(np.cos(theta), np.ones(np.size(phi)))))

    # surface of Re Ylm (theta,phi)/r^(l+1) = 1
    Ylm = sufacecolor(Ylm)

    # plot stuff
    ax = plt.axes(projection ='3d')
    ax.plot_surface(X, Y, Z, rcount = 31, ccount = 61, facecolors=Ylm)
   
    plt.title('Hydrogen orbitals with Quantum Numbers\nn = {}'.format(n) +' l = {}'.format(l) +' m = {}'.format(m))

    plt.show()


hydrogen()
