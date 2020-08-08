import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import mpmath as mp
from sys import exit
from numpy import exp, real, imag, linspace, arange
from mpl_toolkits.mplot3d import Axes3D
#here you need to install mpmath with pip or conda

# normalizing and applying color map to a given ndarray
def sufacecolor(angpart):
    cmap = cm.jet
    norm = plt.Normalize(vmin=angpart.min(), vmax=angpart.max())
    surfang = cmap(norm(angpart))

    return surfang

# calculates phi and psi global variables
phi = ((1 + np.sqrt(5)) / 2)
psi = (-1 / phi)

# here you can choose between some visualization options
def MultiChoice():
    print("Now you can choose between these options:")
    print("1\tThe plot of the positive real input (x>0; y=0)")
    print("2\tThe plot of the negative real input (x<0; y=0)")
    print("3\tThe plot of the positive imaginary input (x=0; y>0)")
    print("4\tThe plot of the negative imaginary input (x=0; y<0)")
    print("5\tThe plot of the real part of the complex input (x,y)")
    print("6\tThe plot of the imaginary part of the complex input (x,y)\n")
    print("Please, select the number of your option.")
    print("If you prefer nothing, hit Ctrl+C and a sad face :(\n")

    opt = input(">>> ")

    if opt == "1":
        return real_pos()
    elif opt == "2":
        return real_neg()
    elif opt == "3":
        return  im_pos()
    elif opt == "4":
        return im_neg()
    elif opt == "5":
        return re_comp()
    elif opt == "6":
        return im_comp()
    else:
        print("That's not a valid option, try again.")
        return MultiChoice()

# positive real valued complex plane of the Fibonacci function
def real_pos():
    X = linspace(0.0, 7.0, 101, dtype=np.complex)
    kk = 0
    F_x = np.zeros(np.size(X), dtype=np.complex)
    Psi = np.complex(psi)

    for i in X:
        F_x[kk] = ((phi**(i)) - (Psi**(i))) / np.sqrt(5)
        kk = kk + 1

    plt.figure()
    plt.plot(real(F_x), imag(F_x))
    plt.xlabel('Re')
    plt.ylabel('Im')
    plt.title('Positive Fibonacci function for real n')

    plt.show()

# negative real valued complex plane of the Fibonacci function    
def real_neg():
    X = linspace(-7.0, 0.0, 142, dtype=np.complex)
    kk = 0
    F_x = np.zeros(np.size(X), dtype=np.complex)
    Psi = np.complex(psi)

    for i in X:
        F_x[kk] = ((phi**(i)) - (Psi**(i))) / np.sqrt(5)
        kk = kk + 1

    plt.figure()
    plt.plot(real(F_x), imag(F_x))
    plt.xlabel('Re')
    plt.ylabel('Im')
    plt.title('Negative Fibonacci function for real n')

    plt.show()

# positive imaginary valued complex plane of the Fibonacci function
def im_pos():
    X = 1j * linspace(0.0, 15.0, 151, dtype=np.complex)
    kk = 0
    F_x = np.zeros(np.size(X), dtype=np.complex)
    Psi = np.complex(psi)

    for i in X:
        F_x[kk] = ((phi**(i)) - (Psi**(i))) / np.sqrt(5)
        kk = kk + 1

    plt.figure()
    plt.plot(real(F_x), imag(F_x))
    plt.xlabel('Re')
    plt.ylabel('Im')
    plt.title('Positive Fibonacci function for imaginary n')

    plt.show()

# negative imaginary valued complex plane of the Fibonacci function
def im_neg():
    X = 1j * linspace(-15.0, 0.0, 151, dtype=np.complex)
    kk = 0
    F_x = np.zeros(np.size(X), dtype=np.complex)
    Psi = np.complex(psi)

    for i in X:
        F_x[kk] = ((phi**(i)) - (Psi**(i))) / np.sqrt(5)
        kk = kk + 1

    plt.figure()
    plt.plot(real(F_x), imag(F_x))
    plt.xlabel('Re')
    plt.ylabel('Im')
    plt.title('Negative Fibonacci function for imaginary n')

    plt.show()

# real part of the complex valued Fibonacci function
def re_comp():
    mp.dps = 5
    Phi = ((1 + mp.sqrt(5)) / 2)
    Psi = (-1 / phi)

    f = lambda z: real((mp.power(Phi,z) - mp.power(Psi,z)) / mp.sqrt(5))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    X = arange(-5, 5, 0.125)
    Y = arange(-5, 5, 0.125)
    X, Y = np.meshgrid(X, Y)
    xn, yn = X.shape
    W = X*0
    for xk in range(xn):
        for yk in range(yn):
            try:
                z = complex(X[xk,yk],Y[xk,yk])
                w = float(f(z))
                if w != w:
                    raise ValueError
                W[xk,yk] = w
            except (ValueError, TypeError, ZeroDivisionError):
                pass

    SW = sufacecolor(W)
    ax.plot_surface(X, Y, W, rstride=1, cstride=1, facecolors=SW)
    ax.set_xlabel('Re')
    ax.set_ylabel('im')
    ax.set_zlabel('Re(F(z))')
    plt.title('Real part of Complex-Valued Fibonacci')

    M = cm.ScalarMappable(cmap=cm.jet)
    M.set_array(SW)
    plt.colorbar(M)

    plt.show()  

# imaginary part of the complex valued Fibonacci function
def im_comp():
    mp.dps = 5
    Phi = ((1 + mp.sqrt(5)) / 2)
    Psi = (-1 / phi)

    f = lambda z: imag((mp.power(Phi,z) - mp.power(Psi,z)) / mp.sqrt(5))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    X = arange(-5, 5, 0.125)
    Y = arange(-5, 5, 0.125)
    X, Y = np.meshgrid(X, Y)
    xn, yn = X.shape
    W = X*0
    for xk in range(xn):
        for yk in range(yn):
            try:
                z = complex(X[xk,yk],Y[xk,yk])
                w = float(f(z))
                if w != w:
                    raise ValueError
                W[xk,yk] = w
            except (ValueError, TypeError, ZeroDivisionError):
                pass

    SW = sufacecolor(W)
    ax.plot_surface(X, Y, W, rstride=1, cstride=1, facecolors=SW)
    ax.set_xlabel('Re')
    ax.set_ylabel('im')
    ax.set_zlabel('Im(F(z))')
    plt.title('Imaginary part of Complex-Valued Fibonacci')

    M = cm.ScalarMappable(cmap=cm.jet)
    M.set_array(SW)
    plt.colorbar(M)

    plt.show()

# Main function. It calls the other functions and calculates the corresponding Fibonacci numbeer.
def gigafibonaci():
    print("""This program takes as an input any number
    (real or complex) and returns the corresponding fibonacci number.
    You can also ask for a plot... There are a lot of options :)
    If you want to enter a real number, type '0' in the imaginary part.
    If you want to enter a complex number you can type any numbers in both parts:\n""")

    x = float(input("Enter the real part x = "))
    y = float(input("Enter the imaginary part y = "))
    if (type(x) == float or type(x) == int) and (type(y) == float or type(y) == int):
        numb = complex(x,y)
    else:  
        exit(0)

    
    F_n = ((phi**numb) - (psi**numb)) / np.sqrt(5)
    
    if y == 0:
        numb = real(numb)
        F_n = real(F_n)
        print(f"The Fibonacci number asociated to {numb} is F_{numb} = {F_n}\n")
    else:
        print(f"The Fibonacci number asociated to {numb} is F_{numb} = {F_n}\n")

    input("Press the Intro key to continue: ")

    
    MultiChoice()
    
    
gigafibonaci()
