import numpy as np
import matplotlib.pyplot as plt
import scipy
import cmath
from sys import exit
from numpy import pi, exp, real, imag, linspace
from mpl_toolkits.mplot3d import Axes3D


# work in progress
def MultiChoice():
    print("Now you can choose between this options:")
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


def real_pos():
    print("die")


def real_neg():
    print("die")


def im_pos():
    print("die")


def im_neg():
    print("die")


def re_comp():
    print("die")


def im_comp():
    print("die")


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

    phi = (1 + np.sqrt(5)) / 2
    psi = -1 / phi

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