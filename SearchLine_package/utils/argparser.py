import argparse
import os

def parse_args():
    """
    Loads Arguments for the script.
    """

    parser = argparse.ArgumentParser(
        description="Python script that finds line emission-like features in an ALMA data cube"
        )
    
    parser.add_argument('-Cube', type=str, required=True,
                        help = 'Path to the Cube fits file where the search will be done'
                        )
    
    parser.add_argument('-OutputPath', type=str, default='OutputLineSearch', required=False,
                        help = 'Directory where the outputs will be saved, if exists the codes finished, otherwise will be created [Default:OutputLineSearch]'
                        )

    parser.add_argument('-MinSN', type=float, default = 5.0, required=False,
                        help = 'Minimum S/N value to save in the outputs. A good value depends on each data cube, reasonable values are bettween 3.5 and 6 [Default:5.0]'
                        )

    parser.add_argument('-MaxSigmas', type=int, default = 10, required=False,
                        help = 'Maximum number of channels to use as sigma value for the spectral Gaussian convolution. [Default:10]'
                        )

    parser.add_argument('-ContinuumImage', type=str, default = 'Continuum.fits', required=False,
                        help = 'Continuum image to use to create a mask. [Default:Continuum.fits]'
                        )

    parser.add_argument('-MaskSN', type=float, default = 5.0, required=False,
                        help = 'S/N value to use as limit to create mask from continuum image. [Default:5.0]'
                        )

    parser.add_argument('-UseMask', type=bool, default = False,choices=[True,False], required=False,
                        help = 'S/N value to use as limit to create mask from continuum image. [Default:5.0]'
                        )

    parser.add_argument('-Kernel', type=str, default = 'Gaussian',choices=['Gaussian','gaussian','Tophat','tophat'], required=False,
                        help = 'Type of kernel to use for the convolution. [Default:Gaussian]'
                        )

    parser.add_argument('-backend', type=str, default = 'CPU', choices=['CPU', 'CPUThread', 'GPUkernel', 'GPUfilter'], required=False,
                        help = 'Backend to be used for convolution: CPU (scipy or multi Threading) or GPU (custome C++ kernel or Gaussian Filter). [Default:CPU]'
                        )
    parser.add_argument('-rms', type=str, default = 'Numpy', choices=['Numpy', 'Numba', 'Cupy'], required=False, 
                        help = 'Method used for the RMS filter. [Default:Numpy]'
                        )
    
    parser.add_argument("-NSigmaSpatial", type=int, default = 0, required=False,
                        help = 'Max spatial sigma for convolution. [Default:0]'
                        )
    
    parser.add_argument('-EPS', type=float, default=5.0, required=False , help = 'EPS value to use if User sets -UserEPS to True [Default:5.0]')
    parser.add_argument('-UserEPS', type=str, default='False',choices=['True','False'], required=False , help = 'Whether to use EPS value entered from user otherwise use number of pixels per bmaj [Default:False]')
    parser.add_argument('-FractionEPS', type=float, default=1.0, required=False , help = 'Fraction of the EPS value to be used, must be from 0 to 1 inclusive [Default:1]')


    args = parser.parse_args()

    return args