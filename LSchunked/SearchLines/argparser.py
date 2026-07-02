import argparse
import os
from astropy.io import fits 

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
    
    #parser.add_argument('-rms', type=str, default = 'Numpy', choices=['Numpy', 'Numba', 'Cupy'], required=False, 
    #                    help = 'Method used for the RMS filter. [Default:Numpy]'
    #                    )
    
    parser.add_argument("-NSigmaSpatial", type=int, default = 0, required=False,
                        help = 'Max spatial sigma for convolution. [Default:0]'
                        )
    
    parser.add_argument('-EPS', type=float, default=5.0, required=False , help = 'EPS value to use if User sets -UserEPS to True [Default:5.0]')
    parser.add_argument('-UserEPS', type=str, default='False',choices=['True','False'], required=False , help = 'Whether to use EPS value entered from user otherwise use number of pixels per bmaj [Default:False]')
    parser.add_argument('-FractionEPS', type=float, default=1.0, required=False , help = 'Fraction of the EPS value to be used, must be from 0 to 1 inclusive [Default:1]')

    parser.add_argument('-Chunk', type=int, default=256,
                        help='Chunk size in XY')

    args = parser.parse_args()

    return args


def check_file(path:str):
    """Verifies file existance"""
    if not os.path.isfile(path):
        raise FileNotFoundError('*** Cube',path,'not found ***\naborting..')
    else:
        print('*** Cube',path,'found ***')

def check_output_path(path:str):
    """Verifies output directory existance"""
    if not os.path.exists(path):
        print('*** Creating Directory',path,' ***')
        os.mkdir(path)
    else:
        raise FileNotFoundError('*** Directory',path,'exists ***\naborting..')
        

def max_sigma(sigma:int):
    """Verifies the minimum sigma value for MaxSigma"""
    if sigma < 1:
        raise ValueError('*** The value for MaxSigmas of',sigma,'is too small ***\naborting..')
    else:
        print('*** The value for MaxSigmas of',sigma,'is ok ***')


def min_SN(SN:float):
    """Verifies the value for Minimum S/N """
    if SN<0:
         raise ValueError('*** The value for MinSN of',SN,'has to be positive ***\naborting..')
    else:
        print('*** The value for MinSN of',SN,'is ok ***')

def use_mask(mask:bool, path:str, MaskSN:float):
    if mask:
        print('*** Will use Continuum image to create mask ***')
        if not os.path.isfile(path):
            raise FileNotFoundError('*** Continuum Image',path,'not found ***\naborting..')
        
        else:
            print('*** Continuum Image',path,'found ***')
            if MaskSN < 0:
                raise ValueError('*** The value for MaskSN of',MaskSN,'has to be positive ***\naborting..')
            
            else:
                print('*** The value for MaskSN of',MaskSN,' is ok ***')

    else:
        print('*** Will not use Continuum image to create mask ***')



def channel_width(cube:str, kernel:str):
    Header = fits.open(cube)[0].header
    RefFrequency = Header['CRVAL3']
    ChannelSpacing = Header['CDELT3']
    ApproxChannelVelocityWidth = (abs(ChannelSpacing)/RefFrequency)*3e5
    ApproxMaxSigmas = int((1000.0/ApproxChannelVelocityWidth)/2.35) + 1
    if kernel.lower() == 'gaussian':
        print('*** MaxSigmas should be of the order of ',ApproxMaxSigmas,'to detect a line width FWHM ~ 1000 km/s ***')
    else:
        ApproxMaxSigmas = int ((1000.0/ApproxChannelVelocityWidth))+1
        print('*** MaxSigmas should be of the order of ',ApproxMaxSigmas,'to detect a line width of ~ 1000 km/s for the Tophat Kernel (considering the reference frequency CRVAL3)***')
    
def FractionEPS_check(fraction:float):
    """
    Checks if FractionEPS is within valid range [0,1].

    Parameters
    ----------
    fraction : float
        Value to validate.

    Raises
    ------
    ValueError
        If fraction is outside [0,1].
    """
    if not 0 <= fraction <= 1:
       raise ValueError(
           f"FractionEPS must be between 0 and 1. Received: {fraction}"
           )


def check_args(args):
    print(20*'#','Checking inputs....',20*'#')
    check_file(args.Cube)
    check_output_path(args.OutputPath)
    max_sigma(args.MaxSigmas)
    min_SN(args.MinSN)
    use_mask(args.UseMask, args.ContinuumImage, args.MaskSN)
    channel_width(args.Cube, "gaussian")#args.Kernel)
    FractionEPS_check(args.FractionEPS)
