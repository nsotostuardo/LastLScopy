import os
from astropy.io import fits

def check_file(path:str):
    """Verifies file existance"""
    if not os.path.isfile(path):
        raise FileNotFoundError('*** Cube',path,'not found ***\naborting..')
    else:
        print('*** Cube',path,'found ***')

def check_output_path(path:str):
    """Verifies output directory existance"""
    if not os.path.isfile(path):
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

def check_kernel(kernel:str):
    """Checks the type of kernel to be used for the convolution"""
    if kernel.lower() == 'gaussian':
        print('*** Using Gaussian Kernel ***')
    elif kernel.lower() == 'tophat':
        print('*** Using Tophat Kernel ***')
    else:
        raise ValueError('***Selected Kernel must be Gaussian or Tophat',kernel,'***\naborting..')

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
    check_kernel(args.Kernel)
    channel_width(args.Cube, args.Kernel)
    FractionEPS_check(args.FractionEPS)

def check_separability(args):
    """
    Checks separability for spatial convolution
    """
    # ['CPU', 'CPUThread', 'GPUkernel', 'GPUfilter']
    if args.backend == "CPU" or args.backend == "CPUThread":
        return True
    else:
        return False