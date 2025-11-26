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


def check_args(args):
    print(20*'#','Checking inputs....',20*'#')
    check_file(args.Cube)
    max_sigma(args.MaxSigmas)
    min_SN(args.MinSN)