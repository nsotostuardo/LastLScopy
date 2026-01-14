from ..utils.helpers import check_args, check_separability
from ..core.functions import GetMinSNEstimate, save_positives, save_negatives, open_cube, pixels_BMAJ
from ..utils.decorators import time_function

def get_pipeline(args):
    if args.backend == 'CPU':
        from ..backends.Scipy import SciPyPipeline
        return SciPyPipeline(args)
    elif args.backend == 'CPUThread':
        from ..backends.Numba import NumbaPipeline
        return NumbaPipeline(args)
    elif args.backend == 'GPUfilter':
        from ..backends.Cupy import CuPyPipeline
        return CuPyPipeline(args)
    elif args.backend == 'GPUkernel':
        from ..backends.CCUDA import CudaPipeline
        return CudaPipeline(args)
    else:
        raise ValueError("Invalid Backend")
    
def get_rms_pipeline(args):
    if args.rms == 'Numpy':
        from ..backends.Scipy import SciPyRMS
        return SciPyRMS(args)
    elif args.rms == 'Numba':
        from ..backends.Numba import NumbaRMS
        return NumbaRMS(args)
    elif args.rms == 'Cupy':
        from ..backends.Cupy import CuPyRMS
        return CuPyRMS(args)
    else:
        raise ValueError("Invalid Backend")


def main(args):
    pipeline = get_pipeline(args)
    rms_pipe = get_rms_pipeline(args)

    try:
        check_args(args)
    except (FileNotFoundError, ValueError) as error:
        print(error)
        exit()

    try:
        GetMinSNEstimate(args.Cube)
    except:
        print('*** problems reading header ***')
     
    data = open_cube(args.Cube) 
    EPS = pixels_BMAJ(args)

    if False:#check_separability(args):
        for spatial in range(args.NSigmaSpatial + 1):
            spatial_modified = spatial * EPS 
            filtered_data = pipeline.gaussian_filtering(data, 0, spatial_modified)
            for sigmas in range(args.MaxSigmas):
                print(100*'#')
                print(f'Starting search of lines with parameter for filter equal to {sigmas} channels')
                filtered_data = pipeline.gaussian_filtering(filtered_data, sigmas, 0)
                filtered_data = rms_pipe.rms_filtering(filtered_data, args.UseMask)
                save_positives(filtered_data, args.MinSN, args.OutputPath, sigmas, spatial)
                save_negatives(filtered_data, args.MinSN, args.OutputPath, sigmas, spatial)

    else:
        for sigmas in range(args.MaxSigmas):
            print(100*'#')
            print(f'Starting search of lines with parameter for filter equal to {sigmas} channels')
            for spatial in range(args.NSigmaSpatial + 1):

                spatial_modified = spatial * EPS
                print(f'                          Using spatial of {spatial}                         ')  
                filtered_data = pipeline.gaussian_filtering(data, sigmas, spatial_modified)
                filtered_data = rms_pipe.rms_filtering(filtered_data, args.UseMask)
                save_positives(filtered_data, args.MinSN, args.OutputPath, sigmas, spatial)
                save_negatives(filtered_data, args.MinSN, args.OutputPath, sigmas, spatial)
    