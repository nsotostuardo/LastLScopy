from ..utils.helpers import check_args
from ..core.functions import pixels_BMAJ, get_LEE_factor, make_figure, get_binning
from ..core.functions import n_positives, plot_N_positive_negative
from ..utils.decorators import time_function
from ..core.poisson import get_poisson_estimates
from ..core.pipeline import Positive, Negative

def get_pipeline(args):
    if args.backend == 'CPU':
        from ..backends.Scipy import SciPyPipeline
        return SciPyPipeline(args)
    elif args.backend == 'GPU':
        from ..backends.Cupy import CuPyPipeline
        return CuPyPipeline(args)
    else:
        raise ValueError("Invalid Backend")
    

def main(args):
    try:
        check_args(args)
    except (FileNotFoundError, ValueError) as error:
        print(error)
        exit()

    pp_BMAJ = pixels_BMAJ(args)
    FactorLEE = get_LEE_factor(args) 
    ax = make_figure()

    bins = get_binning(args.MinSN)

    pipeline = get_pipeline(args)
    positive_sources = Positive(args)
    negative_sources = Negative(args)

    for sigmas in range(args.MaxSigmas):
        for spatial in range(args.NSigmaSpatial):
            if sigmas == 0 and args.SkipIndChan=='True':
                print('skipping individual channels....')
                continue
            print(50*'-')

            positive_sources.current_sigma = sigmas
            negative_sources.current_sigma = sigmas

            positive_sources.get_sources_file(sigmas, spatial, pp_BMAJ)
            negative_sources.get_sources_file(sigmas, spatial, pp_BMAJ)

            positive_sources.real_SN()
            negative_sources.real_SN()

            print('for sigma', sigmas)
            positive_sources.get_ys(bins)
            negative_sources.get_ys(bins)

            positive_sources.plot_y(sigmas, bins, ax)
            negative_sources.plot_y(sigmas, bins, ax)

            positive_sources.fix_SN()
            negative_sources.fix_SN()

            estimates = get_poisson_estimates(bins, positive_sources.SN, negative_sources.SN, args.LimitN, args.MinSN)
        
            NPositive_e1, NPositive_e2 = n_positives(estimates["nPositive"])
            plot_N_positive_negative(estimates, NPositive_e1, NPositive_e2, sigmas, spatial, args)
            
            positive_sources.get_sources_total_pos(estimates, sigmas, spatial)
            negative_sources.get_sources_total_pos(estimates, sigmas, spatial)

    positive_sources.clustering(pipeline, pp_BMAJ)
    negative_sources.clustering(pipeline, pp_BMAJ)

    positive_sources.get_final_candidates(pp_BMAJ, FactorLEE)
    negative_sources.get_final_candidates(pp_BMAJ, FactorLEE)
        