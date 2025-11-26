import argparse

def parse_args():
    """
    Loads Arguments for the script.
    """

    parser = argparse.ArgumentParser(
        description="Python script that finds line emission-like features in an ALMA data cube")
    
    parser.add_argument('-Cube', type=str, required=True,help = 'Path to the Cube fits file where the search will be done')
    parser.add_argument('-LineSearchPath', type=str, default='OutputLineSearch', required=False , help = 'Directory where the outputs will be saved [Default:LineSearchPath]')
    parser.add_argument('-SimulationPath', type=str, default='Simulation', required=False , help = 'Directory where the simulations should be found [Default:Simulation]')
    parser.add_argument('-MaxSigmas', type=int, default = 10, required=False,help = 'Maximum number of channels to use as sigma value for the spectral Gaussian convolution. [Default:10]')
    parser.add_argument('-MinSN', type=float, default = 5.0, required=False,help = 'Minimum S/N value to save in the outputs. A good value depends on each data cube, reasonable values are bettween 3.5 and 6 [Default:5.0]')
    parser.add_argument('-SurveyName', type=str, default='Survey', required=False , help = 'Name to identify the line candidates [Default:Survey]')
    parser.add_argument('-LimitN', type=float, default='20.0', required=False , help = 'Limit for the number of detection above certain S/N to be used in the fitting of the negative counts [Default:20]')
    parser.add_argument('-LegendFontSize', type=float, default='10.0', required=False , help = 'Fontsize fot the figures legends [Default:10]')
    parser.add_argument('-UserEPS', type=str, default='False',choices=['True','False'], required=False , help = 'Whether to use EPS value entered from user otherwise use number of pixels per bmaj [Default:False]')
    parser.add_argument('-EPS', type=float, default=5.0, required=False , help = 'EPS value to use if User sets -UserEPS to True [Default:5.0]')
    parser.add_argument('-UseFactorLEE', type=str, default='False',choices=['True','False'], required=False , help = 'Whether to correct by the Look-Elsewhere effect using only a factor (1 + (ln(MaxSigmas -1) + 1/(2*(MaxSigmas -1))) + 0.577) [Default:False]')
    parser.add_argument('-SkipIndChan', type=str, default='False',choices=['True','False'], required=False , help = 'Whether to skip individual channels. Activate when weird individual channel peaks[Default:False]')
    parser.add_argument('-backend', type=str, default='CPU',choices=['CPU','GPU'], required=False , help = 'A')
    parser.add_argument("-NSigmaSpatial", type=int, default = 0, required=False, help = 'Max spatial sigma for convolution. [Default:0]')

    args = parser.parse_args()

    return args