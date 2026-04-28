from argparser import check_args, parse_args
from itertools import product
from Chunks import SearchLineChunked
from astropy.io import fits
import numpy as np

def pixels_BMAJ(args):
	if args.UserEPS=='False':
		PixelsPerBMAJ = GetPixelsPerBMAJ(args.Cube)
		if args.MaxSigmas == 1:
			PixelsPerBMAJ = 1.0
		print('*** Using EPS value of '+str(PixelsPerBMAJ)+'***')
	else:
		PixelsPerBMAJ = args.EPS
		print('*** Using EPS value of '+str(PixelsPerBMAJ)+'***')
	return PixelsPerBMAJ

def GetPixelsPerBMAJ(CubePath): 
	hdulist =   fits.open(CubePath,memmap=True)
	head = hdulist[0].header
	nchan = head.get('NAXIS3', None)

	try:
		BMAJ = hdulist[1].data.field('BMAJ')
		BMIN = hdulist[1].data.field('BMIN')
		BPA = hdulist[1].data.field('BPA')
	except:
		if nchan is None:
			nchan = hdulist[0].shape[1]
		BMAJ = []
		BMIN = []
		BPA = []
		for i in range(nchan):
			BMAJ.append(head['BMAJ']*3600.0)
			BMIN.append(head['BMIN']*3600.0)
			BPA.append(head['BPA'])
		BMAJ = np.array(BMAJ)
		BMIN = np.array(BMIN)
		BPA = np.array(BPA)
	pix_size = head['CDELT2']*3600.0
	return max(BMAJ/pix_size)

def main(args):
    try:
        check_args(args)
    except Exception as error:
        print(error)
        exit()


    EPS = pixels_BMAJ(args)
    z_list = [spatial for spatial in range(args.NSigmaSpatial + 1)]
    xy_list = [i for i in range(args.MaxSigmas)]

    for sigma_z, sigma_xy in product(z_list, xy_list):
        SearchLineChunked(args, sigma_z, sigma_xy, EPS)

if __name__ == "__main__":
    args = parse_args()
    main(args)