import numpy as np
from astropy.io import fits
from astropy.table import Table
import importlib

def open_cube(path):
	hdulist = fits.open(path,memmap=True)
	if len(np.shape(hdulist[0].data)) == 4:
		data  = hdulist[0].data[0]
	else:
		data  = hdulist[0].data
	return data

def GetMinSNEstimate(CubePath):
	'''
	Function that tried to get a rough estimate to the MinSN to use for the search. 
	This is very difficul to only use it as a rough reference. It is known to fail and give
	totally worng estimate in many cases.
	'''
	hdulist =   fits.open(CubePath,memmap=True)
	head = hdulist[0].header
	data = hdulist[0].data[0]

	try:
		BMAJ = hdulist[1].data.field('BMAJ')
		BMIN = hdulist[1].data.field('BMIN')
		BPA = hdulist[1].data.field('BPA')

	except:
		BMAJ = []
		BMIN = []
		BPA = []
		for i in range(len(data)):
			BMAJ.append(head['BMAJ']*3600.0)
			BMIN.append(head['BMIN']*3600.0)
			BPA.append(head['BPA'])
		BMAJ = np.array(BMAJ)
		BMIN = np.array(BMIN)
		BPA = np.array(BPA)

	pix_size = head['CDELT2']*3600.0
	factor = 2.0*(np.pi*BMAJ*BMIN/(8.0*np.log(2)))/(pix_size**2)
	RefFrequency = head['CRVAL3']
	ChannelSpacing = head['CDELT3']
	ApproxChannelVelocityWidth = (abs(ChannelSpacing)/RefFrequency)*3e5
	ApproxMaxSigmas = 1000.0/ApproxChannelVelocityWidth
	aux = len(data[0][np.isfinite(data[0])].flatten())*1.0/factor[0]*(len(data)/ApproxMaxSigmas)
	Number2Print = round(np.power(10,np.log10(aux)*0.07723905 + 0.19291493),1)
	print('*** A rough guesstimate to use as MinSN is',Number2Print,'***')
	# print len(data[0][np.isfinite(data[0])].flatten())*1.0/factor[0]*(len(data)/ApproxMaxSigmas)
	return 

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
	data = hdulist[0].data[0]

	try:
		BMAJ = hdulist[1].data.field('BMAJ')
		BMIN = hdulist[1].data.field('BMIN')
		BPA = hdulist[1].data.field('BPA')
	except:
		BMAJ = []
		BMIN = []
		BPA = []
		for i in range(len(data)):
			BMAJ.append(head['BMAJ']*3600.0)
			BMIN.append(head['BMIN']*3600.0)
			BPA.append(head['BPA'])
		BMAJ = np.array(BMAJ)
		BMIN = np.array(BMIN)
		BPA = np.array(BPA)
	pix_size = head['CDELT2']*3600.0
	return max(BMAJ/pix_size)


def get_mask(ContinuumImage, MaskSN):
	DataMask = fits.open(ContinuumImage,memmap=True)[0].data[0][0] 
	InitialRMS = np.nanstd(DataMask)
	FinalRMS = np.nanstd(DataMask[DataMask<MaskSN*InitialRMS])
	Mask = np.where(DataMask>=MaskSN*FinalRMS,True,False)
	return Mask

def save_positives(data, MinSN, FolderForLinesFiles, sigmas, ss_sigma):
	pix1,pix2,pix3 = np.where(data>=MinSN)
	t = Table([pix1, pix3, pix2,data[pix1,pix2,pix3]], names=('Channel', 'Xpix', 'Ypix','SN'))
	t.write(FolderForLinesFiles+'/line_dandidates_sn_sigmas'+str(sigmas)+"_"+str(ss_sigma)+'_pos.fits', format='fits',overwrite=True)
	print('Positive pixels in search for Sigmas:',sigmas,'N:',len(pix2))

def save_negatives(data, MinSN, FolderForLinesFiles, sigmas, ss_sigma):
	pix1,pix2,pix3 = np.where(data<=(-1.0*MinSN))
	t = Table([pix1, pix3, pix2,-1.0*data[pix1,pix2,pix3]], names=('Channel', 'Xpix', 'Ypix','SN'))
	t.write(FolderForLinesFiles+'/line_dandidates_sn_sigmas'+str(sigmas)+"_"+str(ss_sigma)+'_neg.fits', format='fits',overwrite=True)
	print('Negative pixels in search for Sigmas:',sigmas,'N:',len(pix2))