import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.table import Table
from scipy.special import gammaincinv
from sklearn.cluster import DBSCAN

try:
	import seaborn as sns
	sns.set_style("white", {'legend.frameon': True})
	sns.set_style("ticks", {'legend.frameon': True})
	sns.set_context("talk")
	sns.set_palette('Dark2', 8,desat=1)
	cc = sns.color_palette()
except:
	print('No seaborn package installed')
	cc = ['red','blue','green','orange','magenta','black']

from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})


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

def get_LEE_factor(args):
	if args.MaxSigmas > 1:
		FactorLEE = 1.0 + (np.log(args.MaxSigmas-1.0) + 1.0/(2.0*(args.MaxSigmas-1)) + 0.577)
	else:
		FactorLEE = 1.0
	return FactorLEE

def make_figure():
	w, h = 1.0*plt.figaspect(0.9)
	fig1 = plt.figure(figsize=(w,h))
	fig1.subplots_adjust(left=0.15, bottom=0.13, right=0.94, top=0.96,wspace=0.10, hspace=0.2)
	ax1 = fig1.add_subplot(111)
	return ax1

def get_binning(min_sigma):
	return np.arange(min_sigma,8.1,0.1)

def n_positives(NPositive):
	NPositive_e1 = []
	NPositive_e2 = []
	for k in NPositive:
		aux = gammaincinv(k + 1, [0.16,0.5,0.84])
		NPositive_e1.append(aux[1]-aux[0])
		NPositive_e2.append(aux[2]-aux[1])
	NPositive_e1 = np.array(NPositive_e1)
	NPositive_e2 = np.array(NPositive_e2)
	return(NPositive_e1, NPositive_e2)

def plot_N_positive_negative(estimates, NPositive_e1, NPositive_e2, sigma, spatial, args):
	w, h = 1.0*plt.figaspect(0.9)
	fig2 = plt.figure(figsize=(w,h))
	fig2.subplots_adjust(left=0.15, bottom=0.13, right=0.94, top=0.96,wspace=0.10, hspace=0.2)
	ax2 = fig2.add_subplot(111)

	bins = estimates['bins']
	NPositive = estimates['nPositive']
	NnegativeReal = estimates['nNegReal']
	Nnegative = estimates["nNegative"]
	Nnegative_e1, Nnegative_e2 =estimates["nNegE1"], estimates["nNegE2"]
	NegativeFitted = estimates["NegFitted"]

	# ax2.semilogy(bins,NPositive,'o',color=cc[0],label='Positive Detections')
	ax2.errorbar(bins[NPositive>0],NPositive[NPositive>0],yerr=[NPositive_e1[NPositive>0],NPositive_e2[NPositive>0]],fmt='o',color=cc[0],label='Positive Detections',zorder=0)
	ax2.errorbar(bins[NnegativeReal>0],Nnegative[NnegativeReal>0],yerr=[Nnegative_e1[NnegativeReal>0],Nnegative_e2[NnegativeReal>0]],fmt='o',color=cc[1],label='Negative Detections for sigmas:'+str(sigma)+"_"+str(spatial),zorder=0)
	ax2.semilogy(bins,NegativeFitted,'-',color=cc[2],label='Fitted negative underlying rate',zorder=1)
	ax2.set_xlabel('S/N',fontsize=20)
	ax2.set_ylabel('N (>S/N)',fontsize=20)

	if len(bins[NnegativeReal>0])>0:
		ax2.legend(loc=0,fontsize=args.LegendFontSize,ncol=1)

	ax2.tick_params(axis='both', which='major', labelsize=20)
	ax2.set_ylim(ymin=0.1)
	ax2.set_xticks(np.arange(int(args.MinSN),8,1))
	ax2.grid(True)
	fig2.savefig('NumberPositiveNegative_'+str(sigma)+"_"+str(spatial)+'.pdf')

def LS_final_candidates(SourcesTotalPos, PixelsPerBMAJ):
	COORD = []
	X = []
	Y = []
	Channel = []
	SN_array = []
	puritySimulation= []
	purityNegative = []
	purityPoisson = []
	psimulationE1 = []
	psimulationE2 = []
	ppoissonE1 = []
	ppoissonE2 = []	
	purityNegativeE1 = []
	purityNegativeE2 = []
	pNegDiv = []
	pNegDivE1 = []
	pNegDivE2 = []
	pSimExp = []
	pSimExpE1 = []
	pSimExpE2 = []
	pPoiExp = []
	pPoiExpE1 = []
	pPoiExpE2 = []

	for NewSource in SourcesTotalPos:
			COORD.append(np.array([NewSource[1],NewSource[2],NewSource[0]]))
			X.append(NewSource[1])
			Y.append(NewSource[2])
			Channel.append(NewSource[0])
			SN_array.append(NewSource[3])		
			puritySimulation.append(NewSource[5])	
			purityNegative.append(NewSource[6])	
			purityPoisson.append(NewSource[7])	
			psimulationE1.append(NewSource[8])	
			psimulationE2.append(NewSource[9])	
			ppoissonE1.append(NewSource[10])	
			ppoissonE2.append(NewSource[11])	
			purityNegativeE1.append(NewSource[12])
			purityNegativeE2.append(NewSource[13])
			pNegDiv.append(NewSource[14])
			pNegDivE1.append(NewSource[15])
			pNegDivE2.append(NewSource[16])
			pSimExp.append(NewSource[17])
			pSimExpE1.append(NewSource[18])
			pSimExpE2.append(NewSource[19])
			pPoiExp.append(NewSource[20])
			pPoiExpE1.append(NewSource[21])
			pPoiExpE2.append(NewSource[22])

	COORD = np.array(COORD)
	X = np.array(X)
	Y = np.array(Y)
	Channel = np.array(Channel)
	SN = np.array(SN_array)
	puritySimulation = np.array(puritySimulation)
	purityNegative = np.array(purityNegative)
	purityPoisson = np.array(purityPoisson)
	psimulationE1 = np.array(psimulationE1)
	psimulationE2 = np.array(psimulationE2)
	ppoissonE1 = np.array(ppoissonE1)
	ppoissonE2 = np.array(ppoissonE2)
	purityNegativeE1 = np.array(purityNegativeE1)
	purityNegativeE2 = np.array(purityNegativeE2)
	pNegDiv = np.array(pNegDiv)
	pNegDivE1 = np.array(pNegDivE1)
	pNegDivE2 = np.array(pNegDivE2)
	pSimExp = np.array(pSimExp)
	pSimExpE1 = np.array(pSimExpE1)
	pSimExpE2 = np.array(pSimExpE2)
	pPoiExp = np.array(pPoiExp)
	pPoiExpE1 = np.array(pPoiExpE1)
	pPoiExpE2 = np.array(pPoiExpE2)

	db = DBSCAN(eps=PixelsPerBMAJ, min_samples=1,leaf_size=30).fit(COORD)
	core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
	core_samples_mask[db.core_sample_indices_] = True
	labels = db.labels_
	unique_labels = set(labels)

	FinalX = []
	FinalY = []
	FinalChannel = []
	FinalSN = []
	FinalPuritySimulation = []
	FinalPurityNegative = []
	FinalPurityPoisson = []
	FinalPSimultionE1 = []
	FinalPSimultionE2 = []	
	FinalPPoissonE1 = []
	FinalPPoissonE2 = []
	FinalPuritySimulationE1 = []
	FinalPuritySimulationE2 = []
	FinalpNegDiv = []
	FinalpNegDivE1 = []
	FinalpNegDivE2 = []
	FinalpSimExp = []
	FinalpSimExpE1 = []
	FinalpSimExpE2 = []
	FinalpPoiExp = []
	FinalpPoiExpE1 = []
	FinalpPoiExpE2 = []

	for k in unique_labels:
		class_member_mask = (labels == k)
		FinalX.append(X[class_member_mask][np.argmax(SN[class_member_mask])])
		FinalY.append(Y[class_member_mask][np.argmax(SN[class_member_mask])])
		FinalChannel.append(Channel[class_member_mask][np.argmax(SN[class_member_mask])])
		FinalSN.append(max(SN[class_member_mask]))
		FinalPuritySimulation.append(min(puritySimulation[class_member_mask]))
		FinalPurityNegative.append(min(min(purityNegative[class_member_mask]),1))
		FinalPurityPoisson.append(min(purityPoisson[class_member_mask]))
		FinalPSimultionE1.append(psimulationE1[class_member_mask][np.argmin(puritySimulation[class_member_mask])])
		FinalPSimultionE2.append(psimulationE2[class_member_mask][np.argmin(puritySimulation[class_member_mask])])
		FinalPPoissonE1.append(ppoissonE1[class_member_mask][np.argmin(purityPoisson[class_member_mask])])
		FinalPPoissonE2.append(ppoissonE2[class_member_mask][np.argmin(purityPoisson[class_member_mask])])
		FinalPuritySimulationE1.append(purityNegativeE1[class_member_mask][np.argmin(purityNegative[class_member_mask])])
		FinalPuritySimulationE2.append(purityNegativeE2[class_member_mask][np.argmin(purityNegative[class_member_mask])])
		FinalpNegDiv.append(min(pNegDiv[class_member_mask]))
		FinalpNegDivE1.append(pNegDivE1[class_member_mask][np.argmin(pNegDiv[class_member_mask])])
		FinalpNegDivE2.append(pNegDivE2[class_member_mask][np.argmin(pNegDiv[class_member_mask])])
		FinalpSimExp.append(min(pSimExp[class_member_mask]))
		FinalpSimExpE1.append(pSimExpE1[class_member_mask][np.argmin(pSimExp[class_member_mask])])
		FinalpSimExpE2.append(pSimExpE2[class_member_mask][np.argmin(pSimExp[class_member_mask])])
		FinalpPoiExp.append(min(pPoiExp[class_member_mask]))
		FinalpPoiExpE1.append(pPoiExpE1[class_member_mask][np.argmin(pPoiExp[class_member_mask])])
		FinalpPoiExpE2.append(pPoiExpE2[class_member_mask][np.argmin(pPoiExp[class_member_mask])])	


	FinalX = np.array(FinalX)
	FinalY = np.array(FinalY)
	FinalChannel = np.array(FinalChannel)
	FinalSN = np.array(FinalSN)
	FinalPuritySimulation = np.array(FinalPuritySimulation)
	FinalPurityNegative = np.array(FinalPurityNegative)
	FinalPurityPoisson = np.array(FinalPurityPoisson)
	FinalPSimultionE1 = np.array(FinalPSimultionE1)
	FinalPSimultionE2 = np.array(FinalPSimultionE2)
	FinalPPoissonE1 = np.array(FinalPPoissonE1)
	FinalPPoissonE2 = np.array(FinalPPoissonE2)
	FinalPuritySimulationE1 = np.array(FinalPuritySimulationE1)
	FinalPuritySimulationE2 = np.array(FinalPuritySimulationE2)
	FinalpNegDiv = np.array(FinalpNegDiv)
	FinalpNegDivE1 = np.array(FinalpNegDivE1)
	FinalpNegDivE2 = np.array(FinalpNegDivE2)
	FinalpSimExp = np.array(FinalpSimExp)
	FinalpSimExpE1 = np.array(FinalpSimExpE1)
	FinalpSimExpE2 = np.array(FinalpSimExpE2)
	FinalpPoiExp = np.array(FinalpPoiExp)
	FinalpPoiExpE1 = np.array(FinalpPoiExpE1)
	FinalpPoiExpE2 = np.array(FinalpPoiExpE2)


	FinalX = FinalX[np.argsort(FinalSN)]
	FinalY = FinalY[np.argsort(FinalSN)]
	FinalChannel = FinalChannel[np.argsort(FinalSN)]
	FinalPuritySimulation = FinalPuritySimulation[np.argsort(FinalSN)]
	FinalPurityNegative = FinalPurityNegative[np.argsort(FinalSN)]
	FinalPurityPoisson = FinalPurityPoisson[np.argsort(FinalSN)]
	FinalPSimultionE1 = FinalPSimultionE1[np.argsort(FinalSN)]
	FinalPSimultionE2 = FinalPSimultionE2[np.argsort(FinalSN)]
	FinalPPoissonE1 = FinalPPoissonE1[np.argsort(FinalSN)]
	FinalPPoissonE2 = FinalPPoissonE2[np.argsort(FinalSN)]
	FinalPuritySimulationE1 = np.array(FinalPuritySimulationE1)[np.argsort(FinalSN)]
	FinalPuritySimulationE2 = np.array(FinalPuritySimulationE2)[np.argsort(FinalSN)]
	FinalpNegDiv = np.array(FinalpNegDiv)[np.argsort(FinalSN)]
	FinalpNegDivE1 = np.array(FinalpNegDivE1)[np.argsort(FinalSN)]
	FinalpNegDivE2 = np.array(FinalpNegDivE2)[np.argsort(FinalSN)]
	FinalpSimExp = np.array(FinalpSimExp)[np.argsort(FinalSN)]
	FinalpSimExpE1 = np.array(FinalpSimExpE1)[np.argsort(FinalSN)]
	FinalpSimExpE2 = np.array(FinalpSimExpE2)[np.argsort(FinalSN)]
	FinalpPoiExp = np.array(FinalpPoiExp)[np.argsort(FinalSN)]
	FinalpPoiExpE1 = np.array(FinalpPoiExpE1)[np.argsort(FinalSN)]
	FinalpPoiExpE2 = np.array(FinalpPoiExpE2)[np.argsort(FinalSN)]
	FinalSN = FinalSN[np.argsort(FinalSN)]

	output = {
				"fX" : FinalX,
				"fY" : FinalY,
				"fChannel" : FinalChannel,
				"fPuritySim" : FinalPuritySimulation,
				"fPurityNeg" : FinalPurityNegative,
				"fPurityPoisson" : FinalPurityPoisson,
				"fSN" : FinalSN,
				"fPSimE1" : FinalPSimultionE1,
				"fPSimE2" : FinalPSimultionE2,
				"fPPoissonE1" : FinalPPoissonE1,
				"fPPoissonE2" : FinalPPoissonE2,
				"fPuritySimE1" : FinalPuritySimulationE1,
				"fPuritySimE2" : FinalPuritySimulationE2,
				"fpNegDiv" : FinalpNegDiv,
				"fpNegDivE1" : FinalpNegDivE1,
				"fpNegDivE2" : FinalpNegDivE2,
				"fpSimExp" : FinalpSimExp,
				"fpSimExpE1" : FinalpSimExpE1,
				"fpSimExpE2" : FinalpSimExpE2,
				"fpPoiExp" : FinalpPoiExp,
				"fpPoiExpE1" : FinalpPoiExpE1,
				"fpPoiExpE2" : FinalpPoiExpE2
				}
	return output







def open_cube(path):
	hdulist = fits.open(path,memmap=True)
	if len(np.shape(hdulist[0].data)) == 4:
		data  = hdulist[0].data[0]
	else:
		data  = hdulist[0].data
	return data

def save_positives(data, MinSN, FolderForLinesFiles, sigmas):
	pix1,pix2,pix3 = np.where(data>=MinSN)
	t = Table([pix1, pix3, pix2,data[pix1,pix2,pix3]], names=('Channel', 'Xpix', 'Ypix','SN'))
	t.write(FolderForLinesFiles+'/line_dandidates_sn_sigmas'+str(sigmas)+'_pos.fits', format='fits',overwrite=True)
	print('Positive pixels in search for Sigmas:',sigmas,'N:',len(pix2))

def save_negatives(data, MinSN, FolderForLinesFiles, sigmas):
	pix1,pix2,pix3 = np.where((data)<=(-1.0*MinSN))
	t = Table([pix1, pix3, pix2,-1.0*data[pix1,pix2,pix3]], names=('Channel', 'Xpix', 'Ypix','SN'))
	t.write(FolderForLinesFiles+'/line_dandidates_sn_sigmas'+str(sigmas)+'_neg.fits', format='fits',overwrite=True)
	print('Negative pixels in search for Sigmas:',sigmas,'N:',len(pix2))


