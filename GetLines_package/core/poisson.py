import numpy as np
import scipy.ndimage
from scipy.optimize import curve_fit


def get_poisson_estimates(bins,SNFinalPos,SNFinalNeg,LimitN,MinSN):

	ProbPoisson = []
	ProbPoissonE1 = []
	ProbPoissonE2 = []
	ProbNegativeOverPositive = []
	ProbNegativeOverPositiveE1 = []
	ProbNegativeOverPositiveE2 = []
	ProbPoissonExpected = []
	ProbPoissonExpectedE1 = []
	ProbPoissonExpectedE2 = []
	ProbNegativeOverPositiveDif = []
	ProbNegativeOverPositiveDifE1 = []
	ProbNegativeOverPositiveDifE2 = []	
	PurityPoisson = []
	Nnegative = []
	NnegativeReal = []
	NPositive = []
	Nnegative_e1 = []
	Nnegative_e2 = []

	for sn in bins:
		if len(SNFinalPos[SNFinalPos>=sn])>0:
			Fraction,FractionE1,FractionE2 = GetPoissonErrorGivenMeasurements(len(SNFinalNeg[SNFinalNeg>=sn]),len(SNFinalPos[SNFinalPos>=sn]))

			if Fraction>1.0:
				Fraction = 1.0
				FractionE1 = 0.0
				FractionE2 = 0.0
			else:
				pass

			ProbNegativeOverPositive.append(Fraction)
			ProbNegativeOverPositiveE1.append(FractionE1)
			ProbNegativeOverPositiveE2.append(FractionE2)
		elif len(SNFinalNeg[SNFinalNeg>=sn])>0:
			ProbNegativeOverPositive.append(1.0)
			ProbNegativeOverPositiveE1.append(0.0)
			ProbNegativeOverPositiveE2.append(0.0)
		else:
			ProbNegativeOverPositive.append(0.0)
			ProbNegativeOverPositiveE1.append(0.0)
			ProbNegativeOverPositiveE2.append(0.0)

		if len(SNFinalPos[(SNFinalPos>=sn) & (SNFinalPos<sn+0.1)])>0:
			Fraction,FractionE1,FractionE2 = GetPoissonErrorGivenMeasurements(len(SNFinalNeg[(SNFinalNeg>=sn) & (SNFinalNeg<sn+0.1)]),len(SNFinalPos[(SNFinalPos>=sn) & (SNFinalPos<sn+0.1)]))
			if Fraction>1.0:
				Fraction = 1.0
				FractionE1 = 0.0
				FractionE2 = 0.0
			else:
				pass

			ProbNegativeOverPositiveDif.append(min(1.0,Fraction))
			ProbNegativeOverPositiveDifE1.append(FractionE1)
			ProbNegativeOverPositiveDifE2.append(FractionE2)
		elif len(SNFinalNeg[(SNFinalNeg>=sn) & (SNFinalNeg<sn+0.1)])>0:
			ProbNegativeOverPositiveDif.append(1.0)
			ProbNegativeOverPositiveDifE1.append(0.0)
			ProbNegativeOverPositiveDifE2.append(0.0)
		else:
			ProbNegativeOverPositiveDif.append(0.0)
			ProbNegativeOverPositiveDifE1.append(0.0)
			ProbNegativeOverPositiveDifE2.append(0.0)

		k = len(SNFinalNeg[SNFinalNeg>=sn])
		aux = scipy.special.gammaincinv(k + 1, [0.16,0.5,0.84])
		NnegativeReal.append(k)
		Nnegative.append(aux[1])
		Nnegative_e1.append(aux[1]-aux[0])
		Nnegative_e2.append(aux[2]-aux[1])
		NPositive.append(1.0*len(SNFinalPos[SNFinalPos>=sn]))

	Nnegative = np.array(Nnegative)
	NPositive = np.array(NPositive)
	NnegativeReal = np.array(NnegativeReal)
	Nnegative_e1 = np.array(Nnegative_e1)
	Nnegative_e2 = np.array(Nnegative_e2)
	

	MinSNtoFit = min(bins)
	UsableBins = len(Nnegative[bins>=MinSNtoFit][Nnegative[bins>=MinSNtoFit]>LimitN])

	AuxiliarOutput = open('SN_UsedInFit.dat','w')
	print('Min SN to do the fit:',round(MinSNtoFit,1),', Number of usable bins:',UsableBins)
	AuxiliarOutput.write(str(round(MinSNtoFit,1))+' ' + str(UsableBins)+'\n')
	if UsableBins<6:
		print('*** We are using ',UsableBins,' points for the fitting of the negative counts ***')
		print('*** We usually get good results with 6 points, try reducing the parameter -MinSN ***')
	while UsableBins>6:
		MinSNtoFit = MinSNtoFit + 0.1
		UsableBins = len(Nnegative[bins>=MinSNtoFit][Nnegative[bins>=MinSNtoFit]>LimitN])
		print('Min SN to do the fit:',round(MinSNtoFit,1),', Number of usable bins:',UsableBins)
		AuxiliarOutput.write(str(round(MinSNtoFit,1))+' ' + str(UsableBins)+'\n')

		if MinSNtoFit>max(bins):
			print('No negative points to do the fit')
			exit()
	AuxiliarOutput.close()

	if UsableBins>=3:
		try:
			# popt, pcov = curve_fit(NegativeRate, bins[bins>=MinSNtoFit][Nnegative[bins>=MinSNtoFit]>LimitN], Nnegative[bins>=MinSNtoFit][Nnegative[bins>=MinSNtoFit]>LimitN],p0=[1e6,1])
			popt, pcov = curve_fit(NegativeRateLog, 
					bins[bins>=MinSNtoFit][Nnegative[bins>=MinSNtoFit]>LimitN],
					np.log10(Nnegative[bins>=MinSNtoFit][Nnegative[bins>=MinSNtoFit]>LimitN]),
					p0=[1e6,1],
					sigma=np.log10(np.average([Nnegative_e1[bins>=MinSNtoFit][Nnegative[bins>=MinSNtoFit]>LimitN],Nnegative_e2[bins>=MinSNtoFit][Nnegative[bins>=MinSNtoFit]>LimitN]],axis=0)),
					absolute_sigma=False)

			perr = np.sqrt(np.diag(pcov))
			# print popt,popt/perr,not np.isfinite(perr[0])
			CounterFitTries = 0
			while not np.isfinite(perr[0]):
				print('*** curve_fit failed to converge ... ***')
				NewParameter1 = np.power(10,np.random.uniform(1,9))
				NewParameter2 = np.random.uniform(0.1,2.0)
				print('*** New Initial Estimates for the fitting (random):',round(NewParameter1),round(NewParameter2,2),' ***')
				# popt, pcov = curve_fit(NegativeRate, bins[bins>=MinSNtoFit][Nnegative[bins>=MinSNtoFit]>LimitN], Nnegative[bins>=MinSNtoFit][Nnegative[bins>=MinSNtoFit]>LimitN],p0=[NewParameter1,NewParameter2])
				popt, pcov = curve_fit(NegativeRateLog, 
										bins[bins>=MinSNtoFit][Nnegative[bins>=MinSNtoFit]>LimitN],
										np.log10(Nnegative[bins>=MinSNtoFit][Nnegative[bins>=MinSNtoFit]>LimitN]),
										p0=[NewParameter1,NewParameter2],
										sigma=np.log10(np.average([Nnegative_e1[bins>=MinSNtoFit][Nnegative[bins>=MinSNtoFit]>LimitN],Nnegative_e2[bins>=MinSNtoFit][Nnegative[bins>=MinSNtoFit]>LimitN]],axis=0)),
										absolute_sigma=False)
				perr = np.sqrt(np.diag(pcov))
				print('*** New Results: N:',round(popt[0]),' +/- ',round(perr[0]),' Sigma:',round(popt[1],2),' +/- ',round(perr[1],2),' ***')
				CounterFitTries += 1
				if CounterFitTries >100:
					print('*** Over 100 attemps and no good fit *** ')
					break

		except:
			print('Fitting failed for LimitN:'+str(LimitN)+' and '+str(MinSN)+'... Will force LimitN=0')
			# popt, pcov = curve_fit(NegativeRate, bins[Nnegative>0], Nnegative[Nnegative>0],p0=[1e6,1])	
			popt, pcov = curve_fit(NegativeRateLog, 
					bins[Nnegative>0],
					np.log10(Nnegative[Nnegative>0]),
					p0=[1e6,1],
					sigma=np.log10(np.average([Nnegative_e1[Nnegative>0],Nnegative_e2[Nnegative>0]],axis=0)),
					absolute_sigma=False)
			perr = np.sqrt(np.diag(pcov))
			# print popt,popt/perr,not np.isfinite(perr[0])
			CounterFitTries = 0
			while not np.isfinite(perr[0]):
				print('*** curve_fit failed to converge ... ***')
				NewParameter1 = np.power(10,np.random.uniform(1,9))
				NewParameter2 = np.random.uniform(0.1,2.0)
				print('*** New Initial Estimates for the fitting (random):',round(NewParameter1),round(NewParameter2,2),' ***')
				# popt, pcov = curve_fit(NegativeRate, bins[Nnegative>0], Nnegative[Nnegative>0],p0=[NewParameter1,NewParameter2])
				popt, pcov = curve_fit(NegativeRateLog, 
										bins[Nnegative>0],
										np.log10(Nnegative[Nnegative>0]),
										p0=[NewParameter1,NewParameter2],
										sigma=np.log10(np.average([Nnegative_e1[Nnegative>0],Nnegative_e2[Nnegative>0]],axis=0)),
										absolute_sigma=False)
				perr = np.sqrt(np.diag(pcov))
				print('*** New Results: N:',round(popt[0]),' +/- ',round(perr[0]),' Sigma:',round(popt[1],2),' +/- ',round(perr[1],2),' ***')
				CounterFitTries += 1
				if CounterFitTries >100:
					print('*** Over 100 attemps and no good fit *** ')
					break
	else:
		print('Number of usable bins is less than 3 for LimitN:'+str(LimitN)+' and '+str(MinSN)+'... Will force LimitN=0')
		# popt, pcov = curve_fit(NegativeRate, bins[Nnegative>0], Nnegative[Nnegative>0],p0=[1e6,1])	
		popt, pcov = curve_fit(NegativeRateLog, 
					bins[Nnegative>0],
					np.log10(Nnegative[Nnegative>0]),
					p0=[1e6,1],
					sigma=np.log10(np.average([Nnegative_e1[Nnegative>0],Nnegative_e2[Nnegative>0]],axis=0)),
					absolute_sigma=False)
		perr = np.sqrt(np.diag(pcov))
		# print popt,popt/perr,not np.isfinite(perr[0])
		CounterFitTries = 0
		while not np.isfinite(perr[0]):
			print('*** curve_fit failed to converge ... ***')
			NewParameter1 = np.power(10,np.random.uniform(1,9))
			NewParameter2 = np.random.uniform(0.1,2.0)
			print('*** New Initial Estimates for the fitting (random):',round(NewParameter1),round(NewParameter2,2),' ***')
			# popt, pcov = curve_fit(NegativeRate, bins[Nnegative>0], Nnegative[Nnegative>0],p0=[NewParameter1,NewParameter2])
			popt, pcov = curve_fit(NegativeRateLog, 
										bins[Nnegative>0],
										np.log10(Nnegative[Nnegative>0]),
										p0=[NewParameter1,NewParameter2],
										sigma=np.log10(np.average([Nnegative_e1[Nnegative>0],Nnegative_e2[Nnegative>0]],axis=0)),
										absolute_sigma=False)
			perr = np.sqrt(np.diag(pcov))
			print('*** New Results: N:',round(popt[0]),' +/- ',round(perr[0]),' Sigma:',round(popt[1],2),' +/- ',round(perr[1],2),' ***')
			CounterFitTries += 1
			if CounterFitTries >100:
				print('*** Over 100 attemps and no good fit *** ')
				break

	NegativeFitted = NegativeRate(bins,popt[0],popt[1])
	SNPeakGaussian = (popt/np.sqrt(np.diag(pcov)))[0]
	# print 'SNPeakGaussian',SNPeakGaussian,popt,np.sqrt(np.diag(pcov))
	# print curve_fit(NegativeRate, bins[bins>=MinSNtoFit][Nnegative[bins>=MinSNtoFit]>LimitN], Nnegative[bins>=MinSNtoFit][Nnegative[bins>=MinSNtoFit]>LimitN],p0=[1e6,1],sigma=np.average([Nnegative_e1[bins>=MinSNtoFit][Nnegative[bins>=MinSNtoFit]>LimitN],Nnegative_e2[bins>=MinSNtoFit][Nnegative[bins>=MinSNtoFit]>LimitN]],axis=0),absolute_sigma=False)
	# print curve_fit(NegativeRateLog, 
	# 				bins[bins>=MinSNtoFit][Nnegative[bins>=MinSNtoFit]>LimitN],
	# 				np.log10(Nnegative[bins>=MinSNtoFit][Nnegative[bins>=MinSNtoFit]>LimitN]),
	# 				p0=[1e6,1],
	# 				sigma=np.log10(np.average([Nnegative_e1[bins>=MinSNtoFit][Nnegative[bins>=MinSNtoFit]>LimitN],Nnegative_e2[bins>=MinSNtoFit][Nnegative[bins>=MinSNtoFit]>LimitN]],axis=0)),
	# 				absolute_sigma=False)

	for i in range(len(bins)):
		aux = []
		auxExpected = []
		for j in range(1000):
			lamb = np.random.normal(NegativeFitted[i],NegativeFitted[i]/SNPeakGaussian)
			while lamb<0:
				lamb = np.random.normal(NegativeFitted[i],NegativeFitted[i]/SNPeakGaussian)
			aux.append(1-scipy.special.gammaincc(0+1,lamb))
			if i ==len(bins)-1:
				if NPositive[i]>0:
					auxExpected.append(1.0-max(0,NPositive[i]-lamb)/NPositive[i])
				else:
					auxExpected.append(0.0)
			else:
				# lamb2 = lamb - np.random.normal(NegativeFitted[i+1],NegativeFitted[i+1]/SNPeakGaussian) 
				lamb2 = (NegativeFitted[i] - NegativeFitted[i+1])*lamb/NegativeFitted[i]
				while lamb2<0:
					lamb2 = lamb - np.random.normal(NegativeFitted[i+1],NegativeFitted[i+1]/SNPeakGaussian) 
				if (NPositive[i] - NPositive[i+1])>0:
					auxExpected.append(1.0-max(0,(NPositive[i] - NPositive[i+1]) - lamb2)/(NPositive[i] - NPositive[i+1]))
				else:
					auxExpected.append(0.0)
					# auxExpected.append(1.0-max(0,0.7 - lamb2)/0.7)


		PP = np.nanpercentile(aux,[16,50,84])
		PPExpected = np.nanpercentile(auxExpected,[16,50,84])
		ProbPoisson.append(PP[1])
		ProbPoissonE1.append(PP[1]-PP[0])
		ProbPoissonE2.append(PP[2]-PP[1])

		ProbPoissonExpected.append(PPExpected[1])
		ProbPoissonExpectedE1.append(PPExpected[1]-PPExpected[0])
		ProbPoissonExpectedE2.append(PPExpected[2]-PPExpected[1])		
		# if i<len(bins)-1:
		# 	print bins[i],PPExpected,NegativeFitted[i],NPositive[i],NPositive[i+1]
		if NPositive[i]>0:
			PurityPoisson.append(max((NPositive[i]-NegativeFitted[i])/NPositive[i],0))
		else:
			PurityPoisson.append(0.0)

	ProbPoisson = np.array(ProbPoisson)
	ProbPoissonE1 = np.array(ProbPoissonE1)
	ProbPoissonE2 = np.array(ProbPoissonE2)
	ProbNegativeOverPositive = np.array(ProbNegativeOverPositive)
	ProbNegativeOverPositiveE1 = np.array(ProbNegativeOverPositiveE1)
	ProbNegativeOverPositiveE2 = np.array(ProbNegativeOverPositiveE2)
	ProbNegativeOverPositiveDif = np.array(ProbNegativeOverPositiveDif)
	ProbNegativeOverPositiveDifE1 = np.array(ProbNegativeOverPositiveDifE1)
	ProbNegativeOverPositiveDifE2 = np.array(ProbNegativeOverPositiveDifE2)
	ProbPoissonExpected = np.array(ProbPoissonExpected)
	ProbPoissonExpectedE1 = np.array(ProbPoissonExpectedE1)
	ProbPoissonExpectedE2 = np.array(ProbPoissonExpectedE2)
	PurityPoisson = np.array(PurityPoisson)

	output = {
		'bins': bins,
		'pPoisson': ProbPoisson,
		'pNegOverPos': ProbNegativeOverPositive,
		'purityPoisson': PurityPoisson,
		'nPositive': NPositive,
        "nNegative": Nnegative,
        "nNegE1": Nnegative_e1,
        "nNegE2": Nnegative_e2,
        "NegFitted": NegativeFitted,
        "nNegReal": NnegativeReal,
        "pPoissonE1": ProbPoissonE1,
        "pPoissonE2": ProbPoissonE2,
        "pNegOverPosE1": ProbNegativeOverPositiveE1,
        "pNegOverPosE2": ProbNegativeOverPositiveE2,
        "pNegOverPosDif": ProbNegativeOverPositiveDif,
        "pNegOverPosDifE1": ProbNegativeOverPositiveDifE1,
        "pNegOverPosDifE2": ProbNegativeOverPositiveDifE2,
        "pPoissonExpected": ProbPoissonExpected,
        "pPoissonExpectedE1": ProbPoissonExpectedE1,
        "pPoissonExpectedE2": ProbPoissonExpectedE2
		   }
	return output


def GetPoissonErrorGivenMeasurements(NMeasured,Total):
	k = NMeasured
	n = Total
	aux = scipy.special.betaincinv(k+1.0, n+1.0-k, [0.16,0.5,0.84])
	Estimate = 1.0*NMeasured/Total
	E1 = aux[1]-aux[0]
	E2 = aux[2]-aux[1]
	return Estimate,E1,E2


def NegativeRate(SNR,N,sigma):
	# return N*np.exp(-1.0*np.power(SNR,2)/(2.0*np.power(sigma,2)))
	return N*0.5 *( 1.0 -  scipy.special.erf(SNR/(np.sqrt(2.0)*sigma)))  #1 - CDF(SNR) assuming Gaussian distribution and N independent elements.

def NegativeRateLog(SNR,N,sigma):
	# return N*np.exp(-1.0*np.power(SNR,2)/(2.0*np.power(sigma,2)))
	return np.log10(N*0.5 *( 1.0 -  scipy.special.erf(SNR/(np.sqrt(2.0)*sigma))))  #1 - CDF(SNR) assuming Gaussian distribution and N independent elements.
