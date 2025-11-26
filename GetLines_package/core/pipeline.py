from abc import ABC, abstractmethod
from ..core.functions import get_binning, LS_final_candidates
from ..core.poisson import GetPoissonErrorGivenMeasurements
from numpy.typing import NDArray
from astropy.io import fits
from sklearn.cluster import DBSCAN
from astropy import wcs
from astropy.coordinates import SkyCoord
import astropy.io.fits as fits
import numpy as np


class PipeLine(ABC):
    def __init__(self, args):
        super().__init__()
        self.args = args

    @abstractmethod
    def clustering(self, eps, COORD, min_samples = 1, leaf_size= 20):
        """
        Applies a Density-Based Spatial Clustering of Applications with Noise 
        as clustering method to identify ...
        
        Parameters:
        - param: text

        Return:
        - object desc
        """
        pass

class RealSources(ABC): #### Agrgar el espacial...
    def __init__(self, args):
        self.args = args
        self.real_sources = []
        self.SN = []
        self.type: str = ''
        self.type_LineCandidate: str = ''
        self.y = [0]
        self.yExpected = [0]
        self.yExpectedSigma = []
        self.N_simulations2 = 0.0
        self.SourcesTotalPos = []
        self.FinalSN = []
        self.SigmaFinalSN = []
        self.sigma_per_source = []
        self.FinalSigma = []
        self.spatial_per_source = []
        self.FinalSpatial = []

    def get_sources_file(self, sigma:int|float, spatial, ppBMAJ) -> NDArray[np.float64] :
        files = [self.args.LineSearchPath+'/line_dandidates_sn_sigmas'+str(sigma)+'_'+str(spatial)+'_'+self.type]
        files.sort()
        sources, sources_aux = [], []
        for f in files:
            coord, x, y, channels, SN = self.get_arrays_file(f, sources, self.args.MinSN)
            if len(coord)>0:
                db = DBSCAN(eps=ppBMAJ, min_samples=1,leaf_size=30).fit(coord)
                core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
                core_samples_mask[db.core_sample_indices_] = True
                labels = db.labels_
                unique_labels = set(labels)
                sources_aux = []

                #print(unique_labels)
                for k in unique_labels:
                    class_member_mask = (labels == k)
                    source = [ channels[class_member_mask],
                              x[class_member_mask], y[class_member_mask],
                              SN[class_member_mask], 
                              max(channels[class_member_mask]) - min(channels[class_member_mask])
                              ]
                    sources_aux.append(source)

            else:
                sources_aux = []

            for source in sources_aux:
                sources.append(source)
        self.real_sources = sources #### real_sources
    
    def get_arrays_file(self, path, Sources, minSN):
        path = path.replace('.fits','').replace('.dat','')

        try:
            table = fits.open(path+'.fits')[1].data
            SN = table['SN']
            X = table['Xpix'][SN>=minSN]
            Y = table['Ypix'][SN>=minSN]
            Channel = table['Channel'][SN>=minSN]
            SN_array = table['SN'][SN>=minSN]
            COORD = np.transpose(np.array([X,Y,Channel]))
            COORD = list(COORD)

            for source in Sources:
                    COORD.append(np.array([source[1][np.argmax(source[3])],source[2][np.argmax(source[3])],source[0][np.argmax(source[3])]]))
                    X = np.append(X,source[1][np.argmax(source[3])])
                    Y = np.append(Y,source[2][np.argmax(source[3])])
                    Channel = np.append(Channel,source[0][np.argmax(source[3])])
                    SN_array = np.append(SN_array,source[3][np.argmax(source[3])])
            
            COORD = np.array(COORD)
            X = np.array(X)
            Y = np.array(Y)
            Channel = np.array(Channel)
            SN = np.array(SN_array)
            return COORD, X, Y, Channel, SN
        except:
            COORD = []
            X = []
            Y = []
            Channel = []
            SN_array = []
            FileReader = open(path+'.dat').readlines()

            for j in FileReader:
                FirstCharacter = j[0]
                j = j.split()
                if FirstCharacter == ' ':
                    continue
                if FirstCharacter == '-' or j[0]== 'max_negative_sn:':
                    continue
                SN = np.float(j[3].replace('SN:',''))

                if SN>=minSN :
                    spw = int(j[0])
                    x = np.float(j[1])
                    y = np.float(j[2])
                    sn = np.float(j[-1].replace('SN:',''))
                    COORD.append(np.array([x,y,spw]))
                    X.append(x)
                    Y.append(y)
                    Channel.append(spw)
                    SN_array.append(sn)

            for source in Sources:
                    COORD.append(np.array([source[1][np.argmax(source[3])],source[2][np.argmax(source[3])],source[0][np.argmax(source[3])]]))
                    X = np.append(X,source[1][np.argmax(source[3])])
                    Y = np.append(Y,source[2][np.argmax(source[3])])
                    Channel = np.append(Channel,source[0][np.argmax(source[3])])
                    SN_array = np.append(SN_array,source[3][np.argmax(source[3])])

            COORD = np.array(COORD)
            X = np.array(X)
            Y = np.array(Y)
            Channel = np.array(Channel)
            SN = np.array(SN_array)
            return COORD,X,Y,Channel,SN
    
    def real_SN(self):
        aux = []
        for source in self.real_sources:
            aux.append(max(source[3]))
        self.SN = np.array(aux)

    def get_ys(self, bins):
        self.y = np.zeros_like(bins)
        self.yExpected = np.zeros_like(bins)
        self.N_simulations2 = 0.0
        self.yExpectedSigma = np.array(self.yExpectedSigma)

    def plot_y(self, sigma, bins, axis):
        if sigma<7:
            axis.plot(bins, self.y, '-', label=r' $\sigma$ = '+str(sigma)+' channels')
        if sigma>=7 and sigma<14:
            axis.plot(bins, self.y, '--' ,label=r' $\sigma$ = '+str(sigma)+' channels')
        if sigma>=14:
            axis.plot(bins, self.y, ':', label=r' $\sigma$ = '+str(sigma)+' channels')

    def fix_SN(self):
        self.real_sources = np.array(self.real_sources, dtype='object')
        self.real_sources = self.real_sources[np.argsort(self.SN)][::-1]
        self.SN = self.SN[np.argsort(self.SN)][::-1]
        self.SN = 1.0*np.round(self.SN, 1)
        self.SN = self.SN.astype(np.float32)

    def get_sources_total_pos(self, estimates, sigma, spatial):
        bins = estimates["bins"]
        SNReal = self.SN
        for source in self.real_sources:
            if max(source[3]) >= self.args.MinSN:
                sn = round(max(source[3]),1)
                if self.N_simulations2 > 0:
                    aux,ErrorPSimulation_1,ErrorPSimulation_2 = GetPoissonErrorGivenMeasurements(np.interp(sn,bins, self.y)*self.N_simulations2,self.N_simulations2)
                else:
                    ErrorPSimulation_1 = 0.0
                    ErrorPSimulation_2 = 0.0

                index = np.argmin(abs(estimates["bins"] - sn))
                ErrorPPoisson_1 = estimates["pPoissonE1"][index]
                ErrorPPoisson_2 = estimates["pPoissonE2"][index]

                PNegativeTotalE1Real = np.interp(sn,bins,estimates["pNegOverPosE1"])
                PNegativeTotalE2Real = np.interp(sn,bins,estimates["pNegOverPosE2"])
                PNegativeDifTotalReal = np.interp(sn,bins,estimates["pNegOverPosDif"])
                PNegativeDifTotalE1Real = np.interp(sn,bins,estimates["pNegOverPosDifE1"])
                PNegativeDifTotalE2Real = np.interp(sn,bins,estimates["pNegOverPosDifE2"])

                if self.N_simulations2>0 and (len(SNReal[SNReal>=sn]) - len(SNReal[SNReal>=sn+0.1]))>0:
                    Rate = np.interp(sn,bins,self.yExpected) - np.interp(sn+0.1,bins,self.yExpected)
                    RateSigma = 0.5*np.sqrt(np.sum(np.power(np.array([np.interp(sn,bins,self.yExpectedSigma),np.interp(sn+0.1,bins,self.yExpectedSigma)]),2)))
                    auxSimulationExpected = [Rate - RateSigma,Rate,Rate + RateSigma]
                    Psimexpected = 1.0 - max(0,len(SNReal[SNReal>=sn]) - len(SNReal[SNReal>=sn+0.1]) - auxSimulationExpected[1])*1.0/(len(SNReal[SNReal>=sn]) - len(SNReal[SNReal>=sn+0.1]))
                    Psimexpected1 = 1.0 - max(0,len(SNReal[SNReal>=sn]) - len(SNReal[SNReal>=sn+0.1]) - auxSimulationExpected[0])*1.0/(len(SNReal[SNReal>=sn]) - len(SNReal[SNReal>=sn+0.1]))
                    Psimexpected2 = 1.0 - max(0,len(SNReal[SNReal>=sn]) - len(SNReal[SNReal>=sn+0.1]) - auxSimulationExpected[2])*1.0/(len(SNReal[SNReal>=sn]) - len(SNReal[SNReal>=sn+0.1]))
                else:
                    Psimexpected = 0.0
                    Psimexpected1 = 0.0
                    Psimexpected2 = 0.0

                PSimulationExpectedTotalReal = Psimexpected
                PSimulationExpectedTotalE1Real = Psimexpected - Psimexpected1
                PSimulationExpectedTotalE2Real = Psimexpected2 - Psimexpected
                # PPoissonExpectedTotalReal = np.interp(sn,bins,ProbPoissonExpected)
                PPoissonExpectedTotalReal = estimates["pPoissonExpected"][index]
                PPoissonExpectedTotalE1Real = estimates["pPoissonExpectedE1"][index]
                PPoissonExpectedTotalE2Real = estimates["pPoissonExpectedE2"][index]

                NewSource = [
                                source[0][np.argmax(source[3])],
                                source[1][np.argmax(source[3])],
                                source[2][np.argmax(source[3])],
                                source[3][np.argmax(source[3])],
                                source[4],
                                np.interp(source[3][np.argmax(source[3])],bins,self.y),
                                np.interp(source[3][np.argmax(source[3])],bins,estimates["pNegOverPos"]),
                                np.interp(source[3][np.argmax(source[3])],bins,estimates["pPoisson"]),ErrorPSimulation_1,
                                ErrorPSimulation_2,
                                ErrorPPoisson_1,
                                ErrorPPoisson_2,
                                PNegativeTotalE1Real,
                                PNegativeTotalE2Real,
                                PNegativeDifTotalReal,
                                PNegativeDifTotalE1Real,
                                PNegativeDifTotalE2Real,
                                PSimulationExpectedTotalReal,
                                PSimulationExpectedTotalE1Real,
                                PSimulationExpectedTotalE2Real,
                                PPoissonExpectedTotalReal,
                                PPoissonExpectedTotalE1Real,
                                PPoissonExpectedTotalE2Real
                            ]
                self.SourcesTotalPos.append(NewSource)
                self.sigma_per_source.append(sigma)
                self.spatial_per_source.append(spatial)

    def clustering(self, pipeline, pp_BMAJ):
        COORD, X, Y, Channel, SN_array, purity = [], [], [], [], [], []
        sigma_array = []
        spatial_array = []
        for NewSource, sigma, spatial in zip(self.SourcesTotalPos, self.sigma_per_source, self.spatial_per_source):
                COORD.append(np.array([NewSource[1],NewSource[2],NewSource[0]]))
                X.append(NewSource[1])
                Y.append(NewSource[2])
                Channel.append(NewSource[0])
                SN_array.append(NewSource[3])		
                purity.append(NewSource[5])
                sigma_array.append(sigma)
                spatial_array.append(spatial)

        COORD = np.array(COORD)
        X = np.array(X)
        Y = np.array(Y)
        Channel = np.array(Channel)
        SN = np.array(SN_array)
        purity = np.array(purity)
        sigma_array = np.array(sigma_array)
        spatial_array = np.array(spatial_array)

        db = pipeline.clustering(eps=pp_BMAJ, COORD= COORD, min_samples=1, leaf_size=30)

        core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
        core_samples_mask[db.core_sample_indices_] = True
        labels = db.labels_
        unique_labels = set(labels)

        self.FinalSigma = []
        self.FinalSpatial = []
        FinalSN = []
        for k in unique_labels:
            class_member_mask = (labels == k)
            best = np.argmax(SN[class_member_mask])
            FinalSN.append(SN[class_member_mask][best]) ##
            self.FinalSigma.append(sigma_array[class_member_mask][best]) ##
            self.FinalSpatial.append(spatial_array[class_member_mask][best])

        
        FinalSN = np.array(FinalSN)
        self.FinalSigma = np.array(self.FinalSigma)
        self.FinalSpatial = np.array(self.FinalSpatial)
        
        order = np.argsort(FinalSN) ##
        self.FinalSN = FinalSN[order] ##
        self.FinalSigma = self.FinalSigma[order] ##
        self.FinalSpatial = self.FinalSpatial[order]

    def get_final_candidates(self, pp_BMAJ, FactorLEE, to_csv= True):
        if type(self.FinalSN) != list:
            print(len(self.FinalSN))
        candidates = LS_final_candidates(self.SourcesTotalPos, pp_BMAJ)
        FinalpSimExpOriginal = candidates["fpSimExp"] * 1 #############
        FinalpSimExpE1Original = candidates["fpSimExpE1"] * 1 ###########
        FinalpSimExpE2Original = candidates["fpSimExpE2"] * 1  ################
        FinalpPoiExpOriginal = candidates["fpPoiExp"] * 1  ################
        FinalpPoiExpE1Original = candidates["fpPoiExpE1"] * 1  ################
        FinalpPoiExpE2Original = candidates["fpPoiExpE2"] * 1  ################

        if self.args.UseFactorLEE:
            FinalpSimExp = candidates["fpSimExp"] * FactorLEE
            FinalpSimExpE1 = candidates["fpSimExpE1"] * FactorLEE
            FinalpSimExpE2 = candidates["fpSimExpE2"] * FactorLEE
            FinalpPoiExp = candidates["fpPoiExp"] * FactorLEE
            FinalpPoiExpE1 = candidates["fpPoiExpE1"] * FactorLEE
            FinalpPoiExpE2 = candidates["fpPoiExpE2"] * FactorLEE


            FinalpSimExp[FinalpSimExp>1.0] = 1.0
            FinalpPoiExp[FinalpPoiExp>1.0] = 1.0


            FinalpSimExpE1[ (FinalpSimExp - FinalpSimExpE1) < 0] = FinalpSimExp[ (FinalpSimExp - FinalpSimExpE1) < 0]
            FinalpPoiExpE1[ (FinalpPoiExp - FinalpPoiExpE1) < 0] = FinalpPoiExp[ (FinalpPoiExp - FinalpPoiExpE1) < 0]


            FinalpSimExpE2[ (FinalpSimExp + FinalpSimExpE2) > 1] = 1.0 - FinalpSimExp[ (FinalpSimExp + FinalpSimExpE2) > 1]
            FinalpPoiExpE2[ (FinalpPoiExp + FinalpPoiExpE2) > 1] = 1.0 - FinalpPoiExp[ (FinalpPoiExp + FinalpPoiExpE2) > 1]

        else:
            aux1 = candidates["fpSimExp"] - candidates["fpSimExpE1"]
            aux2 = candidates["fpSimExp"] + candidates["fpSimExpE2"]
            aux1 = 1.0 - (1.0 - aux1)**(self.args.MaxSigmas)
            aux2 = 1.0 - (1.0 - aux2)**(self.args.MaxSigmas)
            FinalpSimExp = 1.0 - (1.0 -  candidates["fpSimExp"])**(self.args.MaxSigmas)
            FinalpSimExpE1 = FinalpSimExp - aux1
            FinalpSimExpE2 = aux2 - FinalpSimExp

            aux1 = candidates["fpPoiExp"] - candidates["fpPoiExpE1"]
            aux2 = candidates["fpPoiExp"] + candidates["fpPoiExpE2"]
            aux1 = 1.0 - (1.0 - aux1)**(self.args.MaxSigmas)
            aux2 = 1.0 - (1.0 - aux2)**(self.args.MaxSigmas)
            FinalpPoiExp = 1.0 - (1.0 - candidates["fpPoiExp"])**(self.args.MaxSigmas)
            FinalpPoiExpE1 = FinalpPoiExp - aux1
            FinalpPoiExpE2 = aux2 - FinalpPoiExp

        hdulist =   fits.open(self.args.Cube,memmap=True)
        w = wcs.WCS(hdulist[0].header)

        [ra,dec,freq,stoke] =  w.all_pix2world(candidates["fX"], candidates["fY"], candidates["fChannel"],np.zeros_like(candidates["fChannel"]),0)
        c = []
        for i in range(len(ra)):
            c.append(SkyCoord(ra[i], dec[i], frame='icrs', unit='deg'))

        Output = open(f'LineCandidates{self.type_LineCandidate}.dat','w')
        Output.write('#ID RA DEC FREQ SigmaSN SigmaSp SN P PE1 PE2 PSim PSimE1 PSimE2 PPesimistic PPesimisticE1 PPesimisticE2 PSimPesimistic PSimPesimisticE1 PSimPesimisticE2\n')

        if to_csv:
            csv_output = open(f'LineCandidates{self.type_LineCandidate}.csv','w')
            csv_output.write('ID,RA,DEC,FREQ,SigmaSN,SigmaSp,SN,P,PE1,PE2,PSim,PSimE1,PSimE2,PPesimistic,PPesimisticE1,PPesimisticE2,PSimPesimistic,PSimPesimisticE1,PSimPesimisticE2\n')
  
        for i in range(len(candidates["fX"])):
            k = i + 1
            i = len(candidates["fX"])-i-1
        
            Line = self.args.SurveyName+'_EL.'+str(k).zfill(2)+' '+c[i].to_string('hmsdms',sep=':',precision=3).split()[0]+' '
            Line = Line + c[i].to_string('hmsdms',sep=':',precision=3).split()[1]+' '+str(round(freq[i]/1e9,3)).zfill(3)+' ' + str(self.FinalSigma[i])+ ' ' +str(self.FinalSpatial[i]) + ' '+str(round(candidates["fSN"][i],1))+' '
            Line = Line +format(FinalpPoiExpOriginal[i],'.2f')+' '+format(FinalpPoiExpE1Original[i],'.2f')+' '+format(FinalpPoiExpE2Original[i],'.2f')+' '  
            Line = Line +format(FinalpSimExpOriginal[i],'.2f')+' '+format(FinalpSimExpE1Original[i],'.2f')+' '+format(FinalpSimExpE2Original[i],'.2f')+' '
            Line = Line +format(FinalpPoiExp[i],'.2f')+' '+format(FinalpPoiExpE1[i],'.2f')+' '+format(FinalpPoiExpE2[i],'.2f')+' '  
            Line = Line +format(FinalpSimExp[i],'.2f')+' '+format(FinalpSimExpE1[i],'.2f')+' '+format(FinalpSimExpE2[i],'.2f')+'\n'
            Output.write(Line)

            if to_csv:
                csv_line = Line.strip()          
                csv_line = csv_line.replace(" ", ",")
                csv_output.write(csv_line + "\n")

        Output.close()
        if to_csv:
            csv_output.close()
        pass


class Positive(RealSources):
    def __init__(self, args):
        super().__init__(args)
        self.type = 'pos'
        self.type_LineCandidate = "Positive" 

class Negative(RealSources):
    def __init__(self, args):
        super().__init__(args)
        self.type = 'neg'
        self.type_LineCandidate = "Negative" 