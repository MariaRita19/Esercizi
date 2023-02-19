"""
PROGETTO - Corso Abilità Informatiche 
Created on 17 Dec 2022
Author: Maria Rita Mancino

"""
# Librerie
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt

Nbins = 200         # lunghezza vettore di dati
Nmeasures = 10000   # numero di misure 
measures = []       # lista vuota
test = 1            # test = 1,2,3


# 1. LEGGERE UN SINGOLO MULTIPOLO DA UN SET LIMITATO DI DATI

# Si hanno infatti i primi cinque multipoli dell'espansione in serie di Legendre della funzione a due punti
# I multipoli pari sono non nulli, gli altri sono nulli
# In questa prima parte dell'esercizio si considera un singolo multipolo
 
for i in np.arange(Nmeasures)+1:
    # scrivo il nome del file da leggere
    filename = f'C:/Users/pc/Documents/Maria Rita/ProgettoAB_INF/data/MockMeasures_2PCF_Test{test}/MockMeasures_2PCF_Correlation_MULTIPOLES_Test{test}_{i}.fits'
    file = fits.open(filename)      # aprire file FITS (formato astronomico per salvare tabelle)
    table = file[1].data.copy()     # accesso alla tabella
    measures.append(table['XI0'])   # estrarre dalla tabella la colonna XI0
    if i==1:
        scale = table['SCALE']
    del table
    file.close()
    # scale indica un vettore di numeri da 0 a 200 con passo 1 (discretizzazione)
    
measures = np.asarray(measures).transpose()     # asarray: converte input in un array, transpose: trasposizione matrice
# print(measures.shape)

# Calcolo media e covarianza con numpy
mean_xi = np.mean(measures,axis=1)            # axis: asse lungo cui si calcola la media
cov_xi = np.cov(measures)                     # calcolo covarianza con numpy


# 2. CALCOLARE LA COVARIANZA NUMERICA

average = np.zeros((Nbins,),dtype=float)
covariance = np.zeros((Nbins,Nbins),dtype=float)

for i in range(Nmeasures):
    average += measures[:,i]
average /= Nmeasures
    
for i in range(Nbins):
    for j in range(Nbins):
        covariance[i,j] = (np.sum(measures[i]*measures[j]) - average[i]*average[j]*Nmeasures) / (Nmeasures-1)
print('Differenza maggiore tra la covarianza numerica e quella calcolata con numpy: {}'.format(np.max(np.abs(covariance-cov_xi))))

# Matrice di correlazione (è la covarianza normalizzata a 1 sulla diagonale)
corr_xi = np.zeros((Nbins,Nbins),dtype=float)
for i in range(Nbins):
    for j in range(Nbins):
        corr_xi[i,j]=cov_xi[i,j]/(cov_xi[i,i]*cov_xi[j,j])**0.5


# 3. CALCOLARE LA COVARIANZA TEORICA

# Parametri da utilizzare per ciascuno dei tre set di misure
if test==1:
    sigma = [0.02, 0.02, 0.02]
    h = [25, 50, 75]
elif test==2:
    sigma = [0.02, 0.01, 0.005]
    h = [50, 50, 50]
else:
    sigma = [0.02, 0.01, 0.005]
    h = [5, 5, 5]

# Formule per il calcolo della covarianza teorica    
def cov_auto(x1, x2, sigma, h):   # autocorrelazione singolo multipolo
    return sigma**2.*np.exp(-(x1-x2)**2./(2.*h**2.))

def cov_mista(x1, x2, sigma1, h1, sigma2, h2):   # correlazione mista per multipoli correlati
    return (np.sqrt(2.*h1*h2)*np.exp(-(np.sqrt((x1-x2)**2.)**2./(h1**2.+h2**2.)))*sigma1*sigma2)/np.sqrt(h1**2.+h2**2.)

cov_th = np.zeros((Nbins,Nbins),dtype=float)
for i in range(Nbins):
    for j in range(Nbins):
        cov_th[i,j] = cov_auto(scale[i],scale[j],sigma[0],h[0])
     
        
# 4. FARE GRAFICI DELLE MATRICI DI COVARIANZA

gratio = (1.+5.**0.5)/2.
dpi = 300
#climit=max(np.max(theoretical_covariance),np.max(measured_covariance))
cmin = -np.max(cov_th)*0.05
cmax = np.max(cov_th)*1.05
        
# Plot della matrice di covarianza misurata
fig = plt.figure(figsize=(6,4))
plt.title('Matrice di covarianza misurata')
plt.imshow(cov_xi, vmin=cmin, vmax=cmax)
cbar = plt.colorbar(orientation="vertical", pad=0.02)
cbar.set_label(r'$ C^{\xi}_{N}$')
plt.show()

# Plot della matrice di covarianza teorica
fig = plt.figure(figsize=(6,4))
plt.title('Matrice di covarianza teorica')
plt.imshow(cov_th, vmin=cmin, vmax=cmax)
cbar = plt.colorbar(orientation="vertical", pad=0.02)
cbar.set_label(r'$ C^{\xi}_{N}$')
plt.show()

# Plot dei residui
fig = plt.figure(figsize=(6,4))
plt.title('Residui')
plt.imshow(cov_th-cov_xi, vmin=cmin, vmax=-cmin)
cbar = plt.colorbar(orientation="vertical", pad=0.02)
cbar.set_label(r'$ C^{\xi}_{N}$')
plt.show()


# 5. CONFRONTARE COVARIANZA NUMERICA E TEORICA:
#    CALCOLARE LA DIFFERENZA QUADRATICA MEDIA DEI RESIDUI NORMALIZZATI (deve essere ~1: validazione)  
  
print('CONFRONTO COVARIANZA NUMERICA E TEORICA')
norm_residuals = np.zeros_like(cov_th)
for i in range(Nbins):
    for j in range(Nbins):
        rho2 = cov_th[i,j]**2./(np.sqrt(cov_th[i,i]*cov_th[j,j])**2.)
        norm_residuals[i,j]=(cov_th[i,j]-cov_xi[i,j])*np.sqrt((Nmeasures-1.)/((1.+rho2)*cov_th[i,i]*cov_th[j,j]))

rms_deviation = np.std(norm_residuals.reshape(Nbins**2))
print(f"Differenza quadratica media dei residui normalizzati: {rms_deviation}")
if rms_deviation<1.1:
    print("La diff. quadratica media dei residui normalizzati è ~1 come dovrebbe essere; ok!")
else:
    print("La diff. quadratica media dei residui normalizzati NON è ~1; non va bene")


# 6. ESTENDERE LA PROCEDURA A TRE MULTIPOLI PARI (0, 2 E 4) INCLUDENDO LE CROSS CORRELAZIONI
# I multipoli pari sono non nulli

Nbins_tot = 600
measures_tot = []
meas0 = []
meas2 = []
meas4 = []

for i in np.arange(Nmeasures)+1:
    filename = f'C:/Users/pc/Documents/Maria Rita/ProgettoAB_INF/data/MockMeasures_2PCF_Test{test}/MockMeasures_2PCF_Correlation_MULTIPOLES_Test{test}_{i}.fits'
    file = fits.open(filename)
    table = file[1].data.copy()
    meas0.append(table['XI0']) 
    meas2.append(table['XI2'])
    meas4.append(table['XI4'])    
    if i==1:
        scale = table['SCALE']
    del table
    file.close()

meas0 = np.asarray(meas0).transpose()
meas2 = np.asarray(meas2).transpose()
meas4 = np.asarray(meas4).transpose()

measures_tot = np.concatenate((meas0,meas2,meas4))  # concatenate: unire una sequenza di matrici lungo un asse 
# print(misure.shape)


# Calcolo autocorrelazioni singolo multipolo
cov0_th = np.zeros((Nbins,Nbins),dtype=float)
for i in range(Nbins):
    for j in range(Nbins):
        cov0_th[i,j] = cov_auto(scale[i],scale[j],sigma[0],h[0])

cov1_th = np.zeros((Nbins,Nbins),dtype=float)
for i in range(Nbins):
    for j in range(Nbins):
        cov1_th[i,j] = cov_auto(scale[i],scale[j],sigma[1],h[1])

cov2_th = np.zeros((Nbins,Nbins),dtype=float)
for i in range(Nbins):
    for j in range(Nbins):
        cov2_th[i,j] = cov_auto(scale[i],scale[j],sigma[2],h[2])
        
# Calcolo correlazioni miste multipoli
cov01_th = np.zeros((Nbins,Nbins),dtype=float)
for i in range(Nbins):
    for j in range(Nbins):
        cov01_th[i,j] = cov_mista(scale[i],scale[j],sigma[0],h[0],sigma[1],h[1])

cov02_th = np.zeros((Nbins,Nbins),dtype=float)
for i in range(Nbins):
    for j in range(Nbins):
        cov02_th[i,j] = cov_mista(scale[i],scale[j],sigma[0],h[0],sigma[2],h[2])

cov12_th = np.zeros((Nbins,Nbins),dtype=float)
for i in range(Nbins):
    for j in range(Nbins):
        cov12_th[i,j] = cov_mista(scale[i],scale[j],sigma[1],h[1],sigma[2],h[2])

covar1_th = np.hstack((cov0_th,cov01_th,cov02_th))        # hstack: unire le tre matrici in orizzontale
covar2_th = np.hstack((cov01_th,cov1_th,cov12_th)) 
covar3_th = np.hstack((cov02_th,cov12_th,cov2_th)) 

cov_th_tot = np.vstack((covar1_th,covar2_th,covar3_th))   # vstack: unire le tre matrici in verticale

# Plot della matrice di covarianza teorica
fig = plt.figure(figsize=(6,4))
plt.title('Matrice di covarianza teorica')
plt.imshow(cov_th_tot, vmin=cmin, vmax=cmax)
cbar = plt.colorbar(orientation="vertical", pad=0.02)
cbar.set_label(r'$ C^{\xi}_{N}$')
plt.show()


# 7. VALIDARE IL CALCOLO DELLA COVARIANZA NUMERICA SUI TRE SET DI MISURE

# Calcolo media e covarianza con numpy
media_xi = np.mean(measures_tot,axis=1)            
covar_xi = np.cov(measures_tot)              

media = np.zeros((Nbins_tot,),dtype=float)
covarianza = np.zeros((Nbins_tot,Nbins_tot),dtype=float)

for i in range(Nmeasures):
    media += measures_tot[:,i]
media /= Nmeasures
    
for i in range(Nbins_tot):
    for j in range(Nbins_tot):
        covarianza[i,j] = (np.sum(measures_tot[i]*measures_tot[j]) - media[i]*media[j]*Nmeasures) / (Nmeasures-1)


# Matrice di correlazione (è la covarianza normalizzata a 1 sulla diagonale)
correl_xi = np.zeros((Nbins_tot,Nbins_tot),dtype=float)
for i in range(Nbins_tot):
    for j in range(Nbins_tot):
        correl_xi[i,j]=covar_xi[i,j]/(covar_xi[i,i]*covar_xi[j,j])**0.5

# Plot della matrice di covarianza misurata
fig = plt.figure(figsize=(6,4))
plt.title('Matrice di covarianza misurata')
plt.imshow(covar_xi, vmin=cmin, vmax=cmax)
cbar = plt.colorbar(orientation="vertical", pad=0.02)
cbar.set_label(r'$ C^{\xi}_{N}$')
plt.show()

