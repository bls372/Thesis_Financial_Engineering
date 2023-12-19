import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as sc
K = 1
sigma = 0.2
r = 0.06
w = 0.5
n = 2000
p = 9
Nhedge = 52
Nrep = 1000
capT = 1
dt = capT/Nhedge
def TruePrice(price, TimeToMat):
    d1 = (np.log(price/K) + (r + 0.5*sigma**2)*TimeToMat)/(sigma*np.sqrt(TimeToMat))
    d2 = d1 - sigma*np.sqrt(TimeToMat)
    result = np.exp(-r*TimeToMat)*K*sc.norm.cdf(-d2) - price*sc.norm.cdf(-d1)
    return(result)
def TrueDelta(price, TimeToMat):
    d1 = (np.log(price/K) + (r + 0.5*sigma**2)*TimeToMat)/(sigma*np.sqrt(TimeToMat))
    result = -sc.norm.cdf(-d1)
    return(result)
coeffsDeg = np.zeros((Nhedge,  p+1))
coeffsReg = np.zeros((Nhedge, p+1))
Ds = np.zeros((n, Nhedge))
DiscPayoffs = np.zeros((n, Nhedge))
Z = np.random.normal(0, 1, size=n)
S0 = K*np.exp((r-0.5*sigma**2)*capT + sigma*np.sqrt(capT)*Z)
St = np.zeros((n, Nhedge +1))
St[:,0] = S0      
for i in range(Nhedge):
    Z = np.random.normal(0, 1, size=n)
    St[:,i+1] = St[:,i]*np.exp((r-0.5*sigma**2)*dt + sigma*np.sqrt(dt)*Z)
ST = St[:, Nhedge]
payoff = (K-ST)*(ST<K)
for i in range(Nhedge):
    TimeToMat = (capT-i*dt)
    S0 = St[:, i]
    X = np.vander((S0-K), p+1, increasing=True)
    Y = np.c_[np.zeros(n), np.ones(n), range(2,p+1)*np.delete(X, [0,p], 1)]
    P = payoff*np.exp(-r*TimeToMat)
    DiscPayoffs[:,i] = P
    coeffsDeg[i,:] = np.matmul(np.linalg.inv(np.matmul(np.transpose(X),X))
                               , np.matmul(np.transpose(X),P))
    D = -1*np.exp(-r*TimeToMat)*(ST<K)*ST/S0
    Ds[:,i] = D
    coeffsReg[i,:]= np.matmul(np.linalg.inv(w*np.matmul(np.transpose(X),X)
                                            + (1-w)*np.matmul(np.transpose(Y), Y))
                              , w*np.matmul(np.transpose(X), P)
                              + (1-w)*np.matmul(np.transpose(Y), D))
S0 = 1
initialoutlay = TruePrice(S0, capT)           
Vpf = np.zeros((Nrep, Nhedge +1))
St = np.zeros((Nrep, Nhedge +1))
St[:,0]=S0
Vpf[:,0] = initialoutlay
b = np.zeros((Nrep, Nhedge +1))
a = np.zeros((Nrep, Nhedge +1))          
a[:,0] = TrueDelta(S0, capT)
b[:,0] = Vpf[:,0] - a[:,0]*St[:,0]
VpfDeg = np.zeros((Nrep, Nhedge +1))
VpfReg = np.zeros((Nrep, Nhedge +1))          
VpfDeg[:,0] = initialoutlay
VpfReg[:,0] = initialoutlay
bDeg = np.zeros((Nrep, Nhedge +1))
bReg = np.zeros((Nrep, Nhedge +1))
aDeg = np.zeros((Nrep, Nhedge +1))
aReg = np.zeros((Nrep, Nhedge +1))           
X = np.vander((St[:,0]-K), p+1, increasing=True)
Y = np.c_[np.zeros(Nrep), np.ones(Nrep), range(2,p+1)*np.delete(X, [0,p], 1)]
aDeg[:,0] = np.matmul(Y, coeffsDeg[0, :])
bDeg[:,0] = VpfDeg[:,0] - aDeg[:,0]*St[:,0]
aReg[:,0] = np.matmul(Y, coeffsReg[0, :])
bReg[:,0] = VpfReg[:,0] - aReg[:,0]*St[:,0]
for i in range(1, Nhedge):
    Z = np.random.normal(0, 1, size=Nrep)
    St[:,i] = St[:,i-1]*np.exp((r-0.5*sigma**2)*dt+sigma*np.sqrt(dt)*Z)          
    VpfDeg[:,i] = aDeg[:,i-1]*St[:,i] + bDeg[:,i-1]*np.exp(r*dt)
    VpfReg[:,i] = aReg[:,i-1]*St[:,i] + bReg[:,i-1]*np.exp(r*dt)
    Vpf[:,i] = a[:,i-1]*St[:,i] + b[:,i-1]*np.exp(r*dt)
    X = np.vander((St[:,i]-K), p+1, increasing=True)
    Y = np.c_[np.zeros(Nrep), np.ones(Nrep), range(2,p+1)*np.delete(X, [0,p], 1)]
    aDeg[:,i] = np.matmul(Y, coeffsDeg[i, :])
    bDeg[:,i] = (VpfDeg[:,i] - aDeg[:,i]*St[:,i])
    aReg[:,i] = np.matmul(Y, coeffsReg[i, :])
    bReg[:,i] = (VpfReg[:,i] - aReg[:,i]*St[:,i])
    TimeToMat = (capT-i*dt)
    a[:,i] = TrueDelta(St[:,i], TimeToMat)
    b[:,i] = (Vpf[:,i] - a[:,i]*St[:,i])
Z = np.random.normal(0, 1, size=Nrep)
St[:,Nhedge] = St[:,Nhedge-1]*np.exp((r-0.5*sigma**2)*dt+sigma*np.sqrt(dt)*Z)
VpfDeg[:,Nhedge] = aDeg[:,Nhedge-1]*St[:,Nhedge] + bDeg[:,Nhedge-1]*np.exp(r*dt)
VpfReg[:,Nhedge] = aReg[:,Nhedge-1]*St[:,Nhedge] + bReg[:,Nhedge-1]*np.exp(r*dt)
Vpf[:,Nhedge] = a[:,Nhedge-1]*St[:,Nhedge] + b[:,Nhedge-1]*np.exp(r*dt)
payoff_hedge = (K-St[:,Nhedge])*(St[:,Nhedge]<K)           
hedgeerror = (Vpf[:,Nhedge] - payoff_hedge)
relSTDhedgeError = np.std(hedgeerror)/initialoutlay
hedgeerrorDeg = (VpfDeg[:,Nhedge] - payoff_hedge)
relSTDhedgeErrorDeg = np.std(hedgeerrorDeg)/initialoutlay
hedgeerrorReg = (VpfReg[:,Nhedge] - payoff_hedge)
relSTDhedgeErrorReg = np.std(hedgeerrorReg)/initialoutlay
STs = St[:,Nhedge]
Payoffs = payoff_hedge
Vpf = Vpf[:,Nhedge]
Vpfdeg = VpfDeg[:,Nhedge]
Vpfreg = VpfReg[:,Nhedge]
data = {"ST": STs,"payoff": Payoffs, "Vpf": Vpf, "Vpfdeg": Vpfdeg , "Vpfreg": Vpfreg}
df = pd.DataFrame(data)
dfsort = df.sort_values("ST")
plt.plot()
plt.scatter(dfsort["ST"], dfsort["Vpf"], label = "Vpf using True Delta"
            , color = "black")
plt.scatter(dfsort["ST"], dfsort["Vpfdeg"], label = "Vpf price-only-regression"
            , edgecolor = "red", c = "pink", marker = "s")
plt.scatter(dfsort["ST"], dfsort["Vpfreg"], label = "Vpf Delta-regularization"
            ,  edgecolor = "blue", c ="green", marker = "^")
plt.plot(dfsort["ST"], dfsort["payoff"], label = "TruePayoff", color = "black")
plt.xlabel('STs')
plt.ylabel('Vpf')
plt.title('Replicating Portfolio values at expiry')
plt.text(1.5, 0.2, 'True HedgeError = ' + str(round(relSTDhedgeError, 6))
         , fontsize = 10, color = 'black')
plt.text(1.5, 0.18, 'Degree HedgeError = ' + str(round(relSTDhedgeErrorDeg, 6))
         , fontsize = 10, color = 'red')
plt.text(1.5, 0.16, 'Reg HedgeError = ' + str(round(relSTDhedgeErrorReg, 6))
         , fontsize = 10, color = 'blue')
plt.text(1.5, 0.14, 'PolDeg =' + str(p), fontsize = 10, color = 'black')
plt.text(1.5, 0.12, 'Nsim =' + str(n), fontsize = 10, color = 'black')
plt.text(1.5, 0.10, 'Nrep =' + str(Nrep), fontsize = 10, color = 'black')
plt.text(1, 0.20, 'InitialOutlay = ' + str(round(initialoutlay, 6)), fontsize = 10
         , color = 'black')
plt.text(1, 0.18, 'Nhedge =' + str(Nhedge), fontsize = 10, color = 'black')
plt.legend()
plt.show()

