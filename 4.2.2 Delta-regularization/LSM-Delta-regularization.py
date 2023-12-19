import numpy as np
S0 = 1
K = 1
r = 0.06
sigma = 0.2
#This is the LSM algorithm extended with Delta-regularization,
# which is a function of method that can take
# "Price-only-regression" which is the normal LSM
# "Delta-regularization" the extended version that gives unbiased estimates
# with low variance (only Delta has lower variance)
def AmrLSM(spot, capT, Nhedge, NoSim, p, method):
    S0 = spot
    n = NoSim
    dt = capT/Nhedge
    w = 0.5
    St = np.zeros((n, Nhedge +1))
    St[:,0] = S0
    for i in range(Nhedge):
        Z = np.random.normal(0, 1, size=n)
        St[:,i+1] = St[:,i]*np.exp((r-0.5*sigma**2)*dt + sigma*np.sqrt(dt)*Z)
    STstop = np.exp(-r*dt)*St[:,Nhedge]
    ST = St[:,Nhedge]
    PayOffMatrix = (K-St)*(St<K)
    cashflowTJEK = np.exp(-r*dt)*PayOffMatrix[:,Nhedge]
    for i in range(Nhedge-1):
        q = Nhedge -1 -i
        inMonPosition = np.where(PayOffMatrix[:,q] !=0)
        x = St[inMonPosition, q]
        x = x[0]
        Y = cashflowTJEK[inMonPosition]
        X = np.vander(x-K, p+1, increasing=True)
        ST_in = ST[inMonPosition]
        discST_in = STstop[inMonPosition]
        D_in = -1*(ST_in<K)*discST_in/x
        Xdiff = np.c_[np.zeros(len(x)), np.ones(len(x)), range(2,p+1)
                      *np.delete(X, [0,p], 1)]
        if(method == "Price-only-regression"):
            coeff = np.matmul(np.linalg.inv(np.matmul(np.transpose(X),X))
                              , np.matmul(np.transpose(X), Y))
        if(method == "Delta-regularization"):
            coeff = np.matmul(np.linalg.inv(w*np.matmul(np.transpose(X),X)
                                            +(1-w)*np.matmul(np.transpose(Xdiff), Xdiff))
                              , w*np.matmul(np.transpose(X),Y)
                              +(1-w)*np.matmul(np.transpose(Xdiff),D_in))
        contVal = np.matmul(X, coeff)
        exVal = PayOffMatrix[inMonPosition,q]
        cashflowTJEK[inMonPosition] = exVal*(exVal>=contVal)*1 + Y*(exVal<contVal)
        STstop[inMonPosition] = x*(exVal>=contVal)*1+ discST_in*(exVal<contVal)
        ST[inMonPosition] = x*(exVal>=contVal)*1 + ST_in*(exVal<contVal)
        cashflowTJEK = np.exp(-r*dt)*cashflowTJEK
        STstop = np.exp(-r*dt)*STstop
    Price0 = np.mean(cashflowTJEK)
    discST = STstop
    D = -1*(ST<K)*discST/S0
    Delta0 = np.mean(D)
    result = {"Price": Price0, "Delta": Delta0}
    return(result)
print(AmrLSM(spot=S0, capT=1, Nhedge=52, NoSim=10000, p=3
             , method = "Delta-regularization"))
