from operator import concat
from numpy.lib.function_base import append
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statistics as st
from sklearn.linear_model import LinearRegression

liverpool = pd.read_csv('C:\Python\Liverpool\Liverpool\Liverpool.csv', parse_dates=[
                        "Date"], index_col="Date")
# print(liverpool)

liv_mex = liverpool.drop(['CETES91'], axis='columns')
c = liverpool.drop(['LIVEPOLC1', 'MEXBOL'], axis='columns')
cetes = c.drop(c.index[0])

rend = np.log(liv_mex).diff().dropna() * 100
cetes = cetes/36000
# print(rend)

premio = cetes.assign(Premio=rend['MEXBOL'] - cetes['CETES91'])

intercepto = premio.assign(Intercepto=1)
# print(intercepto)

rendimientos = pd.concat([rend, intercepto], axis=1)
# print(rendimientos)

Liv = rendimientos['LIVEPOLC1']
Mexbol = rendimientos['MEXBOL']

# print(len(Liv[-1+1:753-91]))

Beta90 = []  # Vector vacio para almacenar datos
for n in range(0, 753):
    if n >= 0 and n <= 752-91:
        cov = np.cov(Liv[n:n+91], Mexbol[n:n+91])[0][1]
        var = np.var(Mexbol[n:n+91])
        Beta90.append(cov/var)  # Se agregan datos al vector vacio

# El vector se convierte en dataframe
Beta90 = pd.DataFrame(Beta90, columns=['Beta 90'])
Beta = Beta90['Beta 90']

desv = st.stdev(Beta)
media = st.mean(Beta)
Tipo = 1
Alpha = 0.05

dnorm = st.NormalDist(media, desv).inv_cdf(Alpha/2)
dnorm2 = st.NormalDist(media, desv).inv_cdf(0.975)
percentil = np.percentile(Beta, 2.5)
percentil2 = np.percentile(Beta, 97.5)

Tab = Beta90.assign(Lim_inf=dnorm)
Beta90_lim = Tab.assign(Lim_Sup=dnorm2)

Beta90_lim.plot()
# plt.show()

Min_Liv = min(Liv)
Max_Liv = max(Liv)
Prom_Liv = st.mean(Liv)
Rango_Liv = (Max_Liv - Prom_Liv)/100
Grupos = 100
Amplitud_Liv = (Max_Liv-Min_Liv)/Grupos
#print(Min_Liv, Max_Liv, Prom_Liv, Rango_Liv, Grupos, Amplitud_Liv)

Min_Mb = min(Mexbol)
Max_Mb = max(Mexbol)
Prom_Mb = st.mean(Mexbol)
Rango_Mb = Max_Mb - Prom_Mb
Amplitud_Mb = ((Max_Mb-Min_Mb)/Grupos)/100
#print(Min_Mb, Max_Mb, Prom_Mb, Rango_Mb, Grupos, Amplitud_Mb)

Clase_liv = []
Clase_Mb = []
paso = []

for k in range(0, 101):
    Clase_liv.append(Min_Liv + (k * Amplitud_Liv))
    Clase_Mb.append(Min_Mb + (k * Amplitud_Mb))
    paso.append(k)

Clase_rend_liv = pd.DataFrame(paso, columns=['Paso'])
Clase_rend_Mb = Clase_rend_liv.assign(Clase_rend_Live=Clase_liv)
Clases = Clase_rend_Mb.assign(Clase_rend_mexbol=Clase_Mb)
# print(Clases)