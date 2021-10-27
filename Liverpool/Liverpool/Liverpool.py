import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools

liverpool = pd.read_csv('C:\Python\Liverpool\Liverpool\Liverpool.csv', parse_dates=[
                        "Date"], index_col="Date")
#print(liverpool)

liv_mex = liverpool.drop(['CETES91'], axis = 'columns')
c = liverpool.drop(['LIVEPOLC1', 'MEXBOL'], axis= 'columns')
cetes = c.drop(c.index[0])

rend = np.log(liv_mex).diff().dropna()
cetes = cetes/36000 

premio = rend['MEXBOL'] - cetes['CETES91']

intercepto = premio.insert(Intercepto = ['1']) 
print(intercepto)

#rendimientos = pd.concat([rend, cetes, intercepto] , axis=1)
#print(rendimientos)