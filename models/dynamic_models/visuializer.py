import matplotlib.pyplot as plt
import pandas as pd
from typing import List, Optional
import numpy as np

def show_facility_location(xC: List[float], 
                         yC: List[float], 
                         xF: List[float], 
                         yF: List[float], 
                         X: List[float], 
                         Z: List[float], 
                         periods: pd.Series, 
                         homes,
                         locations,
                         vpop: pd.Series,
                         served: Optional[int]=None):
    plt.rcParams["figure.figsize"] = (13,8)
    plt.plot( xF,yF, 's', mfc='none' )
    # Plot households
    for i in range(len(homes)):
        if Z[-1][i]:
            plt.plot(xC[i], yC[i], 'o', color="g", markersize=vpop[i] / 100)  # Size based on population
        else:
            plt.plot(xC[i], yC[i], 'o', color="b", markersize=vpop[i] / 100)  # Default marker size
    for j in range(len(locations)):
      if X[-1][j] > .5:
          plt.plot( xF[j],yF[j], 's', color='y' )
      for n in range(len(periods)-1):
          if X[n][j] == 0 and X[n+1][j] == 1:
            plt.text( xF[j],yF[j], str(n+1))
    if not served is None:
        plt.title( '{:.2f}%'.format(served/len(xC)*100) )
    plt.show()