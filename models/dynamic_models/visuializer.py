import matplotlib.pyplot as plt
import pandas as pd
from typing import List, Optional

def show_facility_location(xC: List[float], 
                         yC: List[float], 
                         xF: List[float], 
                         yF: List[float], 
                         X: List[float], 
                         Z: List[float], 
                         periods: pd.Series, 
                         homes,
                         locations,
                         served: Optional[int]=None):
    plt.plot( xC,yC, 'o' )
    plt.plot( xF,yF, 's', mfc='none' )
    [plt.plot( xC[i], yC[i], 'o', color="g") for i in range(len(homes)) for n in range(len(periods)) if Z[n][i]]
    for j in range(len(locations)):
      for n in range(len(periods)):
        if X[n][j] > .5:
          plt.plot( xF[j],yF[j], 's', color='y' )
        if not served is None:
            plt.title( '{:.2f}%'.format(served/len(xC)*100) )
    plt.show()