import numpy as np

def GenerateFacilityLocationInstance(nofFacilities, nofCustumers, size):
    facilities = range(nofFacilities)
    customers = range(nofCustumers)
    xC = np.random.randint(0, size, nofCustumers)
    yC = np.random.randint(0, size, nofCustumers)
    xF = np.random.randint(0, size, nofFacilities)
    yF = np.random.randint(0, size, nofFacilities)

    installation = np.random.randint(1000, 2000, nofFacilities)

    dist = lambda i, j: ((xC[i] - xF[j]) ** 2 + (yC[i] - yF[j]) ** 2)

    service = [[dist(i, j) for j in facilities] for i in customers]

    return installation, service, xC, yC, xF, yF