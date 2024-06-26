import pyomo.environ as pyo
from pyomo.environ import *
import pandas as pd
import numpy as np

# 
# Calculating the pareto curve is done by solving the the basic model for each period, where in each period it is solved for the number of facilities that should have been built up to that point
# 

def basic_model(distances: pd.DataFrame, 
                homes: pd.DataFrame, 
                locations: pd.DataFrame, 
                max_distance: float,
                build_max: int,
                open_facilities: pd.DataFrame, 
                vpop: pd.DataFrame):
  model = ConcreteModel("Pareto")

  #Sets
  model.I = pyo.Set(initialize = homes.index)
  model.J = pyo.Set(initialize = locations.index)

  #Variables
  model.z = pyo.Var(model.I, within=pyo.Binary)
  model.x = pyo.Var(model.J, within=pyo.Binary)

  if vpop is None:
    vpop = np.ones(len(homes))

  #Objective function
  model.objective = pyo.Objective(
    sense=pyo.maximize,
    expr=pyo.quicksum(
      vpop[i] * model.z[i] for i in model.I
    )
  )

  #Constraints
  def open_max_n(model):
    return pyo.inequality(
      0, pyo.quicksum(model.x[j] for j in model.J if open_facilities[j] != 1), build_max
    )
  model.open_max = pyo.Constraint(rule = open_max_n)

  def max_distance_n_km(model, i):
    return sum(
      model.x[j] for j in model.J if distances[i][j] < max_distance
    ) >= model.z[i]
  model.max_distance = pyo.Constraint(model.I, rule = max_distance_n_km)

  def opened_facilities(model, j):
    return(model.x[j] >= open_facilities[j])
  model.opened_facilities = pyo.Constraint(model.J, rule=opened_facilities)

  return model

def get_pareto_curve(distances: pd.DataFrame, 
                  homes: pd.DataFrame, 
                  locations: pd.DataFrame, 
                  periods:pd.DataFrame,
                  per_period: int,
                  max_distance: float,
                  open_facilities: pd.DataFrame, 
                  vpop: pd.DataFrame):

  #prepare to collect results
  x_result = []
  z_result = []

  #select the solver
  solver = pyo.SolverFactory("gurobi")

  for period in periods:
    #initialize the model for each period
    model =  basic_model(
      distances=distances, 
      homes=homes, 
      locations=locations, 
      max_distance=max_distance,
      build_max=(period)*per_period,
      open_facilities=open_facilities, 
      vpop=vpop
    )

    #Solve the model
    result = solver.solve(model)

    #save results
    x_result.append([pyo.value(model.x[j]) for j in model.J])
    z_result.append([pyo.value(model.z[i] * vpop[i]) for i in model.I])

  all_z = z_result
  all_x = x_result
  pareto_curve = [sum(sublist) for sublist in z_result]

  return all_z, all_x, pareto_curve