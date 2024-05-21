import pyomo.environ as pyo
from pyomo.environ import *
import pandas as pd
import numpy as np

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

def minimize_area_constrained(distances: pd.DataFrame, 
                  homes: pd.DataFrame, 
                  locations: pd.DataFrame, 
                  periods: pd.DataFrame, 
                  per_period: int, 
                  max_distance: float, 
                  open_facilities: pd.DataFrame,
                  optimum: int,
                  vpop = None):
  #Model initialization
  model = pyo.ConcreteModel("Minimize Area")

  #Sets
  model.I = pyo.Set(initialize = homes.index)
  model.J = pyo.Set(initialize = locations.index)
  model.N = pyo.Set(initialize = periods)

  model.N_hat = pyo.Set(initialize = periods[:-1])

  #Variables
  model.z = pyo.Var(model.I, model.N, within=pyo.Binary) #household i is covered in period N
  model.x = pyo.Var(model.J, model.N, within=pyo.Binary) #facility j is open at period n

  if vpop is None:
    vpop = np.ones(len(homes))

  #Objective function
  model.objective = pyo.Objective(
    sense = pyo.maximize,
    expr = pyo.quicksum(
      ((vpop[i] * (model.z[i, n] + model.z[i, n+1]) / 2) for i in model.I for n in model.N_hat)
    ),
    doc="Maximize the area under the building curve"
  )

  #Constraints
  def open_max_n_per(model, n):
    return pyo.inequality(
      0, pyo.quicksum(model.x[j, n] for j in model.J if open_facilities[j] != 1), n * per_period
    )
  model.open_max = pyo.Constraint(model.N, rule=open_max_n_per) #in each period n there is n*per_period facilities open

  def max_distance_n_km(model, i, n):
    return sum(
      model.x[j, n] for j in model.J if distances[i][j] < max_distance
    ) >= model.z[i, n]
  model.max_distance = pyo.Constraint(model.I, model.N, rule=max_distance_n_km)

  def keep_houses(model, i, n):
    return model.z[i, n+1] >= model.z[i, n]
  model.keep_houses = pyo.Constraint(model.I, model.N_hat, rule=keep_houses)

  def keep_facilities(model, j, n):
    return model.x[j, n+1] >= model.x[j, n]
  model.keep_facilites = pyo.Constraint(model.J, model.N_hat, rule=keep_facilities)

  def opened_facilities(model, j):
    return(model.x[j, 1] >= open_facilities[j])
  model.opened_facilities = pyo.Constraint(model.J, rule=opened_facilities)

  def keep_optimum(model):
    return (pyo.quicksum(model.z[i,periods.iloc[-1]] * vpop[i] for i in model.I) == optimum)
  model.optimum = pyo.Constraint(rule=keep_optimum)

  return model

def two_step_constrained(distances: pd.DataFrame, 
                  homes: pd.DataFrame, 
                  locations: pd.DataFrame, 
                  periods: pd.DataFrame, 
                  per_period: int, 
                  max_distance: float, 
                  open_facilities: pd.DataFrame,
                  vpop = None):
  
  #initialize basic model
  model_step_one = basic_model(
    distances=distances,
    homes=homes,
    locations=locations,
    build_max=per_period * periods.iloc[-1],
    max_distance=max_distance,
    open_facilities=open_facilities,
    vpop = vpop
  )

  #set correct solver
  solver = pyo.SolverFactory("gurobi")

  #solve basic model
  result1 = solver.solve(model_step_one)

  if (result1.solver.status == pyo.SolverStatus.error):
    raise RuntimeError("Solver failed to find a solution")
  
  #get the end optimum value
  optimum = sum([pyo.value(model_step_one.z[i] * vpop[i]) for i in model_step_one.I])

  #initialize second model with calculated optimum
  model_step_two = minimize_area_constrained(
    distances=distances,
    homes=homes,
    locations=locations,
    periods=periods,
    per_period=per_period,
    max_distance=max_distance,
    open_facilities=open_facilities,
    optimum=optimum,
    vpop = vpop
  )

  #solve second model
  result2 = solver.solve(model_step_two)

  if (result2.solver.status == pyo.SolverStatus.error):
    raise RuntimeError("Solver failed to find a solution")
  
  #collect results
  x_result = [[pyo.value(model_step_two.x[j, n]) for j in model_step_two.J] for n in model_step_two.N]
  z_result = [[pyo.value(model_step_two.z[i, n] * vpop[i]) for i in model_step_two.I] for n in model_step_two.N]

  all_z = z_result
  all_x = x_result
  building_curve = [sum(sublist) for sublist in z_result]

  return all_z, all_x, building_curve