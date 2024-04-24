import pyomo.environ as pyo
from pyomo.environ import *
import pandas as pd

def minimize_area(distances, homes: int, locations: int, periods: int, per_period: int, max_distance: float):
  #Model initialization
  model = pyo.ConcreteModel("Minimize Area")

  #Sets
  model.I = pyo.Set(initialize = range(homes))
  model.J = pyo.Set(initialize = range(locations))
  model.N = pyo.Set(initialize = range(periods))

  #Variables
  model.z = pyo.Var(model.I, model.N, within=pyo.Binary) #household i is covered in period N
  model.x = pyo.Var(model.J, model.N, within=pyo.Binary) #facility j is open at period n

  #Objective function
  model.objective = pyo.Objective(
    sense = pyo.maximize,
    expr = pyo.quicksum(
      (((model.z[i, n] + model.z[i, n+1]) / 2) for i in model.I for n in model.N[:-1])
    ),
    doc="Maximize the area under the building curve"
  )

  #Constraints
  def open_max_n_per(model, n):
    return pyo.inequality(
      0, pyo.quicksum(model.x[j, n] for j in model.J), n * per_period
    )
  model.open_max = pyo.Constraint(model.N, rule=open_max_n_per) #in each period n there is n*per_period facilities open

  def max_distance_n_km(model, i, n):
    return sum(
      model.x[j, n] for j in model.J if distances[j][i] < max_distance
    ) >= model.z[i, n]
  model.max_distance = pyo.Constraint(model.I, model.N, rule=max_distance_n_km)

  def keep_houses(model, i, n):
    return model.z[i, n+1] >= model.z[i, n]
  model.keep_houses = pyo.Constraint(model.I, model.N[:-1], rule=keep_houses)

  def keep_facilities(model, j, n):
    return model.x[j, n+1] >= model.x[j, n]
  model.keep_facilites = pyo.Constraint(model.J, model.N[:-1], rule=keep_facilities)

  return model

#Define and run a specific model
def single_step_area(distances, homes: int, locations: int, periods: int, per_period: int, max_distance: float):
  #Initialize the model
  model = minimize_area(
    distances=distances,
    homes=homes,
    locations=locations,
    periods=periods,
    per_period=per_period,
    max_distance=max_distance,
  )

  #Get solver
  solver = pyo.SolverFactory("gurobi")

  #Solve the model
  result = solver.solve(model)

  if (result.solver.status == pyo.SolverStatus.error):
    raise RuntimeError("Solver failed to find a solution")
  
  #collect results
  x_result = [[pyo.value(model.x[j, n]) for j in model.J] for n in model.N]
  z_result = [[pyo.value(model.z[i, n]) for i in model.I] for n in model.N]

  last_z = z_result[-1]
  last_x = x_result[-1]
  building_curve = z_result

  return last_z, last_x, building_curve
