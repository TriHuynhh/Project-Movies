import math

Paris=[200,20,200,'Paris']
London = [250,30,120,'London']
Dubai = [370,15,80,'Dubai']
Mumbai = [450,10,70,'Mumbai']
Cities = [Paris,London,Dubai,Mumbai]

total_days = 7
def total_cost(flight,hotel,car,days=0):
    return flight+(hotel*days)+(car*math.ceil(days/7))



def least_expensive(total_days):
    costs = []
    for city in Cities:
        cost = total_cost(city[0],city[1],city[2],total_days)
        costs.append((cost,city[3]))
    min_cost = min(costs)
    return min_cost

least_expensive(total_days)