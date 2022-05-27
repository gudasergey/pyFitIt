# match close atoms in two moleculas by maximum weighted bipartite matching
import numpy as np
import logging


# weights - numpy 2-dimensional array
def wbm(weights):
    import pulp
    pulp.LpSolverDefault.msg = False
    prob = pulp.LpProblem("WBM_Problem", pulp.LpMinimize)
    m,n = weights.shape
    # print(m,n)
    from_nodes = np.arange(m); to_nodes = np.arange(n)
    # Create The Decision variables
    choices = pulp.LpVariable.dicts("e",(from_nodes, to_nodes), 0, 1, pulp.LpInteger)

    # Add the objective function
    prob += pulp.lpSum([weights[u][v] * choices[u][v]
                   for u in from_nodes
                   for v in to_nodes]), "Total weights of selected edges"

    # Constraint set ensuring that the total from/to each node
    # is less than its capacity (= 1)
    ind1 = np.argsort(weights[:,0].reshape(-1))
    # print(ind1)
    ind2 = np.argsort(weights[0,:].reshape(-1))
    if from_nodes.size >= to_nodes.size:
        for v in to_nodes: prob += pulp.lpSum([choices[u][v] for u in from_nodes]) == 1, ""
        for i in range(m):
            #if i < n//2:
            if i < 0:
                prob += pulp.lpSum([choices[from_nodes[ind1[i]]][v] for v in to_nodes]) == 1, ""
            else: prob += pulp.lpSum([choices[from_nodes[ind1[i]]][v] for v in to_nodes]) <= 1, ""
    else:
        for u in from_nodes: prob += pulp.lpSum([choices[u][v] for v in to_nodes]) == 1, ""
        for i in range(n):
            #if i < m//2:
            if i < 0:
                prob += pulp.lpSum([choices[u][to_nodes[ind2[i]]] for u in from_nodes]) == 1, ""
            else: prob += pulp.lpSum([choices[u][to_nodes[ind2[i]]] for u in from_nodes]) <= 1, ""
    # The problem is solved using PuLP's choice of Solver
    prob.solve()

    # The status of the solution is printed to the screen
    # print( "Status:", pulp.LpStatus[prob.status])
    # Each of the variables is printed with it's resolved optimum value
    # for v in prob.variables():
    #     if v.varValue > 1e-3:
    #         print(f'{v.name} = {v.varValue}')
    # print(f"Sum of wts of selected edges = {round(pulp.value(prob.objective), 4)}")

    # print selected edges
    selected_from = [v.name.split("_")[1] for v in prob.variables() if v.value() > 1e-3]
    selected_to   = [v.name.split("_")[2] for v in prob.variables() if v.value() > 1e-3]
    selected_edges = []
    resultInd = np.zeros(m, dtype='int32')-1
    for su, sv in list(zip(selected_from, selected_to)):
        resultInd[int(su)] = int(sv)
        selected_edges.append((su, sv))
    resultIndExtra = np.copy(resultInd)
    forget = np.setdiff1d(np.arange(n), selected_to)
    if forget.size>0: resultIndExtra = np.concatenate([resultIndExtra,forget])
    return resultInd, resultIndExtra
