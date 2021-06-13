import pulp
import numpy as np
import networkx as nx


def get_mds_matching(g):
    if not nx.is_weighted(g):
        G = nx.DiGraph()
        G.add_weighted_edges_from([(edge[0], edge[1], 1) for edge in g.edges()])
        g = G

    if len(set(g.nodes())) != g.number_of_nodes():
        mapping = {}
        idx = 0
        for n in g.nodes():
            mapping[n] = idx
            idx += 1
        nx.relabel_nodes(g, mapping)
    
    edge_list = np.array(g.edges())
    node_list = np.array(g.nodes())
    edge_weight = [g[edge[0]][edge[1]]["weight"] for edge in g.edges()]
    n_col = len(edge_list)

    problem = pulp.LpProblem('get_mds_matching', pulp.LpMaximize)


    conp = np.zeros((g.number_of_nodes(), n_col))
    conm = np.zeros((g.number_of_nodes(), n_col))

    for i in range(g.number_of_nodes()):
        idx = np.where(edge_list[:,0] == node_list[1])[0]
        conp[i, idx] = 1


    for i in range(g.number_of_nodes()):
        idx = np.where(edge_list[:,1] == node_list[1])[0]
        conm[i, idx] = 1


    x = {}
    for i in range(n_col):
        x[i] = pulp.LpVariable(f"{i}", 0,1, pulp.LpBinary )

    problem += pulp.lpSum(x[i] for i in range(n_col)), "TotalCost"
    for i in range(g.number_of_nodes()):
        problem += sum(conp[i][j] * x[j] for j in range(n_col)) <= 1, f"Constraint_{i}_1"

    for i in range(g.number_of_nodes()):
        problem += sum(conm[i][j] * x[j] for j in range(n_col)) <= 1, f"Constraint_{i}_2"

    solver = pulp.PULP_CBC_CMD()
    result_status = problem.solve(solver)

    res = [True if x[i].value() == 0.0 else False for i in range(n_col)]
    driver_node = set(g.nodes()) - set(edge_list[res, 1])

    return len(driver_node) ,driver_node, res

    



    





