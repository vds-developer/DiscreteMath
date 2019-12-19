
import networkx as nx
import matplotlib.pyplot as plt
import math

#weighted connected graph with format
# list of [vertex a, vertex b, weight w]

def get_weight(elem:list):
    return elem[2]

def plot_prim(graph):
    network = nx.Graph()
    for edge in graph:
        network.add_edge(edge[0], edge[1], color=edge[3], weight= edge[2], step=edge[4])
    pos = nx.spring_layout(network)
    colors = [network[u][v]['color'] for u, v in network.edges()]
    nx.draw(network, pos, edges=network.edges(), edge_color=colors, width=5)
    nx.draw_networkx_labels(network, pos, font_size=20, font_family='sans-serif')
    labels = nx.get_edge_attributes(network, 'weight')
    nx.draw_networkx_edge_labels(network, pos, edge_labels=labels)
    step = nx.get_edge_attributes(network, 'step')
    nx.draw_networkx_edge_labels(network, pos, edge_labels=step, font_color='red')
    plt.show()


# Represent G as a list of edges, each edge being a triple [u, v, w], where w
# is the weight of the edge uv. Assume that this list is ordered by non-decreasing weights
# of the edges

# graph should be connected
def prim(graph : list  , num_v : int):
    graph.sort(key=get_weight)
    for edge in graph:
        edge.append('b')
        edge.append('')

    min_edge = graph[0]
    min_edge[3] = 'r'
    min_edge[4] = '1' + ' (w:'+str(min_edge[2])+')'
    T = [min_edge] # tree
    vertices_in_T = [min_edge[0], min_edge[1]]

    for i in range(2,num_v):
        target_edge = None
        for edge in graph:
            if (edge[0] in vertices_in_T) != (edge[1] in vertices_in_T):
                target_edge = edge
                break
            #cannot find
        if target_edge is None:
            plot_prim(graph)
            exit('Graph is disconnected')
        T.append(target_edge)
        target_edge[3] = 'r' #color edge red
        target_edge[4] = str(i) + ' (w:'+str(target_edge[2]) +')' #relabel edge {{step discovered}} (w:{{weight}})
        vertices_in_T.append(target_edge[0])
        vertices_in_T.append(target_edge[1]) #vertices_in T can have duplicate vertices
    plot_prim(graph)
    return T




def plot_dijk(distance, matrix):
    network = nx.Graph()
    label_dic = {}
    for i in range(len(matrix)):
        for j in range(len(matrix)):
            if matrix[i][j] != 0 :
                network.add_edge(str(i), str(j),weight= matrix[i][j])
    pos = nx.spring_layout(network)

    for i in range (len(distance)):
        label_dic[str(i)] = str(i) + '(distance: ' + str(distance[i])+ ')' # label the vertices {{vertice label}} (distance:{{distance}})

    nx.draw(network,pos=pos, labels=label_dic, with_labels=True)
    labels = nx.get_edge_attributes(network, 'weight')
    nx.draw_networkx_edge_labels(network, pos, edge_labels=labels)
    plt.show()

# Represent G and its weight function as a matrix A: the (i, j)-entry of the
# matrix is the weight of edge ij if ij is in E(G), and 0 otherwise.
#matrix is a represent a simple graph with non-negative weights
# u is a vertex in the graph
# graph should be connected
def dijkstra(matrix, u):
    distance = []
    for i in range(len(matrix)):
        if i == u:
            distance.append(0)
        else:
            distance.append(math.inf) # represents infinity

    s = []
    for i in range(len(matrix)):
        min_elem = math.inf #assume all weights are non-negative
        v = None
        index_v = None
        for k in range(len(distance)):
            if (not (k in s)) and (distance[k] < min_elem):
                min_elem = distance[k]
                v = k
        if v is None:
            exit("disjoint graph");

        s.append(v)
        for b in range(len(distance)):
            if (not b in s) and matrix[b][v] !=0 and (distance[b] > distance[v] + matrix[b][v]):
                distance[b] = distance[v] + matrix[b][v]
    plot_dijk(distance, matrix)
    return distance ## index-i represents node i in this list


def plot_nnh(matrix, path):
    network = nx.Graph()

    for i in range(len(matrix)):
        for j in range(len(matrix)):
            if matrix[i][j] != 0:
                network.add_edge(str(i), str(j), weight=matrix[i][j], path='', color='b')

    for i in range(len(path)-1):
        row = path[i]
        col = path[i+1]
        step = str(i+1) + '(w: ' + str(matrix[row][col]) +')'
        network.add_edge(str(row), str(col), weight=matrix[row][col], path=step, color='r')
    pos = nx.spring_layout(network)
    colors = [network[u][v]['color'] for u, v in network.edges()]
    nx.draw(network, pos, edges=network.edges(), edge_color=colors, width=5)
    labels = nx.get_edge_attributes(network, 'weight')
    nx.draw_networkx_edge_labels(network, pos, edge_labels=labels)
    path = nx.get_edge_attributes(network, 'path')
    nx.draw_networkx_edge_labels(network, pos, edge_labels=path, font_color='red')
    nx.draw_networkx_labels(network, pos, font_size=15, font_family='sans-serif')
    plt.axis('off')

    plt.show()

# note we assume that the graph is a simple undirected graph
# u is vertex in graph
def nearest_neighbour (matrix, u) :
    path = [u]
    y = u
    for j in range (1, len(matrix)):
        min = math.inf
        next_v = None
        for i in range (len(matrix)):
            if matrix[y][i] != 0 and (not i in path) and matrix[y][i] < min:
                min = matrix[y][i]
                next_v = i
        if(min == math.inf):
            exit('no hamilton cycle')
        path.append(next_v)
        y = next_v
    if matrix[y][u] == 0:
        exit('no hamilton cycle')
    path.append(u)
    plot_nnh(matrix, path)
    return path



# graph1 = [['a','b', 2],['b','c', 4],['c','d', 7],['d','e', 11],['c','e', 4],['e','f', 1],['f','g', 4],['g','e', 1],['b','d', 32],['a','f', 2]]
# graph1 = [['a','b', 4],['b','c', 2],['c','d', 1],['d','e', 9],['c','e', 2]]
# graph2 = [
#           [ 0, 0, 0, 0, 2, 0, 0],
#           [ 0, 0, 1, 1, 1, 1, 1],
#           [ 0, 1, 0, 0, 0, 0, 0],
#           [ 0, 1, 0, 0, 5, 0, 0],
#           [ 2, 1, 0, 5, 0, 3, 0],
#           [ 0, 1, 0, 0, 0, 0, 0],
#           [ 0, 1, 0, 0, 3, 0, 0]]
# graph2 = [
#           [ 0, 0, 0, 0, 2],
#           [ 0, 0, 2, 5, 3],
#           [ 0, 2, 0, 0, 0],
#           [ 0, 5, 0, 0, 5],
#           [ 2, 3, 0, 5, 0]]

# graph3 = [
#           [ 0, 5, 2, 2],
#           [ 5, 0, 1, 3],
#           [ 2, 1, 0, 6],
#           [ 2, 3, 6, 0]]
#
graph3 = [
          [ 0, 1, 5, 3],
          [ 1, 0, 1, 3],
          [ 5, 1, 0, 6],
          [ 3, 3, 6, 0]]

# dijkstra(graph2, 0)
# nearest_neighbour(graph3, 0)
# prim(graph1, 7)