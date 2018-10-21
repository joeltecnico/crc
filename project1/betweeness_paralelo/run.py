from multiprocessing import Pool
import time

import matplotlib.pyplot as plt
import networkx as nx

import defs

def betweenness_centrality_parallel(G, processes=None):
    """Parallel betweenness centrality  function"""
    p = Pool(processes=processes)
    node_divisor = len(p._pool) * 4
    node_chunks = list(defs.chunks(G.nodes(), int(G.order() / node_divisor)))
    num_chunks = len(node_chunks)
    bt_sc = p.map(defs._betmap,
                  zip([G] * num_chunks,
                      [True] * num_chunks,
                      [None] * num_chunks,
                      node_chunks))
    
    # Reduce the partial solutions
    bt_c = bt_sc[0]
    for bt in bt_sc[1:]:
        for n in bt:
            bt_c[n] += bt[n]
    return bt_c

if __name__ == "__main__":
    with open('facebook_combined.csv', 'rb') as inf:
        next(inf, '')   # skip a line
        G = nx.read_edgelist(inf, delimiter=',', nodetype=int, encoding="utf-8")
        N = G.number_of_nodes()
        E = G.number_of_edges()
        
        n_triangles = sum(nx.triangles(G).values()) / 3
        
        print(n_triangles)
        
        print("\tParallel version")
        start = time.time()
        bt = betweenness_centrality_parallel(G)
        print("\t\tTime: %.4F" % (time.time() - start))
        print("\t\tBetweenness centrality for node 0: %.5f" % (bt[0]))
        print("\tNon-Parallel version")
        start = time.time()
        bt = nx.betweenness_centrality(G)
        print("\t\tTime: %.4F seconds" % (time.time() - start))
        print("\t\tBetweenness centrality for node 0: %.5f" % (bt[0]))
        
        nx.draw(G)
        plt.show()