import networkx as nx
import dgl
import torch as th

# g_nx = nx.petersen_graph()
# g_dgl = dgl.DGLGraph(g_nx)

import matplotlib.pyplot as plt
# plt.subplot(121)
# nx.draw(g_nx, with_labels=True)
# plt.subplot(122)
# nx.draw(g_dgl.to_networkx(), with_labels=True)

# plt.show()
g = dgl.DGLGraph()
g.add_nodes(10)
# A couple edges one-by-one
for i in range(1, 4):
    g.add_edge(i, 0)
# A few more with a paired list
src = list(range(5, 8)); dst = [1]*3
g.add_edges(src, dst)
# finish with a pair of tensors
src = th.tensor([8, 9]); dst = th.tensor([0, 0])
g.add_edges(src, dst)

nx.draw(g.to_networkx(), with_labels=True)
plt.show()

