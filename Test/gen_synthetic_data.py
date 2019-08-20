#from graph_pair import GraphPair
import networkx as nx
from random import sample, choice
import random
import sys
from config import FLAGS


#def gen_synthetic_mcs(gs):
#    for g in gs:
#        assert type(g) is nx.Graph
#    rtn = []
#    for g in gs:
#        g_fake = g
#        rtn.append(GraphPair(g, g_fake, [{0: 0, 2: 1, 1: 2}]))
#    return rtn

#Object for each fake graph that contains all relevant data structures
class FakeMCSResult:
    def __init__(self, original_graph, fake_graph, mcs_g, op, node_mapping):
        self.original_graph = original_graph
        self.fake_graph = fake_graph
        self.mcs = mcs_g
        self.op = op
        self.node_mapping = node_mapping

def _change_nodes(new_g, op, op_times):
    original_type_dict = {}  # can only change to a new type dif with original type, to prevent type changing back and forth for same node.
    count = 0
    mcs = new_g.copy()
    nodes = [int(n) for n in new_g.nodes()]
    mapping = {i:i for i in nodes}
    while count < op_times:
        nodes = new_g.nodes()
        if len(nodes) > 0:
            new_g, mcs, to_change, mapping = _change_node(new_g, mcs, original_type_dict, mapping)
            count += 1
        else:
            raise AssertionError("********* Fail! Graph is empty, no node can be changed! *********")
    op += '_' + str(count)
    return new_g, mcs, op, mapping


def _change_node(new_g, mcs, original_type_dict, mapping):
    if not FLAGS.node_feat_name:
        return new_g, mcs, 'isomorphic', mapping
    new_g_temp = new_g.copy()
    nodes = new_g.nodes()
    is_connected = False
    while not is_connected:
        to_change = choice(list(nodes))
        old_types = nx.get_node_attributes(new_g, FLAGS.node_feat_name)
        old_type = old_types[to_change]
        new_type = _set_node_to_random_type(new_g, to_change)
        if to_change not in original_type_dict.keys():
            original_type_dict[to_change] = old_type
        if old_type != new_type and new_type != original_type_dict[to_change]:
            assert (len(new_g.nodes()) == len(nodes))
            if to_change not in mcs.nodes() and nx.is_connected(mcs):
                is_connected = True
                print("change node id " + str(to_change) + " from old type " + old_type + " to new type " + new_type + ", not in mcs")
            else:
                mcs_temp = mcs.copy()
                mcs_temp.remove_node(to_change)
                if nx.is_connected(mcs_temp):
                    mcs = mcs_temp
                    is_connected = True
                    del mapping[int(to_change)]
                    print("change node id " + str(to_change) + " from old type " + old_type + " to new type " + new_type + ", in mcs and deleted")
                else:
                    new_g = new_g_temp.copy()
        else:
            new_g = new_g_temp.copy()
    return new_g, mcs, to_change, mapping

#sanity check to make sure the original graph has at least one node, it is connected
def _sanity_check(g):
    num_nodes = g.number_of_nodes()
    if num_nodes < 1:
        raise RuntimeError('Wrong graph {} with {} nodes < 1'.format(g.graph['gid'], num_nodes))
#    if num_nodes > FLAGS.max_nodes:
#        raise RuntimeError('Wrong graph {} with {} nodes > max nodes'.format(
#                                                                             g.graph['gid'], num_nodes))
    if num_nodes >= 2 and len(list(nx.isolates(g))) != 0:
        print('Wrong graph {} with {} isolated nodes'.format(g.graph['gid'], len(nx.isolates(g))))

def _del_edges(new_g, op, op_times):
    count = 0
    mcs = new_g.copy()
    nodes = [int(n) for n in new_g.nodes()]
    mapping = {i:i for i in nodes}
    while count < op_times:
        edges = new_g.edges()
        if len(edges) > 0:
            new_g, mcs, mapping = _del_edge(new_g, mcs, mapping)
            count += 1
        else:
            raise AssertionError("********* Fail! Graph has no edge, no edge can be deleted! *********")
    op += '_' + str(count)
    return new_g, mcs, op, mapping


def _del_edge(new_g, mcs, mapping):
    is_connected = False
    nodes = new_g.nodes()
    edges = len(new_g.edges())
    while not is_connected:
        i, j = sample(nodes, 2)
        assert (i != j)
        if new_g.has_edge(i, j):
            new_g.remove_edge(i, j)
            assert (len(new_g.edges()) == edges - 1)
            if i not in mcs.nodes() and nx.is_connected(mcs):
                is_connected = True
            elif j not in mcs.nodes() and nx.is_connected(mcs):
                is_connected = True
            else:  # need to try to delete one node (i or j) from mcs
                mcs1 = mcs.copy()
                mcs1.remove_node(i)
                mcs2 = mcs.copy()
                mcs2.remove_node(j)
                if nx.is_connected(mcs1):
                    mcs = mcs1
                    is_connected = True
                    del mapping[int(i)]
                elif nx.is_connected(mcs2):
                    mcs = mcs2
                    is_connected = True
                    del mapping[int(j)]
                else:  # deleting i or j will make mcs unconnected, need to sample another two nodes to add edge
                    new_g.add_edge(i, j)
    return new_g, mcs, mapping


def _del_nodes(new_g, op, op_times):
    count = 0
    nodes = [int(n) for n in new_g.nodes()]
    mapping = {i:i for i in nodes}
    while count < op_times:
        nodes = new_g.nodes()
        if len(nodes) > 0:
            new_g, mapping = _del_node(new_g, mapping)
            count += 1
        else:
            raise AssertionError("********* Fail! Graph is empty, no node can be deleted! *********")
    op += '_' + str(count)
    mcs = new_g
    return new_g, mcs, op, mapping


def _del_node(new_g, mapping):
    is_connected = False
    new_g_temp = new_g.copy()
    nodes = new_g.nodes()
    edges = new_g.edges()
    while not is_connected:
        to_del = str(choice(list(nodes)))
        new_g.remove_node(to_del)  # Removes the node and all adjacent edges
        if len(edges) == 0 or nx.is_connected(new_g):
            is_connected = True
            del mapping[int(to_del)]
        else:
            new_g = new_g_temp.copy()
    return new_g, mapping

def _add_edges(new_g, op, op_times):
    count = 0
    mcs = new_g.copy()
    nodes = [int(n) for n in new_g.nodes()]
    mapping = {i:i for i in nodes}
    while count < op_times:
        nodes = new_g.nodes()
        if len(nodes) >= 2:
            new_g, mcs, mapping = _add_edge(new_g, mcs, mapping)
            count += 1
        else:
            raise AssertionError("********* Fail! Graph has < 2 nodes, no edge can be added! *********")
    op += '_' + str(count)
    return new_g, mcs, op, mapping


def _add_edge(new_g, mcs, mapping):
    is_connected = False
    nodes = new_g.nodes()
    edges = len(new_g.edges())
    cnt = 0
    while not is_connected:
        
        if cnt >= 10:
            return new_g, mcs
        print(1)
        
        i, j = sample(nodes, 2)
        assert (i != j)
        
        if not new_g.has_edge(i, j):
            print(2)
            new_g.add_edge(i, j)
            assert (len(new_g.edges()) == edges + 1)
            if i not in mcs.nodes() and nx.is_connected(mcs):
                print(3)
                is_connected = True
            elif j not in mcs.nodes() and nx.is_connected(mcs):
                print(4)
                is_connected = True
            else:  # need to try to delete one node (i or j) from mcs
                print(5)
                mcs1 = mcs.copy()
                mcs1.remove_node(i)
                mcs2 = mcs.copy()
                mcs2.remove_node(j)
                if nx.is_connected(mcs1):
                    print(6)
                    mcs = mcs1
                    is_connected = True
                    del mapping[int(i)]
                elif nx.is_connected(mcs2):
                    print(7)
                    mcs = mcs2
                    is_connected = True
                    del mapping[int(j)]
                else:  # deleting i or j will make mcs unconnected, need to sample another two nodes to add edge
                    print(8)
                    new_g.remove_edge(i, j)
        else:
            cnt += 1
    print(9)
    return new_g, mcs, mapping


def _add_nodes(new_g, op, op_times):
    mcs = new_g.copy()
    nodes = [int(n) for n in new_g.nodes()]
    mapping = {i:i for i in nodes}
    count = 0
    for t in range(0, op_times):
        new_g = _add_node(new_g)
        count += 1
    op += '_' + str(count)
    return new_g, mcs, op, mapping

#given a graph choose a random node to connect the new node to
def _add_node(new_g):
    nodes = new_g.nodes()
    to_connect = choice(list(nodes))
    new_g = _add_node_with_an_edge(new_g, nodes, to_connect)
    return new_g

#add the new node with an edge and give the node a type
def _add_node_with_an_edge(graph, nodes, existing_node):
    nodes = [int(n) for n in nodes]
    new_node = str(max(nodes) + 1)
    graph.add_edge(str(existing_node), new_node)
    if FLAGS.node_feat_name:
        _set_node_to_random_type(graph, new_node)
    assert(len(graph.nodes) == len(nodes) + 1)
    return graph

#give a node a random type
def _set_node_to_random_type(g, node):
    exsiting_types = list(nx.get_node_attributes(g, FLAGS.node_feat_name).values())
    type = choice(exsiting_types)
    nx.set_node_attributes(g, {node: type}, FLAGS.node_feat_name)
    return type


def perturb_graph_with_same_opts(graph, gid, op, op_times):
    
    new_g = graph.copy()
    _sanity_check(graph)
    
    mcs = None
    mapping = {}
    num_nodes = len(new_g.nodes())
    num_edges = len(new_g.edges())
    
    if op == '':
        op = choice(['add_nodes', 'add_edges', 'del_nodes', 'del_edges', 'change_nodes',
                     'isomorphic'])
    
    #the mapping is determined before the new graph is even made because adding a node will not change the mapping!
    if op == 'add_nodes':
        print('add_nodes')
        if op_times == 0:
            op_times = random.randint(1, num_nodes)
        print("op_times")
        print(op_times)
        new_g, mcs, op, mapping = _add_nodes(new_g, op, op_times)
    #the mapping is determined before new graph is made and then when we delete a node from the graph that mapping is removed!
    if op == 'add_edges':
        print('add_edges')
        num_edges_ava = num_nodes * (num_nodes - 1) / 2 - num_edges
        if op_times == 0:
            op_times = random.randint(1, num_edges_ava)
        print("op_times")
        print(op_times)
        new_g, mcs, op, mapping = _add_edges(new_g, op, op_times)
    
    #the mapping is determined before the new graph is made and then when we delete a node we remove the mapping!
    if op == 'del_nodes':
        print('del_nodes')
        if op_times == 0:
            op_times = random.randint(1, num_nodes - 1)
        print("op_times")
        print(op_times)
        new_g, mcs, op, mapping = _del_nodes(new_g, op, op_times)

    #the mapping is determined before the new graph is made and then when we delete a node we remove the mapping!
    if op == 'del_edges':
        print('del_edges')
        if op_times == 0:
            op_times = random.randint(1, num_edges)
        print("op_times")
        print(op_times)
        new_g, mcs, op, mapping = _del_edges(new_g, op, op_times)

    #the mapping is determined before the new graph is made and then when we change a node we remove the mapping!
    if op == 'change_nodes':
        print('change_nodes')
        if op_times == 0:
            op_times = random.randint(1, num_nodes)
        print("op_times")
        print(op_times)
        new_g, mcs, op, mapping = _change_nodes(new_g, op, op_times)
    
    #mapping is simple here
    if op == 'isomorphic':
        print('isomorphic')
        new_g = graph.copy()  # new_g may have been messed up, so re-copy
        mcs = new_g
        nodes = [int(n) for n in new_g.nodes()]
        mapping = {i:i for i in nodes}

    new_g.graph['gid'] = 'fake_' + gid + '_' + str(op)
    _sanity_check(graph)
    _sanity_check(new_g)
    return new_g, mcs, op, mapping

#print to console the number of fake graphs we are generating
#then call the perturb_graph_with_same_opts function
def graph_generator_mcs(nxgraph, op, op_times, num_fake_graphs):
    op_original = op
    if num_fake_graphs == 0:
        num_fake_graphs = random.randint(1, 10)
    
    fake_result_list = [] #will be a list of FakeMCSResult objects

    print("Generating " + str(num_fake_graphs) + " fake graphs from original graph - " + str(nxgraph.graph['gid']))
    for index in range(0, num_fake_graphs):
        op = op_original
        print("Generating " + str(index) + " ......")
        fake_id = str(nxgraph.graph['gid']) + '_' + str(index)
        new_g, mcs, op, mapping = perturb_graph_with_same_opts(nxgraph, fake_id, op, op_times)
        result = FakeMCSResult(nxgraph, new_g, mcs, op, mapping)
        fake_result_list.append(result)
        print(op + " DONE!")
    return fake_result_list


#loads the training and testing datasets and their labels
def my_load_data(ds_name):
    import sys
    from os.path import dirname, abspath
    cur_folder = dirname(abspath(__file__))
    sys.path.insert(0, '{}/../../src'.format(cur_folder))
    from utils import load_data
    train_gs = load_data(ds_name, train=True)
    test_gs = load_data(ds_name, train=False)
    train_labels = []
    test_labels = []
    for g_train in train_gs.graphs:
        train_labels.append(g_train.graph["glabel"])
    for g_test in test_gs.graphs:
        test_labels.append(g_test.graph["glabel"])
    return train_gs.graphs, test_gs.graphs, train_labels, test_labels

#draws the graph and saves it into the output directory as png file
def draw_graph(graph, save_path):
    import matplotlib.pyplot as plt
    graph_pos = nx.shell_layout(graph)
    nx.draw_networkx_nodes(graph, graph_pos, node_size=1000, node_color='blue', alpha=0.3)
    nx.draw_networkx_edges(graph, graph_pos)
    nx.draw_networkx_labels(graph, graph_pos, font_size=12, font_family='sans-serif')
    plt.savefig(save_path)
    plt.close()

if __name__ == "__main__":
    DATA_DIR = "datasets/"
    OUTPUT_DIR = "FakeGenPlots/" #where the resulting graphs will be drawn
    ds_name = 'aids80nef'  # dataset name
    
    op = sys.argv[1]  #specified operation
    op_times = int(sys.argv[2]) #how many times to perform operation
    num_fake_graphs = int(sys.argv[3]) #how many fake graphs to generate
    
    #load the test and train graphs and labels
    train_graphs, test_graphs, train_labels, test_labels = my_load_data(ds_name)
    graphs_for_test = train_graphs[0:1] #for testing purposes only look at the first training graph
    results = [] #this will be a list of FakeMCSResult objects
    
    for g in graphs_for_test:
        result = graph_generator_mcs(g, op, op_times, num_fake_graphs)
        print("=============" + "original graph" + "===============")
        print("nodes")
        print(g.nodes())
        print("edges")
        print(g.edges())
        draw_index = 0
         # draw original_graph
        original_id = str(g.graph['gid'])
        draw_graph(g, OUTPUT_DIR+original_id+'_original.png')
        for fakeGraph in result:
            draw_graph(fakeGraph.fake_graph, OUTPUT_DIR+original_id+'_'+str(draw_index)+fakeGraph.op+'_fake.png')
            draw_graph(fakeGraph.mcs, OUTPUT_DIR+original_id+'_'+str(draw_index)+fakeGraph.op+'_mcs.png')
            draw_index += 1
        index = 0
        for fakeGraph in result:
            print("=============" + "fake graph " + str(index) + "===============")
            print("nodes")
            print(fakeGraph.fake_graph.nodes())
            print("edges")
            print(fakeGraph.fake_graph.edges())
            print("=============" + "mcs graph " + str(index) + "===============")
            print("nodes")
            print(fakeGraph.mcs.nodes())
            print("edges")
            print(fakeGraph.mcs.edges())
            print("=============" + "mapping" + str(index) + "===============")
            print(fakeGraph.node_mapping)
            index += 1
            results.append(result)
    print(len(results))
