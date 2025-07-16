import traceback

import numpy as np
def get_list_residual_nodes(tree, last_changed_node_id):
    tree.start_node.count_inner_nodes()
    no_residuals = get_no_residual_nodes(last_changed_node_id, tree)
    nodes_per_layer = get_dict_nodes_per_layer(tree)
    list_residual_nodes = sort_nodes_per_layer(nodes_per_layer)
    delete_no_residual(list_residual_nodes, no_residuals)
    return list_residual_nodes


def delete_no_residual(list_residual_nodes, no_residuals):
    for node_id in no_residuals:
        try:
            list_residual_nodes.remove(node_id)
        except:
            pass


def sort_nodes_per_layer(nodes_per_layer):
    list_residual_nodes = []
    for depth, layer in nodes_per_layer.items():
        if len(layer['num_child']) > 0:
            sort_index = np.argsort(layer['num_child'])
            list_residual_nodes += list(np.array(layer['id'])[sort_index])
    return list_residual_nodes


def get_dict_nodes_per_layer(tree):
    nodes_per_layer = {}
    for i in range(tree.current_depth + 1):
        nodes_per_layer[i] = {
            'id': [],
            'num_child': []
        }
    for node_id, node in tree.dict_of_nodes.items():
        if node_id == -1 or node_id == 0:
            continue
        nodes_per_layer[node.depth]['id'].append(node_id)
        nodes_per_layer[node.depth]['num_child'].append(
            node.num_child_inner_nodes
        )
    return nodes_per_layer


def get_no_residual_nodes(last_changed_node_id, tree):
    try:
        last_changed_node = tree.dict_of_nodes[last_changed_node_id]
    except Exception:
        print(traceback.format_exc())
        print('last changed tree')
        tree.print()
        return []
    no_residuals = [last_changed_node.node_id]
    while last_changed_node.parent_node:
        no_residuals.append(last_changed_node.parent_node.node_id)
        last_changed_node = last_changed_node.parent_node
    return no_residuals