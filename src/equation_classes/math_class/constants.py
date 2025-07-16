import numpy as np
from src.equation_classes.math_class.abstract_operator import AbstractOperator


class Constants(AbstractOperator):
    def __init__(self, node):
        super(Constants, self).__init__(node)
        self.num_child = 0
        self.node = node
        self.invertible = True
        self.node.node_symbol = f"c_{self.node.tree.num_constants_in_complete_tree}"
        self.node.tree.constants_in_tree[self.node.node_symbol] = {
            'node_id': self.node.node_id,
            'value': None
        }
        self.node.tree.num_constants_in_complete_tree += 1
        if self.node.tree.num_constants_in_complete_tree > self.node.tree.args.max_constants_in_tree:
            self.node.tree.max_constants_reached = True

    def prefix_notation(self, call_node_id, kwargs):
        if call_node_id == self.node.node_id:
            return self.node.parent_node.math_class.prefix_notation(
                call_node_id=self.node.node_id, kwargs=kwargs)
        elif len(list(kwargs.keys())):
            # constant dict is given
            return f"{kwargs[self.node.node_symbol]['value']}"
        else:
            return 'c'

    def infix_notation(self, call_node_id, kwargs):
        if call_node_id == self.node.node_id:
            return self.node.parent_node.math_class.infix_notation(call_node_id=self.node.node_id, kwargs=kwargs)
        elif len(list(kwargs.keys())):
            return f"{kwargs[self.node.node_symbol]['value']:.4f}"
        else:
            return f"{self.node.node_symbol}"

    def residual(self, call_node_id, dataset, kwargs):
        return self.node.parent_node.math_class.residual(
            call_node_id=self.node.node_id, dataset=dataset, kwargs=kwargs
        )

    def evaluate_subtree(self, call_node_id, dataset, kwargs):
        symbol = self.node.node_symbol
        if 'c_0' in kwargs:
            return np.full(
                shape=dataset.shape[0],
                fill_value=kwargs[symbol]['value'],
                dtype=np.float64
            )
        else:
            system_id_column = self.node.tree.args.system_id_column
            replace_dict = dict(
                [(key, kwargs[key][symbol]['value']) for key in kwargs.keys()
                 if isinstance(kwargs[key], dict)]
            )
            words_array = dataset[system_id_column].to_numpy()

            replaced_array = np.vectorize(
                lambda x: replace_dict.get(x, kwargs['average'][symbol]['value']))(words_array)
            return replaced_array

    def delete(self):
        self.node.tree.num_constants_in_complete_tree -= 1

    def __str__(self):
        return 'c'
