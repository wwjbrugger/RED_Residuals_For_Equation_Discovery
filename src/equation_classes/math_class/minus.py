import numpy as np
from src.equation_classes.math_class.abstract_operator import AbstractOperator

class Minus(AbstractOperator):
    def __init__(self, node):
        super().__init__(node)
        self.num_child = 2
        self.node = node
        self.invertible = True
        self.neutral_element = 0

    def prefix_notation(self, call_node_id, kwargs):
        if call_node_id == self.node.node_id:
            return self.node.parent_node.math_class.prefix_notation(
                call_node_id=self.node.node_id, kwargs=kwargs)

        elif call_node_id == self.node.parent_node.node_id or call_node_id is None:
            str_0 = self.node.list_children[0].math_class.prefix_notation(
                self.node.node_id,kwargs
            )
            str_1 = self.node.list_children[1].math_class.prefix_notation(
                self.node.node_id, kwargs)
            return f" - {str_0} {str_1}"

        elif call_node_id == self.node.list_children[0].node_id:
            str_0 = self.node.parent_node.math_class.prefix_notation(
                self.node.node_id, kwargs)
            str_1 = self.node.list_children[1].math_class.prefix_notation(
                self.node.node_id, kwargs)
            return f' + {str_0} {str_1} '

        elif call_node_id == self.node.list_children[1].node_id:
            str_0 = self.node.list_children[0].math_class.prefix_notation(
                self.node.node_id, kwargs)
            str_1 = self.node.parent_node.math_class.prefix_notation(
                self.node.node_id,kwargs)
            return f'- {str_0} {str_1}'

    def infix_notation(self, call_node_id, kwargs):
        if call_node_id == self.node.node_id:
            return self.node.parent_node.math_class.infix_notation(call_node_id=self.node.node_id, kwargs=kwargs)
        elif call_node_id == self.node.parent_node.node_id or call_node_id is None:
            return f'( {self.node.list_children[0].math_class.infix_notation(self.node.node_id, kwargs)} -' \
                   f' {self.node.list_children[1].math_class.infix_notation(self.node.node_id, kwargs)} ) '
        elif call_node_id == self.node.list_children[0].node_id:
            return f'( {self.node.parent_node.math_class.infix_notation(self.node.node_id, kwargs)} +' \
                   f' {self.node.list_children[1].math_class.infix_notation(self.node.node_id, kwargs)} ) '
        elif call_node_id == self.node.list_children[1].node_id:
            return f'( {self.node.list_children[0].math_class.infix_notation(self.node.node_id, kwargs)} -' \
                   f' {self.node.parent_node.math_class.infix_notation(self.node.node_id, kwargs)} ) '

    def residual(self, call_node_id, dataset, kwargs):
        if call_node_id == self.node.list_children[0].node_id:
            p = self.node.parent_node.math_class.residual(self.node.node_id, dataset, kwargs)
            c_1 = self.node.list_children[1].math_class.evaluate_subtree(self.node.node_id, dataset, kwargs)
            return np.add(p,c_1, dtype=np.float64)
        elif call_node_id == self.node.list_children[1].node_id:
            p = self.node.parent_node.math_class.residual(self.node.node_id, dataset, kwargs)
            c_0 = self.node.list_children[0].math_class.evaluate_subtree(self.node.node_id, dataset, kwargs)
            return np.subtract(c_0,p, dtype=np.float64)
        elif call_node_id == self.node.node_id:
            p = self.node.parent_node.math_class.residual(self.node.node_id, dataset, kwargs)
            return p

    def evaluate_subtree(self, call_node_id, dataset, kwargs):
        c_0 = self.node.list_children[0].math_class.evaluate_subtree(self.node.node_id, dataset, kwargs)
        c_1 = self.node.list_children[1].math_class.evaluate_subtree(self.node.node_id, dataset, kwargs)
        return np.subtract(c_0, c_1, dtype=np.float64)

    def delete(self):
        pass

    def __str__(self):
        return '-'
