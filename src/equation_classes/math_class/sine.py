from src.equation_classes.math_class.abstract_operator import AbstractOperator
import numpy as np


class Sine(AbstractOperator):
    def __init__(self, node):
        super().__init__(node)
        self.num_child = 1
        self.node = node
        self.invertible = False
        self.neutral_element = 0

    def prefix_notation(self, call_node_id, kwargs):
        if call_node_id == self.node.node_id:
            return self.node.parent_node.math_class.prefix_notation(
                call_node_id=self.node.node_id, kwargs=kwargs)
        elif call_node_id == self.node.parent_node.node_id or call_node_id is None:
            return f' sin {self.node.list_children[0].math_class.prefix_notation(self.node.node_id, kwargs)}'
        elif call_node_id == self.node.list_children[0].node_id:
            raise AssertionError('Sine is not invertible so this part should never be called')

    def infix_notation(self, call_node_id, kwargs):
        if call_node_id == self.node.node_id:
            return self.node.parent_node.math_class.infix_notation(call_node_id=self.node.node_id, kwargs=kwargs)
        elif call_node_id == self.node.parent_node.node_id or call_node_id is None:
            return f' sin ( {self.node.list_children[0].math_class.infix_notation(self.node.node_id, kwargs)} )'
        elif call_node_id == self.node.list_children[0].node_id:
            raise AssertionError('Sine is not invertible so this part should never be called')

    def residual(self, call_node_id, dataset, kwargs):
        if call_node_id == self.node.node_id:
            p = self.node.parent_node.math_class.residual(self.node.node_id, dataset, kwargs)
            return p
        elif call_node_id == self.node.list_children[0].node_id:
            raise AssertionError('Sine is not invertible so this part should never be called '
                                 f'{self.node.tree.print()}')

    def evaluate_subtree(self, call_node_id, dataset, kwargs):
        c_0 = self.node.list_children[0].math_class.evaluate_subtree(self.node.node_id, dataset, kwargs)
        return np.sin(c_0, dtype=np.float64)

    def delete(self):
        pass

    def __str__(self):
        return 'sin'
