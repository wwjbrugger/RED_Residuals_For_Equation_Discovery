from argparse import ArgumentParser
from src.utils.argument_parser import str2bool
import numpy as np
class ConfigE2E:
    @staticmethod
    def arguments_parser(parser=None) -> ArgumentParser:
        if not parser:
            parser = ArgumentParser(description="Parser for E2E")
        parser.add_argument('--max_input_points', type=int,
                            default=200,
                            help='')
        parser.add_argument('--n_trees_to_refine', type=int,
                            default=100,
                            help='')
        parser.add_argument('--rescale', type=str2bool,
                                    default=True,
                                    help='')

        return parser
