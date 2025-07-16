from argparse import ArgumentParser
from src.utils.argument_parser import str2bool
import numpy as np
class ConfigPySR:
    @staticmethod
    def arguments_parser(parser=None) -> ArgumentParser:
        if not parser:
            parser = ArgumentParser(description="Parser for PySR")
        parser.add_argument("--pysr_niterations", type=int,
                            default=10,
                            help='Seed for expeiment')

        return parser

