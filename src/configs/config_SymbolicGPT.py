from argparse import ArgumentParser
from src.utils.argument_parser import str2bool
import numpy as np
class ConfigSymbolicGPT:
    @staticmethod
    def arguments_parser(parser=None) -> ArgumentParser:
        if not parser:
            parser = ArgumentParser(description="Parser for SymbolicGPT")

        return parser

