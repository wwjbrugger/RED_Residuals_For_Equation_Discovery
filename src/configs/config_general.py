from argparse import ArgumentParser
from src.utils.argument_parser import str2bool
from definitions import ROOT_DIR
import time
class ConfigGeneral():
    @staticmethod
    def arguments_parser(parser = None) -> ArgumentParser:
        if not parser:
            parser = ArgumentParser(description="Parser for General setting")
        parser.add_argument("--exp_name", type=str, default="",
                            help="Path to project")

        parser.add_argument("--path_to_old_results", type=str, default="",
                            help="Path to old results to finish them")

        parser.add_argument("--seed", type=int,
                            default=1,
                            help='Seed for expeiment')
        parser.add_argument("--logging_level", type=int, default=30,
                            help="CRITICAL = 50, ERROR = 40, "
                                 "WARNING = 30, INFO = 20, "
                                 "DEBUG = 10, NOTSET = 0")
        parser.add_argument("--root_dir", type=str, default=ROOT_DIR,
                            help="Path to project")
        parser.add_argument("--max_num_residuals", type=int, default=10,
                            help="How many residuals are maximal calculated for"
                                 "one equation")
        parser.add_argument("--noise_factor", type=float, default=0,
                            help="How much noise is added to each column in the frame")
        parser.add_argument("--dataset_folder", type=str, default='datasets_srbench',# 'datasets_srbench',# 'datasets_dso' 'datasets_dso_1000'
                            help="Where to load the datasets")
        parser.add_argument("--max_dataset_size", type=int, default=300,
                            help="How many rows are sampled from dataset. Test, Val and train split follows afterwards. ")
        parser.add_argument("--only_classic", type= str2bool, default= False,
                            help="If only the classic approach should run")
        
        return parser