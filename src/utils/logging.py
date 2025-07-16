import logging



class CustomFormatter(logging.Formatter):

    grey = "\x1b[37;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s "

    FORMATS = {
        logging.DEBUG: grey + format + reset,
        logging.INFO: grey + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt, datefmt='%H:%M:%S')
        return formatter.format(record)


def get_log_obj(args, name="AlphaZeroEquation"):
    logger = logging.getLogger(name)
    logger.setLevel(args.logging_level)

    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(args.logging_level)

    ch.setFormatter(CustomFormatter())

    logger.addHandler(ch)
    logger.propagate = False
    return logger

def log_classic_experiments(best_equation_prefix, output):
    logging.debug(f"Best equation found: {output['infix']}")
    logging.debug(f"Best equation found: {best_equation_prefix:<20}"
                  f" with error:{output['error']}  ")

def logging_residuals(rep_dict,output_res):
    logging.debug(f"Calculate residual for: {rep_dict['eq_placeholder']}"
                  f" by searching X = {rep_dict['residual_equation']}")
    logging.debug(f"Old X: {rep_dict['old_X']}")
    logging.debug(f"New X = {rep_dict['new_X']}")
    logging.debug(f"Best equation using residuals: {rep_dict['complete_res_eq_sympy']} with error:{output_res['error']}  ")

def logging_gp(rep_dict,output_res):
    logging.debug(f"Calculate gp for: {rep_dict['eq_placeholder']}")
    logging.debug(f"New Equation: {rep_dict['gp_infix_sympy']} with error:{output_res['error']} ")

