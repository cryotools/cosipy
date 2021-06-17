def read_opt(opt_dict, glob):
    """ Reads the opt_dict and overwrites the key-value pairs in glob - the calling function's
    globals() dictionary."""
    if opt_dict is not None:
        for key in opt_dict:
            glob[key] = opt_dict[key]
