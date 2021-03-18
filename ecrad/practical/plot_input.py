#!/usr/bin/env python3 

def warn(*args, **kwargs):
    pass
    
import os, warnings
warnings.warn = warn

from ecradplot import plot as eplt

def main(input_srcfile, dstdir, mode, include_t):
    """
    Plot input files
    """
    
    if not os.path.isdir(dstdir):
        os.makedirs(dstdir)

    #Get input file name
    name_string      = os.path.splitext(os.path.basename(input_srcfile))[0]
    
    dstfile = os.path.join(dstdir, name_string + ".png")
    
    print(f"Plotting inputs to {dstfile}")
    if mode == 'all':
        eplt.plot_inputs(input_srcfile, dstfile=dstfile)
    elif mode == 'scalars':
        eplt.plot_inputs_scalars(input_srcfile, include_t=include_t, dstfile=dstfile)
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Plot surface properties, atmospheric composition and clouds from input file to ecRAD.")
    parser.add_argument("input",    help="ecRAD input file")
    parser.add_argument("--dstdir", help="Destination directory for plots", default="./")
    parser.add_argument("--mode", choices=['all', 'scalars'], default='all')
    parser.add_argument("--include-t", action='store_true')
    args = parser.parse_args()
    
    main(args.input, args.dstdir, args.mode, args.include_t)
