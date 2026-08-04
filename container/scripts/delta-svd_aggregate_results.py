#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Aggregate DELTA-SVD result tables across subjects/patients
"""

import sys, os, glob, argparse, re
import numpy as np
import pandas as pd
import datetime

from delta_svd_version import __version__

def isDir(path):
    if not os.path.isdir(path):
        raise argparse.ArgumentTypeError("File path for directory has to be an existing directory. Please check: %s"%(path))
    else:
        return path
    
def extCSV(path):
    if not path.endswith('.csv'):
        raise argparse.ArgumentTypeError("File path for filename has to end with '.csv'. Please check: %s"%(path))
    else:
        return path

def plausiblePath(path):
    if os.path.isdir(path):
        return path
    else:
        directory = os.path.dirname(path)
        base = os.path.basename(path)
        if len(base)>4 and base[-4:]=='.csv' and (len(directory)==0 or os.path.isdir(directory)):
            return path
        else:
            raise argparse.ArgumentTypeError("File path has to be an existing directory or a filename. If filename, it must end with '.csv' and can be optionally prepended by the path to an existing directory. Please check: %s"%(path))

fnOutDefault = "delta-svd_results_aggregated.csv"
license = "https://github.com/isdneuroimaging/DELTA-SVD"

def iniParser():
    parser = argparse.ArgumentParser(description=f"DELTA-SVD {__version__}. Aggregate multiple DELTA-SVD result tables into one table. Files with result tables will be globbed in the specified DIRECTORY using the specified FILENAME and DEPTH.",
                                     add_help=False,
                                     epilog=f'Notice: By using DELTA-SVD, you agree to the license terms (CC BY-NC-ND 4.0) described in the LICENSE file at "{license}"')
    group0 = parser.add_argument_group()
    group0.add_argument(dest="directory", metavar='DIRECTORY', type=isDir, help="path to directory containing DELTA-SVD result files (at any depth). Will be used for globbing.")
    group0.add_argument("-f", dest="filename", type=extCSV, default="delta-svd_results.csv", help="name of DELTA-SVD result files (default: %(default)s). Will be used for globbing. Requires extension '.csv'.")
    group0.add_argument("-d", dest="depth", type=int, default=-1, help="depth for globbing (default: %(default)s, which means any depth). If set to '0', only the top-level directory will be searched, making sense only with wildcards ('*') in filename.")
    group0.add_argument("-o", dest="output", metavar="OUTPUT-PATH", type=plausiblePath, default=fnOutDefault, help="path to write aggregated table to (default: %(default)s). If left at default or only a filename is provided, it will be saved into the DIRECTORY provided for globbing. If only a directory is provided, the default output-filename will be used.")
    group0.add_argument("-s", dest="split", action='store_true', help='split into separate output tables for metrics and debugging information. The output table names will be constructed by appending "_metrics" and "_debugging" respectively.')
    group0.add_argument("-p", dest="insertPath", action='store_const', const=0, default=-1, help="insert a column 'path' with the path names of input CSV files into the aggregated table. By default, this will only be done, if the 'ID' column is missing in the input CSV files.")
    group0.add_argument("-t", dest="appendDate", choices=['date', 'time', 'datetime'], default=None, help="append output filename with current date, time, or datetime (default: %(default)s), formatted as '*[_YYYY-MM-DD][_HHMMSS].csv'")
    group0.add_argument("-x", dest="overwrite", action='store_true', help="allow overwriting output if existing. By default, already existing output will raise an error. (Be careful not to glob previous aggregation files when repeating aggregation.)")
    group0.add_argument("-q", dest="verbose", action='store_false', help="quiet mode")
    group0.add_argument("--version", action="version", version=f"DELTA-SVD {__version__}", help="show the DELTA-SVD version and exit")
    group0.add_argument("-h", action="help", help="show this help message and exit")
    group0.add_argument("-help","--help", action="help", help=argparse.SUPPRESS)
    return parser      


if __name__ == "__main__":

    parser = iniParser()
    if len(sys.argv)==1:
        parser.print_usage()
        print(f'\nDELTA-SVD {__version__}\n'
              f'Run "{os.path.basename(__file__)} -h" for detailed help\n'
              f'Notice: By using DELTA-SVD, you agree to the license terms (CC BY-NC-ND 4.0) described in the LICENSE file at "{license}"\n')
        parser.exit()

    args = parser.parse_args()


    if args.verbose:
        print(f"DELTA-SVD {__version__}")
        print("Running: " + " ".join([os.path.basename(sys.argv[0])]+sys.argv[1::]))
    
    if os.path.isdir(args.output):
        fnOut = os.path.join(args.output, fnOutDefault)
    elif os.path.dirname(args.output)=='':
        fnOut = os.path.join(args.directory, args.output)
    else:
        fnOut = args.output
    
    if args.appendDate:
        if args.appendDate == 'date':
            date = datetime.datetime.now().strftime("%Y-%m-%d")
        elif args.appendDate == 'time':
            date = datetime.datetime.now().strftime("%H%M%S")
        elif args.appendDate == 'datetime': 
            date = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
        fnOut = re.sub(r'\.csv$', f'_{date}.csv', fnOut)

    # split output names, and check none of them exists already
    if args.split:
        fnMetric = re.sub(r'\.csv$', '_metrics.csv', fnOut)
        fnDebug = re.sub(r'\.csv$', '_debugging.csv', fnOut)
        if os.path.isfile(fnMetric):
            if not args.overwrite:
                print(f'\nERROR: Output file already exists:\n {fnMetric}')
                print(" Use option '-x' to overwrite existing file or change output filepath with option '-o' or '-t datetime'.\n")
                sys.exit(1)
            else:
                print(f'\nWARNING: Output file already exists and will be overwritten:\n {fnMetric}')
        if os.path.isfile(fnDebug):
            if not args.overwrite:
                print(f'\nERROR: Output file already exists:\n {fnDebug}')
                print(" Use option '-x' to overwrite existing file or change output filepath with option '-o' or '-t datetime'.\n")
                sys.exit(1) 
            else:
                print(f'\nWARNING: Output file already exists and will be overwritten:\n {fnDebug}')
    else:
        if os.path.isfile(fnOut):
            if not args.overwrite:
                print(f'\nERROR: Output file already exists:\n {fnOut}')
                print(" Use option '-x' to overwrite existing file or change output filepath with option '-o' or '-t datetime'.\n")
                sys.exit(1)
            else:
                print(f'\nWARNING: Output file already exists and will be overwritten:\n {fnOut}')
    
    if args.depth == -1:
        depth = '**'
        depthStr = 'any depth'
    else:
        depth = '/'.join(['*'] * args.depth)
        depthStr = f'search depth = {args.depth}'
    
    if args.verbose: print(f'\nSearching for CSV files in "{args.directory}" at {depthStr} and with pattern "{args.filename}":')
    fnames = sorted(glob.glob(os.path.join(args.directory, depth, args.filename), recursive=True))

    if len(fnames) == 0:
        print(f'\nNo CSV files found in "{args.directory}" at {depthStr} and with pattern "{args.filename}"')
        sys.exit(1)

    if args.verbose:
        print(f'\nFound {len(fnames)} CSV files')

    # Row counts are taken here, not reconstructed from the concatenated index
    # later: a file contributing no rows leaves no trace there and would shift
    # the path assignment of every following file (see "Insert pathnames").
    headers = list()
    dfs = list()
    nRows = list()
    fnamesOk = list()
    for file in fnames:
        try:
            dfT = pd.read_csv(file)
        except Exception:
            print('WARNING: Excluding file due to reading error:', file)
            continue
        if len(dfT) == 0:
            print('WARNING: Excluding file without any data rows:', file)
            continue
        headers.append(list(dfT.columns))
        dfs.append(dfT)
        nRows.append(len(dfT))
        fnamesOk.append(file)
    if len(fnamesOk)>0:
        fnames = fnamesOk
    else:
        print('ERROR: none of the globbed files contains readable data!')
        sys.exit(1)

    headersStr = [f'{header}' for header in headers]
    headersStrUnq = set(headersStr)
    if len(headersStrUnq) > 1:
        print("\nERROR: Not all Headers are the same.")
        HeadersStrUnqCnt = [headersStr.count(uH) for uH in headersStrUnq]
        HeadersStrUnqIdx = [headersStr.index(uH) for uH in headersStrUnq]
        HeadersStrUnqLen = [len(f'{header}') for header in headersStrUnq]
        zipped = sorted(zip(HeadersStrUnqCnt, headersStrUnq, HeadersStrUnqLen, [fnames[i] for i in HeadersStrUnqIdx]),reverse=True)        
        ll = max(HeadersStrUnqLen)
        print("#Files | Header" + ' '*(ll-6+2) + " | Example file")
        for i, row in enumerate(zipped):
            print("{:>6d} | {:s}".format(*row[0:2]) + '-'*(ll-row[2]+2) + " | {:<45s}".format(row[3]))
        print('Aborting! Aggregation of tables with different headers not supported!')
        sys.exit(1)

    df = pd.concat(dfs)

    if not 'ID' in headers[0]:
        # without 'ID' the paths are the only thing telling the rows apart
        if args.insertPath == -1:
            print("\nWARNING: Column 'ID' not found. Inserting a column 'path' with CSV file paths as an alternative identifier.")
        args.insertPath = 0
    elif args.insertPath == 0:
        # insert "path" right after the existing "ID" column
        args.insertPath = list(headers[0]).index("ID") + 1
    else:
        # No 'path' column, so 'ID' alone has to distinguish the rows: constant
        # within each file (which may hold many rows), and unique across files.
        fileGroup = np.repeat(np.arange(len(fnames)), nRows)
        idsPerFile = df.groupby(fileGroup)['ID'].agg(['nunique', 'first'])
        if (idsPerFile['nunique'] > 1).any():
            print("\nERROR: Column 'ID' is not constant within every input CSV file.")
            print(" Use option '-p' to insert a column 'path' with file path-names as additional, unique identifiers.\n")
            sys.exit(1)
        if not idsPerFile['first'].is_unique:
            print("\nERROR: Patient identifiers in column 'ID' are not all distinct.")
            print(" Use option '-p' to insert a column 'path' with file path-names as additional, unique identifiers.\n")
            sys.exit(1)


    uL, uC = np.unique(nRows, return_counts=True)
    if args.verbose:
        if len(uL)==1:
            pass #print(f'All CSV files had {uL[0]} rows')
        else:
            print('\nWARNING: Not all CSV files have the same number of rows:')
            print(pd.DataFrame({'row count':uL, 'found in # files': uC}))
    
    
    # Insert pathnames
    if args.insertPath >= 0:
        fnamesRep = [n*[fn] for n,fn in zip(nRows,fnames)]
        fnamesRep = [item for sublist in fnamesRep for item in sublist]
        df.insert(args.insertPath, 'path', fnamesRep)


    df.reset_index(drop=True,inplace=True)

    if args.verbose:
        print("\nAggregated table:")
        print(df)
    
    if args.split:
        # 'metric' is NaN for bookkeeping rows because pandas' read_csv coerces the
        # literal 'NA' string written by integrate_masks() to NaN by default. The
        # explicit string check keeps this working even if that default is ever
        # overridden (e.g. keep_default_na=False) upstream.
        isDebug = df['metric'].isna() | (df['metric'].astype(str).str.strip() == 'NA')
        dfMetric = df[~isDebug]
        dfDebug = df[isDebug].dropna(axis=1, how='all')
        if args.verbose:
            print(f'\nWriting aggregated metrics table to:\n {fnMetric}')
            print(f'Writing aggregated debugging table to:\n {fnDebug}\n')
        dfMetric.to_csv(fnMetric, index=False)
        dfDebug.to_csv(fnDebug, index=False)
    else:
        if args.verbose:
            print(f'\nWriting aggregated table to:\n {fnOut}\n')
        df.replace(np.nan, '', inplace=True)
        df.to_csv(fnOut, index=False)
