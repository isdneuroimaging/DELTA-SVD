#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Aggregate PSMD2 result tables across subjects/patients
"""

import sys, os, glob, argparse, re
import numpy as np
import pandas as pd
import datetime

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
        if len(base)>4 and base[-4:]=='.csv' and (len(directory)==0 or (len(directory)>1 and os.path.isdir(directory))):
            return path
        else:
            raise argparse.ArgumentTypeError("File path has to be an existing directory or a filename. If filename, it must end with '.csv' and can be optionally prepended by the path to an existing directory. Please check: %s"%(path))

fnOutDefault = "psmd2_results_aggregated.csv"
license = "http://psmd-marker.com"

def iniParser():
    parser = argparse.ArgumentParser(description="Aggregate multiple PSMD2 result tables into one table. Files with result tables will be globbed in the specified DIRECTORY using the specified FILENAME and DEPTH.",
                                     add_help=False,
                                     epilog=f'Notice: By using PSMD, you agree to the software license terms described at "{license}"')
    group0 = parser.add_argument_group()
    group0.add_argument(dest="directory", metavar='DIRECTORY', type=isDir, help="path to directory containing PSMD2 result files (at any depth). Will be used for globbing.")
    group0.add_argument("-f", dest="filename", type=extCSV, default="psmd2_results.csv", help="name of PSMD2 result files (default: %(default)s). Will be used for globbing. Requires extension '.csv'.")
    group0.add_argument("-d", dest="depth", type=int, default=-1, help="depth for globbing (default: %(default)s, which means any depth). If set to '0', only the top-level directory will be searched, making sense only with wildcards ('*') in filename.")
    group0.add_argument("-o", dest="output", metavar="OUTPUT-PATH", type=plausiblePath, default=fnOutDefault, help="path to write aggregated table to (default: %(default)s). If left at default or only a filename is provided, it will be saved into the DIRECTORY provided for globbing. If only a directory is provided, the default output-filename will be used.")
    group0.add_argument("-s", dest="split", action='store_true', help='split into separate output tables for metrics and debugging information. The output table names will be constructed by appending "_metrics" and "_debugging" respectively.')
    group0.add_argument("-p", dest="insertPath", action='store_const', const=0, default=-1, help="insert a column 'path' with the path names of input CSV files into the aggregated table. By default, this will only be done, if the 'ID' column is missing in the input CSV files.")
    group0.add_argument("-t", dest="appendDate", choices=['date', 'time', 'datetime'], default=None, help="append output filename with current date, time, or datetime (default: %(default)s), formated as '*[_YYYY-MM-DD][_HHMMSS].csv'")
    group0.add_argument("-x", dest="overwrite", action='store_true', help="allow overwriting output if existing. By default, already existing output will raise an error. (Be careful not to glob previous aggregation files when repeating aggregation.)")
    group0.add_argument("-q", dest="verbose", action='store_false', help="quiet mode")
    group0.add_argument("-h", action="help", help="show this help message and exit")
    group0.add_argument("-help","--help", action="help", help=argparse.SUPPRESS)
    # parser._action_groups.reverse()
    return parser      


if __name__ == "__main__":

    parser = iniParser()
    if len(sys.argv)==1:
        parser.print_usage()
        print(f'\nRun "{os.path.basename(__file__)} -h" for detailed help\n'
              f'Notice: By using PSMD, you agree to the software license terms described at "{license}"\n')
        parser.exit()
    
    args = parser.parse_args()


    if args.verbose:
        print("Running: " + " ".join([os.path.basename(sys.argv[0])]+sys.argv[1::]))
    
    # construct output filename path
    if os.path.isdir(args.output):
        fnOut = os.path.join(args.output, fnOutDefault)
    elif os.path.dirname(args.output)=='':
        fnOut = os.path.join(args.directory, args.output)
    else:
        fnOut = args.output
    
    # append date
    if args.appendDate:
        if args.appendDate == 'date':
            date = datetime.datetime.now().strftime("%Y-%m-%d")
        elif args.appendDate == 'time':
            date = datetime.datetime.now().strftime("%H%M%S")
        elif args.appendDate == 'datetime': 
            date = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
        fnOut = re.sub(r'\.csv$', f'_{date}.csv', fnOut)

    # append with different suffices, if split into metics and debugging info is requested      
    # and check existence of output files
    if args.split:
        fnMetric = re.sub(r'\.csv$', f'_metrics.csv', fnOut)
        fnDebug = re.sub(r'\.csv$', f'_debugging.csv', fnOut)
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
    
    # find all PSMD2 output CSV files
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
        # nShow = min(5, len(fnames))
        # print('Showing first '+str(nShow)+' files:')
        # for i in range(nShow):
        #     print(' '+fnames[i])

    # Read headers from each file
    # headers = [list(pd.read_csv(file, nrows=0).columns) for file in fnames]
    headers = list()
    fnamesOk = list()
    for i, file in enumerate(fnames):
        try:
            headers.append(list(pd.read_csv(file, nrows=0).columns))
            fnamesOk.append(file)
        except:
            print('WARNING: Excluding file due to reading error:', file)
    if len(fnamesOk)>0:
        fnames = fnamesOk
    else:
        print('ERROR: none of the globbed files can be read!')
        sys.exit(1)

    # Check if all table headers are the same
    headersStr = [f'{header}' for header in headers]
    headersStrUnq = set(headersStr)
    if len(headersStrUnq) > 1:
        print("\nERROR: Not all Headers are the same.")
        HeadersStrUnqCnt = [headersStr.count(uH) for uH in headersStrUnq]
        HeadersStrUnqIdx = [headersStr.index(uH) for uH in headersStrUnq]
        HeadersStrUnqLen = [len(f'{header}') for header in headersStrUnq]
        zipped = sorted(zip(HeadersStrUnqCnt, headersStrUnq, HeadersStrUnqLen, [fnames[i] for i in HeadersStrUnqIdx]),reverse=True)        
        ll = max(HeadersStrUnqLen)
        print("#Files | Header" + ' '*(ll-6+2) + " | Exapmple file")
        for i, row in enumerate(zipped):
            print("{:>6d} | {:s}".format(*row[0:2]) + '-'*(ll-row[2]+2) + " | {:<45s}".format(row[3]))
        print('Aborting! Aggregation of tables with diffrent headers not supported!')
        sys.exit(1)

    # Read all files
    df = pd.concat([pd.read_csv(fn) for fn in fnames])

    # Check if files contain the column 'ID'
    if not 'ID' in headers[0] and args.insertPath == -1:
        # Insert column 'path' if column 'ID' is missing, even if not explicitly requested
        print("\nWARNING: Column 'ID' not found. Inserting a column 'path' with CSV file paths as an alternative identifier.")
        args.insertPath = 0
    elif args.insertPath == 0:
        # We know already that column "ID" exists. Column "path" shell be inserted after column "ID". Here we calculate the index for it.
        args.insertPath = list(headers[0]).index("ID") + 1 
    elif isinstance(df.loc[0,'ID'], pd.Series) and not df.loc[0,'ID'].is_unique:
        # We know already that column 'ID' does exist but 'path' shall not be inserted. Additionally we found out that IDs are not unique and throw an error
        print("\nERROR: Patient identifiers in column 'ID' are not all distinct.")
        print(" Use option '-p' to insert a column 'path' with file path-names as additional, unique identifiers.\n")
        sys.exit(1)
    
        
    # Identify number of rows read per CSV file
    trueWhereIdxZero = list(df.index==0)
    trueWhereIdxLocalMax = trueWhereIdxZero[1:] + [trueWhereIdxZero[0]]
    nRows = df.index[trueWhereIdxLocalMax]+1
    uL, uC = np.unique(nRows, return_counts=True)
    if args.verbose:
        if len(uL)==1:
            pass #print(f'All CSV files had {uL[0]} rows')
        else:
            print(f'\nWARNING: Not all CSV files have the same number of rows:')
            print(pd.DataFrame({'row count':uL, 'found in # files': uC}))
    
    
    # Insert pathnames
    if args.insertPath >= 0:
        # Replicate CSV path names
        fnamesRep = [n*[fn] for n,fn in zip(nRows,fnames)]
        fnamesRep = [item for sublist in fnamesRep for item in sublist]
        # Insert path names as column into the aggregated table
        df.insert(args.insertPath, 'path', fnamesRep)


    df.reset_index(drop=True,inplace=True)
    # df = df.sort_values(by=['ID','timepoint'],axis=0)

    if args.verbose:
        print("\nAggregated table:")
        print(df)
    
    # split table, if requested, and save
    if args.split:
        dfMetric = df[df['metric'].notna()]
        dfDebug = df[df['metric'].isna()].dropna(axis=1, how='all')
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
