"""Opt-in invariant caching for the pinned official model; defaults unchanged."""
import argparse
from pathlib import Path
import runpy
import sys
from cached_diffusion import install


def main():
    parser=argparse.ArgumentParser(description=__doc__,add_help=False)
    parser.add_argument('--cache-record',type=Path)
    args,cli=parser.parse_known_args()
    save=install(args.cache_record)
    sys.argv=['run_openfold',*cli]
    try:runpy.run_module('openfold3.run_openfold',run_name='__main__')
    finally:save()


if __name__=='__main__':main()
