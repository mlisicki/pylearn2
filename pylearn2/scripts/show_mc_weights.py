#!/usr/bin/env python
"""
modified by Michal Lisicki
show multi-channel weights
"""
#usage: show_weights.py model.pkl
from pylearn2.gui import get_mc_weights_report
import argparse

def main():
    """
    .. todo::

        WRITEME
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("--rescale", default="individual")
    parser.add_argument("--out", default=None)
    parser.add_argument("--border", action="store_true", default=False)
    parser.add_argument("--channels", default='0,1,2')
    parser.add_argument("path")

    options = parser.parse_args()

    pv = get_mc_weights_report.get_weights_report(model_path=options.path,
                                                  rescale=options.rescale,
                                                  border=options.border,
                                                  channels=options.channels)

    if options.out is None:
        pv.show()
    else:
        pv.save(options.out)

if __name__ == "__main__":
    main()
