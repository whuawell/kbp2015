""" align_reports.py
Usage: align_reports.py <report> [--gabor=<REPORT>]

Options:
    --gabor=<REPORT>        [default: gabor_report.txt]

"""
from collections import OrderedDict

__author__ = 'victor'
from docopt import docopt
from tabulate import tabulate

if __name__ == '__main__':
    args = docopt(__doc__)

    report = OrderedDict()
    with open(args['<report>']) as f:
        lines = f.readlines()[2:-2] # first two lines are headers, last two lines are averages
        for line in lines:
            #                         no_relation       0.86      0.34      0.49      6191
            relation, precision, recall, f1, support = line.split()
            precision, recall, f1 = ["{:.2%}".format(float(e)) for e in [precision, recall, f1]]
            report[relation] = (precision, recall, f1, support)

    gabor = {}
    with open(args['--gabor']) as f:
        for line in f:
            # [org:number_of_employees/members]  #: 9  P: 100.00%  R: 0.00%  F1: 0.00%
            relation, _, support, _, precision, _, recall, _, f1 = line.split()
            gabor[relation.strip('[]')] = (precision, recall, f1, support)

    header = ['relation', 'nn_precision', 'nn_recall', 'nn_f1', 'nn_support', 'sup_precision', 'sup_recall', 'sup_f1', 'sup_support']
    table = []
    for relation in report.keys():
        precision, recall, f1, support = report[relation]
        g_precision, g_recall, g_f1, g_support = gabor[relation] if relation in gabor else 4 * ['N/A']
        row = [relation, precision, recall, f1, support, g_precision, g_recall, g_f1, g_support]
        table += [row]

    with open(args['<report>'] + '.comparison', 'wb') as f:
        f.write(tabulate(table, headers=header))
