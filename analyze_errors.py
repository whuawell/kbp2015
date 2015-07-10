""" analyze_errors.py
Usage: analyze_errors.py <pred_out>

"""
from collections import Counter

import json
from text.dataset import Example

""" wanted format
in may 0000 , after finishing the 5-year-term of president of the republic of [macedonia]_2 , [branko crvenkovski]_1 returned to the sdum and was reelected leader of the party .
branko crvenkovski      PERSON
macedonia       LOCATION
per:countries_of_residence
PATH=7
LOCATION **PAD** LOCATION
republic nmod1 LOCATION
president nmod1 O
5-year-term nmod1 DURATION
finishing dobj1 O
returned advcl1 O
PERSON nsubj2 PERSON
"""

def safe_encode(lines):
    return u"\n".join(lines).encode('utf-8').strip()

def print_example(ex, fout):
    lines = [
        ' '.join(ex.debug),
        ex.subject + ' ' + ex.subject_ner,
        ex.object + ' ' + ex.object_ner,
        ex.relation + ' ' + ex.predicted_relation,
        'PATH = ' + str(len(ex.words)),
    ]
    for word, dep, ner in zip(ex.words, ex.parse, ex.ner):
        lines += [' '.join([word, dep, ner])]
    fout.write(safe_encode(lines) + "\n\n")

if __name__ == '__main__':
    from docopt import docopt
    args = docopt(__doc__)

    error_by_length = Counter()
    length_count = Counter()

    with open(args['<pred_out>']) as fin, open(args['<pred_out>'] + '.analysis', 'wb') as fout:
        for line in fin:
            ex = Example(**json.loads(line))
            length_count[len(ex.parse)] += 1
            if ex.relation != ex.predicted_relation:
                print_example(ex, fout)
                error_by_length[len(ex.parse)] += 1

        print >> fout, "length\tcount\tnum_error\tpercent_error"
        for length, count in length_count.most_common():
            num_error = error_by_length[length]
            print >> fout, "\t".join([str(e) for e in [length, count, num_error, num_error/float(count)]])
