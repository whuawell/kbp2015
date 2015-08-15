#!/usr/bin/env zsh
set -x
if [ $HOMENAME = jagupard6.stanford.edu ]; then
  DEVICE=gpu2
else
  DEVICE=gpu1
fi

echo "querying ${GREENPLUM_HOST} and running using ${DEVICE}"

# THEANO_FLAGS='device=${DEVICE}' psql -h ${GREENPLUM_HOST} -p 4242 -U kbp kbp -c "copy (select * from test_data where subject_entity <> object_entity and corpus_id=2010 and subject_ner like 'PERSON') to stdout with null as ''" | ./kbp.py > kbp_pred.2010.person.tsv &
# THEANO_FLAGS='device=${DEVICE}' psql -h ${GREENPLUM_HOST} -p 4242 -U kbp kbp -c "copy (select * from test_data where subject_entity <> object_entity and corpus_id=2010 and subject_ner not like 'PERSON') to stdout with null as ''" | ./kbp.py > kbp_pred.2010.nonperson.tsv &
# THEANO_FLAGS='device=${DEVICE}' psql -h ${GREENPLUM_HOST} -p 4242 -U kbp kbp -c "copy (select * from test_data where subject_entity <> object_entity and corpus_id=2013 and subject_ner like 'PERSON') to stdout with null as ''" | ./kbp.py > kbp_pred.2013.person.tsv &
# THEANO_FLAGS='device=${DEVICE}' psql -h ${GREENPLUM_HOST} -p 4242 -U kbp kbp -c "copy (select * from test_data where subject_entity <> object_entity and corpus_id=2013 and subject_ner not like 'PERSON') to stdout with null as ''" | ./kbp.py > kbp_pred.2013.nonperson.tsv &

THEANO_FLAGS='device=${DEVICE}' psql -h ${GREENPLUM_HOST} -p 4242 -U kbp kbp -c "copy (select * from test_data where subject_entity <> object_entity and corpus_id=2010) to stdout with null as ''" | ./kbp.py > kbp_pred.2010.tsv &
THEANO_FLAGS='device=${DEVICE}' psql -h ${GREENPLUM_HOST} -p 4242 -U kbp kbp -c "copy (select * from test_data where subject_entity <> object_entity and corpus_id=2013) to stdout with null as ''" | ./kbp.py > kbp_pred.2013.tsv &
