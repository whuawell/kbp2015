#!/usr/bin/env zsh
psql -h julie0 -p 4242 -U kbp kbp -c "copy (select * from test_data where subject_entity <> object_entity) to stdout with null as ''" | ./kbp.py > kbp_pred.tsv
