#!/bin/bash
d=gowalla

python nway_train.py --data_path ./data/$d --train_file ${d}_train.tsv --valid_file ${d}_valid.tsv --test_file ${d}_test.tsv --result_dir ./result/$d --model SCPN  --batch-size 2048;
python nway_train.py --data_path ./data/$d --train_file ${d}_train.tsv --valid_file ${d}_valid.tsv --test_file ${d}_test.tsv --result_dir ./result/$d --model STuckerN  --batch-size 256;
python nway_train.py --data_path ./data/$d --train_file ${d}_train.tsv --valid_file ${d}_valid.tsv --test_file ${d}_test.tsv --result_dir ./result/$d --model SMDMTF  --batch-size 256;
python nway_train.py --data_path ./data/$d --train_file ${d}_train.tsv --valid_file ${d}_valid.tsv --test_file ${d}_test.tsv --result_dir ./result/$d --model SCostCoN  --batch-size 256;
python nway_train.py --data_path ./data/$d --train_file ${d}_train.tsv --valid_file ${d}_valid.tsv --test_file ${d}_test.tsv --result_dir ./result/$d --model SMLPN  --batch-size 256;
python nway_train.py --data_path ./data/$d --train_file ${d}_train.tsv --valid_file ${d}_valid.tsv --test_file ${d}_test.tsv --result_dir ./result/$d --model SNeAT  --batch-size 256;