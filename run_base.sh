#!/bin/bash
d=gowalla

python nway_train_base.py --model CP --batch-size 2048 --dataset $d;
python nway_train_base.py --model Tucker --batch-size 256 --dataset $d;
python nway_train_base.py --model CostCo --batch-size 256 --dataset $d;
python nway_train_base.py --model MLP --batch-size 256 --dataset $d;
python nway_train_base.py --model MDMTF --batch-size 256 --dataset $d;
python nway_train_base.py --model NeAT --batch-size 256 --dataset $d;