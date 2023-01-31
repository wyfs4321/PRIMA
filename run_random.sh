# April
# epsilon test
python prefix_sum.py --alpha 1  --beta 0 --epsilon 1 --value_dimen_origin 30 --target_column 12 --n 100000 --clip_method April --disable_maxentropy False --data_random
python prefix_sum.py --alpha 1  --beta 0 --epsilon 2 --value_dimen_origin 30 --target_column 12 --n 100000 --clip_method April --disable_maxentropy False --data_random
python prefix_sum.py --alpha 1  --beta 0 --epsilon 3 --value_dimen_origin 30 --target_column 12 --n 100000 --clip_method April --disable_maxentropy False --data_random
python prefix_sum.py --alpha 1  --beta 0 --epsilon 4 --value_dimen_origin 30 --target_column 12 --n 100000 --clip_method April --disable_maxentropy False --data_random
python prefix_sum.py --alpha 1  --beta 0 --epsilon 5 --value_dimen_origin 30 --target_column 12 --n 100000 --clip_method April --disable_maxentropy False --data_random

# value_dimen test
python prefix_sum.py --alpha 1  --beta 0 --epsilon 3 --value_dimen_origin 10 --target_column 12 --n 100000 --clip_method April --disable_maxentropy False --data_random
python prefix_sum.py --alpha 1  --beta 0 --epsilon 3 --value_dimen_origin 20 --target_column 12 --n 100000 --clip_method April --disable_maxentropy False --data_random
python prefix_sum.py --alpha 1  --beta 0 --epsilon 3 --value_dimen_origin 30 --target_column 12 --n 100000 --clip_method April --disable_maxentropy False --data_random
python prefix_sum.py --alpha 1  --beta 0 --epsilon 3 --value_dimen_origin 40 --target_column 12 --n 100000 --clip_method April --disable_maxentropy False --data_random
python prefix_sum.py --alpha 1  --beta 0 --epsilon 3 --value_dimen_origin 50 --target_column 12 --n 100000 --clip_method April --disable_maxentropy False --data_random

# target_column test
python prefix_sum.py --alpha 1  --beta 0 --epsilon 3 --value_dimen_origin 30 --target_column 10 --n 100000 --clip_method April --disable_maxentropy False --data_random
python prefix_sum.py --alpha 1  --beta 0 --epsilon 3 --value_dimen_origin 30 --target_column 11 --n 100000 --clip_method April --disable_maxentropy False --data_random
python prefix_sum.py --alpha 1  --beta 0 --epsilon 3 --value_dimen_origin 30 --target_column 12 --n 100000 --clip_method April --disable_maxentropy False --data_random
python prefix_sum.py --alpha 1  --beta 0 --epsilon 3 --value_dimen_origin 30 --target_column 13 --n 100000 --clip_method April --disable_maxentropy False --data_random
python prefix_sum.py --alpha 1  --beta 0 --epsilon 3 --value_dimen_origin 30 --target_column 14 --n 100000 --clip_method April --disable_maxentropy False --data_random

# n test
python prefix_sum.py --alpha 1  --beta 0 --epsilon 3 --value_dimen_origin 30 --target_column 12 --n 10000 --clip_method April --disable_maxentropy False --data_random
python prefix_sum.py --alpha 1  --beta 0 --epsilon 3 --value_dimen_origin 30 --target_column 12 --n 50000 --clip_method April --disable_maxentropy False --data_random
python prefix_sum.py --alpha 1  --beta 0 --epsilon 3 --value_dimen_origin 30 --target_column 12 --n 100000 --clip_method April --disable_maxentropy False --data_random
python prefix_sum.py --alpha 1  --beta 0 --epsilon 3 --value_dimen_origin 30 --target_column 12 --n 500000 --clip_method April --disable_maxentropy False --data_random
python prefix_sum.py --alpha 1  --beta 0 --epsilon 3 --value_dimen_origin 30 --target_column 12 --n 1000000 --clip_method April --disable_maxentropy False --data_random

# SVTP
# epsilon test
python prefix_sum.py --alpha 1  --beta 0 --epsilon 1 --value_dimen_origin 30 --target_column 12 --n 100000 --clip_method SVTP --disable_maxentropy False --disable_prefixSum --data_random
python prefix_sum.py --alpha 1  --beta 0 --epsilon 2 --value_dimen_origin 30 --target_column 12 --n 100000 --clip_method SVTP --disable_maxentropy False --disable_prefixSum --data_random
python prefix_sum.py --alpha 1  --beta 0 --epsilon 3 --value_dimen_origin 30 --target_column 12 --n 100000 --clip_method SVTP --disable_maxentropy False --disable_prefixSum --data_random
python prefix_sum.py --alpha 1  --beta 0 --epsilon 4 --value_dimen_origin 30 --target_column 12 --n 100000 --clip_method SVTP --disable_maxentropy False --disable_prefixSum --data_random
python prefix_sum.py --alpha 1  --beta 0 --epsilon 5 --value_dimen_origin 30 --target_column 12 --n 100000 --clip_method SVTP --disable_maxentropy False --disable_prefixSum --data_random

# value_dimen test
python prefix_sum.py --alpha 1  --beta 0 --epsilon 3 --value_dimen_origin 10 --target_column 12 --n 100000 --clip_method SVTP --disable_maxentropy False --disable_prefixSum --data_random
python prefix_sum.py --alpha 1  --beta 0 --epsilon 3 --value_dimen_origin 20 --target_column 12 --n 100000 --clip_method SVTP --disable_maxentropy False --disable_prefixSum --data_random
python prefix_sum.py --alpha 1  --beta 0 --epsilon 3 --value_dimen_origin 30 --target_column 12 --n 100000 --clip_method SVTP --disable_maxentropy False --disable_prefixSum --data_random
python prefix_sum.py --alpha 1  --beta 0 --epsilon 3 --value_dimen_origin 40 --target_column 12 --n 100000 --clip_method SVTP --disable_maxentropy False --disable_prefixSum --data_random
python prefix_sum.py --alpha 1  --beta 0 --epsilon 3 --value_dimen_origin 50 --target_column 12 --n 100000 --clip_method SVTP --disable_maxentropy False --disable_prefixSum --data_random

# target_column test
python prefix_sum.py --alpha 1  --beta 0 --epsilon 3 --value_dimen_origin 30 --target_column 10 --n 100000 --clip_method SVTP --disable_maxentropy False --disable_prefixSum --data_random
python prefix_sum.py --alpha 1  --beta 0 --epsilon 3 --value_dimen_origin 30 --target_column 11 --n 100000 --clip_method SVTP --disable_maxentropy False --disable_prefixSum --data_random
python prefix_sum.py --alpha 1  --beta 0 --epsilon 3 --value_dimen_origin 30 --target_column 12 --n 100000 --clip_method SVTP --disable_maxentropy False --disable_prefixSum --data_random
python prefix_sum.py --alpha 1  --beta 0 --epsilon 3 --value_dimen_origin 30 --target_column 13 --n 100000 --clip_method SVTP --disable_maxentropy False --disable_prefixSum --data_random
python prefix_sum.py --alpha 1  --beta 0 --epsilon 3 --value_dimen_origin 30 --target_column 14 --n 100000 --clip_method SVTP --disable_maxentropy False --disable_prefixSum --data_random

# n test
python prefix_sum.py --alpha 1  --beta 0 --epsilon 3 --value_dimen_origin 30 --target_column 12 --n 10000 --clip_method SVTP --disable_maxentropy False --disable_prefixSum --data_random
python prefix_sum.py --alpha 1  --beta 0 --epsilon 3 --value_dimen_origin 30 --target_column 12 --n 50000 --clip_method SVTP --disable_maxentropy False --disable_prefixSum --data_random
python prefix_sum.py --alpha 1  --beta 0 --epsilon 3 --value_dimen_origin 30 --target_column 12 --n 100000 --clip_method SVTP --disable_maxentropy False --disable_prefixSum --data_random
python prefix_sum.py --alpha 1  --beta 0 --epsilon 3 --value_dimen_origin 30 --target_column 12 --n 500000 --clip_method SVTP --disable_maxentropy False --disable_prefixSum --data_random
python prefix_sum.py --alpha 1  --beta 0 --epsilon 3 --value_dimen_origin 30 --target_column 12 --n 1000000 --clip_method SVTP --disable_maxentropy False --disable_prefixSum --data_random

# R2TP
# epsilon test
python prefix_sum.py --alpha 1  --beta 0 --epsilon 1 --value_dimen_origin 30 --target_column 12 --n 100000 --clip_method R2TP --disable_maxentropy False --disable_prefixSum --data_random
python prefix_sum.py --alpha 1  --beta 0 --epsilon 2 --value_dimen_origin 30 --target_column 12 --n 100000 --clip_method R2TP --disable_maxentropy False --disable_prefixSum --data_random
python prefix_sum.py --alpha 1  --beta 0 --epsilon 3 --value_dimen_origin 30 --target_column 12 --n 100000 --clip_method R2TP --disable_maxentropy False --disable_prefixSum --data_random
python prefix_sum.py --alpha 1  --beta 0 --epsilon 4 --value_dimen_origin 30 --target_column 12 --n 100000 --clip_method R2TP --disable_maxentropy False --disable_prefixSum --data_random
python prefix_sum.py --alpha 1  --beta 0 --epsilon 5 --value_dimen_origin 30 --target_column 12 --n 100000 --clip_method R2TP --disable_maxentropy False --disable_prefixSum --data_random

# value_dimen test
python prefix_sum.py --alpha 1  --beta 0 --epsilon 3 --value_dimen_origin 10 --target_column 12 --n 100000 --clip_method R2TP --disable_maxentropy False --disable_prefixSum --data_random
python prefix_sum.py --alpha 1  --beta 0 --epsilon 3 --value_dimen_origin 20 --target_column 12 --n 100000 --clip_method R2TP --disable_maxentropy False --disable_prefixSum --data_random
python prefix_sum.py --alpha 1  --beta 0 --epsilon 3 --value_dimen_origin 30 --target_column 12 --n 100000 --clip_method R2TP --disable_maxentropy False --disable_prefixSum --data_random
python prefix_sum.py --alpha 1  --beta 0 --epsilon 3 --value_dimen_origin 40 --target_column 12 --n 100000 --clip_method R2TP --disable_maxentropy False --disable_prefixSum --data_random
python prefix_sum.py --alpha 1  --beta 0 --epsilon 3 --value_dimen_origin 50 --target_column 12 --n 100000 --clip_method R2TP --disable_maxentropy False --disable_prefixSum --data_random

# target_column test
python prefix_sum.py --alpha 1  --beta 0 --epsilon 3 --value_dimen_origin 30 --target_column 10 --n 100000 --clip_method R2TP --disable_maxentropy False --disable_prefixSum --data_random
python prefix_sum.py --alpha 1  --beta 0 --epsilon 3 --value_dimen_origin 30 --target_column 11 --n 100000 --clip_method R2TP --disable_maxentropy False --disable_prefixSum --data_random
python prefix_sum.py --alpha 1  --beta 0 --epsilon 3 --value_dimen_origin 30 --target_column 12 --n 100000 --clip_method R2TP --disable_maxentropy False --disable_prefixSum --data_random
python prefix_sum.py --alpha 1  --beta 0 --epsilon 3 --value_dimen_origin 30 --target_column 13 --n 100000 --clip_method R2TP --disable_maxentropy False --disable_prefixSum --data_random
python prefix_sum.py --alpha 1  --beta 0 --epsilon 3 --value_dimen_origin 30 --target_column 14 --n 100000 --clip_method R2TP --disable_maxentropy False --disable_prefixSum --data_random

# n test
python prefix_sum.py --alpha 1  --beta 0 --epsilon 3 --value_dimen_origin 30 --target_column 12 --n 10000 --clip_method R2TP --disable_maxentropy False --disable_prefixSum --data_random
python prefix_sum.py --alpha 1  --beta 0 --epsilon 3 --value_dimen_origin 30 --target_column 12 --n 50000 --clip_method R2TP --disable_maxentropy False --disable_prefixSum --data_random
python prefix_sum.py --alpha 1  --beta 0 --epsilon 3 --value_dimen_origin 30 --target_column 12 --n 100000 --clip_method R2TP --disable_maxentropy False --disable_prefixSum --data_random
python prefix_sum.py --alpha 1  --beta 0 --epsilon 3 --value_dimen_origin 30 --target_column 12 --n 500000 --clip_method R2TP --disable_maxentropy False --disable_prefixSum --data_random
python prefix_sum.py --alpha 1  --beta 0 --epsilon 3 --value_dimen_origin 30 --target_column 12 --n 1000000 --clip_method R2TP --disable_maxentropy False --disable_prefixSum --data_random

