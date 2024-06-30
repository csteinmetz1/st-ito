# IDMT-SMT-Guitar (acoustic and electric guitar)
python scripts/data/vst_datagen_psm_pedalboard.py \
"/import/c4dm-datasets-ext/IDMT_SMT_GUITAR/dataset4/Ibanez 2820" \
/import/c4dm-datasets-ext/lcap-datasets/psm-benchmark-dataset/idmt-smt-guitar-electric-pedalboard \
--sample_rate 48000 \
--num_examples 10 \
--multi_effect

python scripts/data/vst_datagen_psm_pedalboard.py \
"/import/c4dm-datasets-ext/IDMT_SMT_GUITAR/dataset4/acoustic_mic" \
/import/c4dm-datasets-ext/lcap-datasets/psm-benchmark-dataset/idmt-smt-guitar-acoustic-pedalboard \
--sample_rate 48000 \
--num_examples 10 \
--multi_effect

# VocalSet (singing voice)
python scripts/data/vst_datagen_psm_pedalboard.py \
"/import/c4dm-datasets/VocalSet1-2/data_by_singer" \
/import/c4dm-datasets-ext/lcap-datasets/psm-benchmark-dataset/vocalset-pedalboard \
--sample_rate 48000 \
--num_examples 10 \
--multi_effect

# DAPS (speech)
#python scripts/data/vst_datagen_psm_pedalboard.py \
#"/import/c4dm-datasets/daps_dataset/clean" \
#/import/c4dm-datasets-ext/lcap-datasets/psm-benchmark-dataset/daps-pedalboard \
#--sample_rate 48000 \
#--num_examples 10 \
#--multi_effect

# ENST-drums (drums) (mixes only)
python scripts/data/vst_datagen_psm_pedalboard.py \
"/import/c4dm-datasets/ENST-drums-mixes" \
/import/c4dm-datasets-ext/lcap-datasets/psm-benchmark-dataset/enst-drums-pedalboard \
--sample_rate 48000 \
--num_examples 10 \
--multi_effect