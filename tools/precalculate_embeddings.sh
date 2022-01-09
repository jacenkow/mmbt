python precalculate_embeddings.py \
    --input_folder ../../datasets/mimic_cxr/annotations \
    --model_file ../../datasets/mimic_cxr/extra/BioWordVec_PubMed_MIMICIII_d200.bin

python precalculate_embeddings.py \
    --input_folder ../../datasets/mimic_cxr/annotations \
    --model_file ../../datasets/mimic_cxr/extra/BioWordVec_PubMed_MIMICIII_d200.bin \
    --lmdb True
