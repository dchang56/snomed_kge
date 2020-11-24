# SNOMED-CT Knowledge Graph Embeddings

This repository accompanies the paper Benchmark and Best Practices for Biomedical Knowledge Graph Embeddings: https://www.aclweb.org/anthology/2020.bionlp-1.18.pdf


Make sure to have graphvite and pytorch installed, and replace the application.py file in the graphvite site-packages directory in your environment with the application.py file in this repo (made some modification to include different functionalities for training, evaluating, and visualization).

## Downloads

- MRCONSO.RRF, MRREL.RRF, and MRSTY.RRF from the UMLS
- transitive closure resources from https://www.nlm.nih.gov/healthit/snomedct/us_edition.html
- semantic group files from https://metamap.nlm.nih.gov/SemanticTypesAndGroups.shtml

## Preprocessing and dataset building

- follow umls_preprocess.txt to get active_concepts.txt, active_relations.txt, and semantic_types.txt
- follow transitive_closure.ipynb to get transitive_closure_full.txt
- follow umls_utils.ipynb to further subset concepts and relations according to semantic types/groups, and create datasets for training/validation/testing.


## Train, eval, and visualize

- graphvite_experiments.py: runs the training and evaluation

To train all 5 KGE models:

```bash
for X in TransE DistMult SimplE ComplEx RotatE; do
  OUTPUT_DIR='output directory'
  DATA_DIR='data directory'
  GPU=0

  mkdir -p $OUTPUT_DIR

  python graphvite_experiments.py \
    --data_dir $DATA_DIR \
    --transitive_closure \
    --model_name $X \
    --num_epoch 200 \
    --dim 512 \
    --gpu $GPU \
    --num_negative 60 \
    --margin 8 \
    --learning_rate 5e-4 \
	--output_dir=$OUTPUT_DIR \
    --per_relation_eval \
    --target both \
    --save_model $X
    
done
```

- entity_classification.py: runs entity classification (requires cui2vec and snomed2vec)
- visualization.py: visualizes the trained embeddings

## Pretrained embeddings

Here's the GDrive link to download pretrained embeddings for the 5 KGE models: https://drive.google.com/drive/folders/1HN-TfZT9S3qHBIZO4_CItn4OWbuKZtll?usp=sharing

## Comments

When I was working on this project almost a year ago, I chose Graphvite as the package for kge training due to its advantages at the time. There has been a lot of progress in the field since then, and currently my personal favorite for training KGEs is dgl-ke: https://github.com/awslabs/dgl-ke
