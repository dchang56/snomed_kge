# snomed_kge

Make sure to have graphvite and pytorch installed, and replace the application.py file in the graphvite site-packages directory in your environment with the application.py file in this repo (made some modification to include different functionalities for training, evaluating, and visualization).

Downloads:

-MRCONSO.RRF, MRREL.RRF, and MRSTY.RRF from the UMLS

-transitive closure resources from https://www.nlm.nih.gov/healthit/snomedct/us_edition.html

-semantic group files from https://metamap.nlm.nih.gov/SemanticTypesAndGroups.shtml

Preprocessing and dataset building:

-follow umls_preprocess.txt to get active_concepts.txt, active_relations.txt, and semantic_types.txt

-follow transitive_closure.ipynb to get transitive_closure_full.txt

-follow umls_utils.ipynb to further subset concepts and relations according to semantic types/groups, and create datasets for training/validation/testing.


Train, eval, and visualize:
-graphvite_experiments.py: runs the training and evaluation

To train all 5 KGE models:
`
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
'



