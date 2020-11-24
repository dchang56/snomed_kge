import os
import torch
import numpy as np
import pandas as pd
import graphvite as gv
import graphvite.application as gap
import argparse
import time
from collections import defaultdict
from torch.optim.lr_scheduler import ExponentialLR
from tqdm import tqdm, trange
import pickle


def main():

    #argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str)
    parser.add_argument('--transitive_closure', action='store_true')
    parser.add_argument('--output_dir', type=str)

    parser.add_argument('--per_relation_eval', action='store_true')

    parser.add_argument('--model_name', type=str, default='RotatE')
    parser.add_argument('--num_epoch', type=int, default=500)

    parser.add_argument("--dim", type=int, default=512)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--num_negative", type=int, default=24)
    parser.add_argument("--margin", type=int, default=7)
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--batch_size', type=int, default=2000)
    parser.add_argument('--test_batch_size', type=int, default=2000)

    parser.add_argument('--adversarial_temperature', type=float, default=1.0)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--schedule', type=str, default='linear')
    parser.add_argument('--log_frequency', type=int, default=400)

    parser.add_argument('--target', type=str, default='both')
    parser.add_argument('--fast_mode', type=int, default=None)
    parser.add_argument('--save_model', type=str)

    parser.add_argument('--load_model', type=str)
    parser.add_argument('--eval_only', action='store_true')

    #text embedding stuff
    # parser.add_argument('--entity_text_embeddings', type=str)
    # parser.add_argument('--relation_text_embeddings', type=str)
    # parser.add_argument('--entity_df', type=str)
    # parser.add_argument('--relation_df', type=str)

    args = parser.parse_args()
    args.cuda = True

    train_file = os.path.join(args.data_dir, 'train.txt')
    valid_file = os.path.join(args.data_dir, 'valid.txt')
    test_file = os.path.join(args.data_dir, 'test.txt')

    # if args.entity_text_embeddings:
    #     print('loading pretrained text embeddings!')
    #     args.resume = True
    #     entity_text_embeddings = np.load(args.entity_text_embeddings)
    #     relation_text_embeddings = np.load(args.relation_text_embeddings)
    #     entity_list = pd.read_csv(args.entity_df)
    #     entity_list.columns = ['CUI', 'text']
    #     entity_list = entity_list.set_index('CUI').index

    #     relation_list = pd.read_csv(args.relation_df)
    #     relation_list = relation_list['relations'].reset_index().set_index('relations').index

    #     entity2id = app.graph.entity2id
    #     relation2id = app.graph.relation2id

    #     filter_entity_list = entity_list.isin(entity2id)
    #     entity_text_embeddings = entity_text_embeddings[filter_entity_list]
    #     filter_relation_list = relation_list.isin(relation2id)
    #     relation_text_embeddings = relation_text_embeddings[filter_relation_list]

    #     #filter to seen entities and relations
    #     f_entity_list = entity_list[filter_entity_list]
    #     new_entity_indexes = [entity2id[cui] for cui in f_entity_list]
    #     entity_text_embeddings = entity_text_embeddings[new_entity_indexes]

    #     #normalize
    #     entity_text_embeddings = entity_text_embeddings / np.linalg.norm(entity_text_embeddings)
    #     # relation_text_embeddings = relation_text_embeddings / np.linalg.norm(relation_text_embeddings)

    #     f_relation_list = relation_list[filter_relation_list]
    #     new_relation_indexes = [relation2id[rel] for rel in f_relation_list]
    #     relation_text_embeddings = relation_text_embeddings[new_relation_indexes]

    #     app.solver.entity_embeddings[:] = entity_text_embeddings
    #     # app.solver.relation_embeddings[:] = relation_text_embeddings

    #     print('finished loading pretrained embeddings')

    #putting resume in train just prevents initializing tensors
    args.resume = False
    print(args.resume)

    app = gap.KnowledgeGraphApplication(dim=args.dim, gpus=[args.gpu])
    app.load(file_name=train_file)
    app.build(optimizer=gv.optimizer.Adam(lr=args.learning_rate,
                                          weight_decay=args.weight_decay),
              num_negative=args.num_negative)

    if args.eval_only:
        args.num_epoch = 0
        #Training
    app.train(model=args.model_name, num_epoch=args.num_epoch, sample_batch_size=args.batch_size , margin=args.margin, adversarial_temperature=args.adversarial_temperature, \
        log_frequency=args.log_frequency, resume=args.resume)

    if args.load_model:
        args.resume = True
        with open(args.load_model, 'rb') as fin:
            model = pickle.load(fin)
        app.solver.entity_embeddings[:] = model.solver.entity_embeddings
        app.solver.relation_embeddings[:] = model.solver.relation_embeddings

    # Evaluating
    eval_start = time.time()
    filter_files = [train_file, valid_file, test_file]
    if args.transitive_closure:
        filter_files.append(
            os.path.join(args.data_dir, 'transitive_closure_triplets.txt'))
    print(args.target)
    print('embeddings at eval time')
    print(app.solver.entity_embeddings[0][:10])
    results = app.link_prediction(file_name=test_file,
                                  filter_files=filter_files,
                                  target=args.target,
                                  data_dir=args.data_dir,
                                  output_dir=args.output_dir,
                                  per_relation_eval=args.per_relation_eval,
                                  fast_mode=args.fast_mode)
    print('evaluation took {}s'.format(time.time() - eval_start))
    print(results)

    with open(os.path.join(args.output_dir, 'results.txt'), 'w') as fout:
        for k, v in results.items():
            fout.write('{}: {}\n'.format(k, str(round(v, 4))))

    if args.save_model:
        app.save_model(
            os.path.join(args.output_dir, '{}.pkl'.format(args.save_model)))

    print('final embeddings')
    print(app.solver.entity_embeddings[0][:10])


if __name__ == '__main__':
    main()
    print('done!')
