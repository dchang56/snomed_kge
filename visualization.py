import numpy as np
import matplotlib.pyplot as plt
import graphvite as gv
import graphvite.application as gap
import pickle
import os
import argparse
import random


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--output_dir', type=str, default='figures')
    parser.add_argument('--model', type=str)
    parser.add_argument('--visualize_relations', action='store_true')

    parser.add_argument('--num_epoch', type=int, default=100)
    parser.add_argument('--dim', type=int, default=2)
    parser.add_argument('--perplexity', type=int, default=80)
    parser.add_argument('--weight_decay', type=float, default=7e-5)
    parser.add_argument('--learning_rate', type=float, default=0.5)

    args = parser.parse_args()

    random.seed(42)
    # args.output_dir = os.path.join(args.output_dir, args.model)
    # if not os.path.exists(args.output_dir):
    #     os.makedirs(args.output_dir)

    kge_models = ('TransE', 'SimplE', 'DistMult', 'ComplEx', 'RotatE')
    other_models = ('Snomed2Vec', 'Cui2Vec')
    if args.model in kge_models:
        args.data_dir = os.path.join(args.data_dir, 'kge')
        with open(os.path.join(args.data_dir, '{}.pkl'.format(args.model)),
                  'rb') as fin:
            model = pickle.load(fin)
        entity_embeddings = model.solver.entity_embeddings
        relation_embeddings = model.solver.relation_embeddings
        sty_labels = np.load(os.path.join(args.data_dir, 'sty_labels.npy'))
        sg_labels = np.load(os.path.join(args.data_dir, 'sg_labels.npy'))
    elif args.model in other_models:
        entity_embeddings = np.load(
            os.path.join(args.data_dir,
                         '{}/{}.npy'.format(args.model, args.model)))
        sty_labels = np.load(
            os.path.join(args.data_dir,
                         '{}/sty_labels.npy'.format(args.model)))
        sg_labels = np.load(
            os.path.join(args.data_dir, '{}/sg_labels.npy'.format(args.model)))
    else:
        raise ValueError('models should be one of: {}'.format(
            ', '.join(kge_models + other_models)))

    assert len(sg_labels) == len(sty_labels)
    assert len(sg_labels) == len(entity_embeddings)

    app = gap.VisualizationApplication(dim=args.dim)
    app.load(vectors=entity_embeddings, perplexity=args.perplexity)
    app.build(optimizer=gv.optimizer.Adam(lr=args.learning_rate,
                                          weight_decay=args.weight_decay))
    app.train(num_epoch=args.num_epoch)

    ## For specific semantic type visualization, choose a broad semantic group to hone in on (otherwise it's too cluttered).
    # semantic_groups = np.unique(sg_labels)
    semantic_groups = ['CHEM', 'PROC']
    for sg in semantic_groups:
        specific_labels = np.copy(sty_labels)
        specific_labels[sg_labels != sg] = 'OTHER'
        # trim specific labels for the sake of visibility
        specific_labels[specific_labels ==
                        'Biomedical or Dental Material'] = 'OTHER'
        specific_labels[specific_labels ==
                        'Chemical Viewed Structurally'] = 'OTHER'
        specific_labels[specific_labels == 'Vitamin'] = 'OTHER'
        specific_labels[specific_labels == 'Receptor'] = 'OTHER'
        specific_labels[specific_labels == 'Chemical'] = 'OTHER'
        specific_labels[specific_labels == 'Antibiotic'] = 'OTHER'
        specific_labels[specific_labels == 'Hormone'] = 'OTHER'
        specific_labels[specific_labels ==
                        'Biologically Active Substance'] = 'OTHER'
        specific_labels[specific_labels ==
                        'Element, Ion, or Isotope'] = 'OTHER'
        specific_labels[specific_labels ==
                        'Hazardous or Poisonous Substance'] = 'OTHER'
        specific_labels[specific_labels ==
                        'Nucleic Acid, Nucleoside, or Nucleotide'] = 'OTHER'
        specific_labels[specific_labels == 'Inorganic Chemical'] = 'OTHER'
        specific_labels[specific_labels ==
                        'Molecular Biology Research Technique'] = 'OTHER'
        specific_labels[specific_labels == 'Research Activity'] = 'OTHER'
        specific_labels[specific_labels == 'Manufactured Object'] = 'OTHER'

        ax = app.visualization(Y=specific_labels)
        ax.set_title('{}: {} Semantic Types'.format(args.model, sg),
                     fontsize=27)
        plt.savefig(os.path.join(args.output_dir,
                                 '{}_{}.png'.format(args.model, sg)),
                    bbox_inches='tight')

    # semantic groups
    print('visualizing semantic groups')
    ax = app.visualization(Y=sg_labels)
    ax.set_title('{}'.format(args.model), fontsize=27)

    plt.savefig(os.path.join(args.output_dir, '{}_sg.png'.format(args.model)),
                bbox_inches='tight')

    print('starting relations viz')
    if args.visualize_relations:
        if args.model in kge_models:
            print(args.model)

            relation_embeddings = model.solver.relation_embeddings

            # oneormany_labels = np.load(os.path.join(args.data_dir, 'oneormany_labels.npy'))
            oneormany_labels = np.load(
                os.path.join(args.data_dir, 'oneormany_labels.npy'))

            assert len(oneormany_labels) == len(relation_embeddings)

            from sklearn.manifold import TSNE
            tsne = TSNE(n_components=2,
                        perplexity=12,
                        early_exaggeration=12,
                        learning_rate=150,
                        n_iter=8000,
                        n_iter_without_progress=500)
            coordinates = tsne.fit_transform(relation_embeddings)

            mean = np.mean(coordinates, axis=0)
            std = np.std(coordinates, axis=0)
            inside = np.abs(coordinates - mean) < 3 * std
            indexes, = np.where(np.all(inside, axis=1))
            coordinates = coordinates[indexes]
            oneormany_labels = oneormany_labels[indexes]

            classes = sorted(np.unique(oneormany_labels))
            fig = plt.figure(figsize=(10, 10))
            ax = fig.gca()
            markers = ['*', '>', '<', 'o', '+', 'X']
            for cls, m in zip(classes, markers):
                indexes, = np.where(oneormany_labels == cls)
                ax.scatter(*coordinates[indexes].T, s=35, marker=m)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.legend(classes, markerscale=1, loc='upper right')
            ax.set_title('{} Relation Type Embeddings'.format(args.model),
                         fontsize=27)
            plt.savefig(os.path.join(
                args.output_dir,
                '{}_oneormany_relations.png'.format(args.model)),
                        bbox_inches='tight')


if __name__ == '__main__':
    main()
    print('done!')