import argparse
from trainer import ModelTrainer, train_run
import torch
import os
import logging
from datetime import datetime
from data.data import QM9Dataloader, QMugsDataloader


def main(args):
    args.device = torch.device(args.device) if torch.cuda.is_available() else torch.device('cpu')

    directory = datetime.now().strftime("%Y%m%d_%H%M%S") + '_' + args.exp_name
    directory = os.path.join('logs', directory)
    if not os.path.exists(directory):
        os.makedirs(directory)

    logging.basicConfig(format='%(asctime)s %(message)s', filename=os.path.join(directory, f"{args.exp_name}.log"), level=logging.DEBUG)
    logger = logging.getLogger()
    sh = logging.StreamHandler()
    sh.setLevel(logging.DEBUG)
    logger.addHandler(sh)

    logging.info(args)

    # Prepare dataset
    if args.dataset == 'QM9':
        train_loader, valid_loader, test_loader, indices = QM9Dataloader(args)
    elif args.dataset == 'QMugs':
        train_loader, valid_loader, test_loader, indices = QMugsDataloader(args)


    if args.lacl:
        trainer = ModelTrainer(args, directory, train_loader, valid_loader, test_loader, indices[2])
        trainer.train()
    else:
        train_run(args, directory, train_loader, valid_loader, test_loader, indices[2], lacl=False)
    print(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='dipole moment prediction training')
    """Experiment setting."""
    parser.add_argument('--exp-name', type=str, default='qm9_mmff_G_src', help="Save experiment name")
    parser.add_argument('--lacl', type=bool, default=False, help="True for training LACL, False for training")
    parser.add_argument('--finetune', type=bool, default=False, help="")
    parser.add_argument('--freeze', type=bool, default=False, help="")
    parser.add_argument('--update_moving_average', type=bool, default=True, help="For BGRL")
    parser.add_argument('--loss', type=str, default='contrastive+prediction', help="")
    parser.add_argument('--num-workers', type=int, default=6, help="Number of workers for dataloader")
    parser.add_argument('--dataset', type=str, default='QM9', help="QM9 or QMugs")
    parser.add_argument('--set', type=str, default='src', help="'src' for source domain and 'tgt' for target domain")
    parser.add_argument('--target', type=str, default='G', help="homo, lumo, gap, mu, ...")
    parser.add_argument('--geometry', type=str, default='MMFF', help="")
    parser.add_argument('--epochs', type=int, default=300, help="")
    parser.add_argument('--num-train', type=int, default=110000, help="110000/65000")
    parser.add_argument('--num-valid', type=int, default=10000, help="10000/1500")
    parser.add_argument('--num-test', type=int, default=10829, help="10829/1706")
    parser.add_argument('--batch-size', type=int, default=64, help="")
    parser.add_argument('--learning-rate', type=float, default=1e-3, help="")
    parser.add_argument('--weight-decay', type=float, default=0, help="")
    parser.add_argument('--max-norm', type=float, default=1000.0, help="")
    parser.add_argument('--scheduler', type=str, default='plateau', help="")
    parser.add_argument('--cutoff', type=float, default=5.0, help="")
    parser.add_argument('--device', type=str, default='cuda:3', help="cuda device")
    '''Model setting'''
    parser.add_argument('--embedding-type', type=str, default='cgcnn', help="")
    parser.add_argument('--alignn-layers', type=int, default=4, help="")
    parser.add_argument('--gcn-layers', type=int, default=4, help="")
    parser.add_argument('--atom-input-features', type=int, default=92, help="")
    parser.add_argument('--edge-input-features', type=int, default=80, help="")
    parser.add_argument('--triplet-input-features', type=int, default=40, help="")
    parser.add_argument('--embedding-features', type=int, default=64, help="")
    parser.add_argument('--hidden-features', type=int, default=256, help="")
    parser.add_argument('--output-features', type=int, default=1, help="")
    args = parser.parse_args()

    # Learning
    main(args)
