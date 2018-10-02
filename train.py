import argparse
import time

from torch import nn
from torch import optim
from torch.utils.data import DataLoader

from config import *
from data_gen import ZslDataset
from models import Encoder
from utils import *


def train(epoch, train_loader, model, optimizer, attributes_per_class):
    # Ensure dropout layers are in train mode
    model.train()

    # Loss function
    criterion = nn.MSELoss().to(device)

    batch_time = ExpoAverageMeter()  # forward prop. + back prop. time
    losses = ExpoAverageMeter()  # loss (per word decoded)
    accs = ExpoAverageMeter()  # accuracy

    start = time.time()

    # Batches
    for i_batch, (imgs, label_ids, attributes) in enumerate(train_loader):
        # Zero gradients
        optimizer.zero_grad()

        # Set device options
        imgs = imgs.to(device)
        # print(img.size())
        label_ids = label_ids.view(-1).to(device)
        # print('label_ids: ' + str(label_ids))
        # print('label_ids.size(): ' + str(label_ids.size()))
        attributes = attributes.to(device)
        # print('targets: ' + str(targets))
        # print('targets.size(): ' + str(targets.size()))

        preds = model(imgs)
        _, scores = batched_KNN(preds, 1, attributes_per_class)
        # print('scores: ' + str(scores))
        # print('scores.size(): ' + str(scores.size()))

        loss = criterion(preds, attributes)
        loss.backward()

        optimizer.step()

        acc = accuracy(scores, label_ids)
        # print('acc: ' + str(acc))

        # Keep track of metrics
        losses.update(loss.item())
        batch_time.update(time.time() - start)
        accs.update(acc)

        start = time.time()

        # Print status
        if i_batch % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Accuracy {accs.val:.3f} ({accs.avg:.3f})'.format(epoch, i_batch, len(train_loader),
                                                                    batch_time=batch_time,
                                                                    loss=losses,
                                                                    accs=accs))


def valid(val_loader, model, attributes_per_class):
    model.eval()  # eval mode (no dropout or batchnorm)

    # Loss function
    criterion = nn.MSELoss().to(device)

    batch_time = ExpoAverageMeter()  # forward prop. + back prop. time
    losses = ExpoAverageMeter()  # loss (per word decoded)
    accs = ExpoAverageMeter()  # accuracy

    start = time.time()

    with torch.no_grad():
        # Batches
        for i_batch, (imgs, label_ids, attributes) in enumerate(val_loader):
            # Set device options
            imgs = imgs.to(device)
            label_ids = label_ids.view(-1).to(device)
            attributes = attributes.to(device)  # (batch_size, 123)

            preds = model(imgs)  # (batch_size, 123)

            loss = criterion(preds, attributes)

            _, scores = batched_KNN(preds, 1, attributes_per_class)
            acc = accuracy(scores, label_ids)

            # Keep track of metrics
            losses.update(loss.item())
            batch_time.update(time.time() - start)
            accs.update(acc)

            start = time.time()

            # Print status
            if i_batch % print_freq == 0:
                print('Validation: [{0}/{1}]\t'
                      'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Accuracy {accs.val:.3f} ({accs.avg:.3f})'.format(i_batch, len(val_loader),
                                                                        batch_time=batch_time,
                                                                        loss=losses,
                                                                        accs=accs))

    return accs.avg, losses.avg


def main(args):
    superclass = args['superclass']
    if superclass is None:
        superclass = 'Animals'
    train_loader = DataLoader(dataset=ZslDataset(superclass, 'train'), batch_size=batch_size, shuffle=True,
                              pin_memory=True, drop_last=True)
    val_loader = DataLoader(dataset=ZslDataset(superclass, 'valid'), batch_size=batch_size, pin_memory=True,
                            drop_last=True)

    embedding_size = get_embedding_size_by_superclass(superclass)
    print('embedding_size: ' + str(embedding_size))

    attributes_per_class = get_attributes_per_class_by_superclass(superclass)

    # Initialize encoder
    model = Encoder(embedding_size=embedding_size)

    # Use appropriate device
    model = model.to(device)

    # Initialize optimizers
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    best_acc = 0
    epochs_since_improvement = 0

    # Epochs
    for epoch in range(start_epoch, epochs):
        # Decay learning rate if there is no improvement for 8 consecutive epochs, and terminate training after 20
        if epochs_since_improvement == 20:
            break
        if epochs_since_improvement > 0 and epochs_since_improvement % 8 == 0:
            adjust_learning_rate(optimizer, 0.8)

        # One epoch's training
        train(epoch, train_loader, model, optimizer, attributes_per_class)

        # One epoch's validation
        val_acc, val_loss = valid(val_loader, model, attributes_per_class)
        print('\n * ACCURACY - {acc:.3f}, LOSS - {loss:.3f}\n'.format(acc=val_acc, loss=val_loss))

        # Check if there was an improvement
        is_best = val_acc > best_acc
        best_acc = max(best_acc, val_acc)

        if not is_best:
            epochs_since_improvement += 1
            print("\nEpochs since last improvement: %d\n" % (epochs_since_improvement,))
        else:
            epochs_since_improvement = 0

        # Save checkpoint
        save_checkpoint(epoch, model, optimizer, val_acc, is_best)


if __name__ == '__main__':
    # Parse arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-s", "--superclass",
                    help="superclass ('Animals', 'Fruits', 'Vehicles', 'Electronics', 'Hairstyles')")
    args = vars(ap.parse_args())

    main(args)
