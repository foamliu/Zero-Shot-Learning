import time

from torch import nn
from torch import optim
from torch.utils.data import DataLoader

from config import *
from data_gen import ZslDataset
from models import Encoder
from utils import *


def train(epoch, train_loader, model, optimizer):
    # Ensure dropout layers are in train mode
    model.train()

    # Loss function
    criterion = nn.MSELoss().to(device)

    batch_time = AverageMeter()  # forward prop. + back prop. time
    losses = AverageMeter()  # loss (per word decoded)
    accs = AverageMeter()  # accuracy

    start = time.time()

    # Batches
    for i_batch, (imgs, label_ids, attributes) in enumerate(train_loader):
        # Zero gradients
        optimizer.zero_grad()

        # Set device options
        imgs = imgs.to(device)
        # print(img.size())
        label_ids = label_ids.to(device)
        print(label_ids.size())
        attributes = attributes.to(device)
        # print('targets: ' + str(targets))
        # print('targets.size(): ' + str(targets.size()))

        preds = model(imgs)
        _, scores = batched_KNN(preds, 1)
        # print('scores: ' + str(scores))
        print('scores.size(): ' + str(scores.size()))

        loss = criterion(preds, attributes)
        loss.backward()

        optimizer.step()

        acc = accuracy(scores, label_ids)

        # Keep track of metrics
        losses.update(loss.item(), batch_size)
        batch_time.update(time.time() - start)
        accs.update(acc, batch_size)

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


def valid(val_loader, model):
    model.eval()  # eval mode (no dropout or batchnorm)

    # Loss function
    criterion = nn.MSELoss().to(device)

    batch_time = AverageMeter()  # forward prop. + back prop. time
    losses = AverageMeter()  # loss (per word decoded)
    accs = AverageMeter()  # accuracy

    start = time.time()

    with torch.no_grad():
        # Batches
        for i_batch, (imgs, label_ids, attributes) in enumerate(val_loader):
            # Set device options
            imgs = imgs.to(device)
            label_ids = label_ids.to(device)
            attributes = attributes.to(device)  # (batch_size, 123)

            preds = model(imgs)  # (batch_size, 123)

            loss = criterion(preds, attributes)

            _, scores = batched_KNN(preds, 1)
            acc = accuracy(scores, label_ids)

            # Keep track of metrics
            losses.update(loss.item())
            batch_time.update(time.time() - start)
            accs.update(acc, batch_size)

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

    return losses.avg


def main():
    train_loader = DataLoader(dataset=ZslDataset('Animals', 'train'), batch_size=batch_size, shuffle=True,
                              pin_memory=True, drop_last=True)
    val_loader = DataLoader(dataset=ZslDataset('Animals', 'valid'), batch_size=batch_size, pin_memory=True,
                            drop_last=True)

    # Initialize encoder
    model = Encoder(embedding_size=123)

    # Use appropriate device
    model = model.to(device)

    # Initialize optimizers
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Epochs
    for epoch in range(start_epoch, epochs):
        # One epoch's training

        train(epoch, train_loader, model, optimizer)

        val_loss = valid(val_loader, model)
        print('\n * LOSS - {loss:.3f}\n'.format(loss=val_loss))


if __name__ == '__main__':
    main()
