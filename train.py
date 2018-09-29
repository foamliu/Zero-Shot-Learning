from torch import optim
import time
from config import *
from models import Encoder


def main():
    # Initialize encoder
    encoder = Encoder()

    # Use appropriate device
    encoder = encoder.to(device)

    # Initialize optimizers
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)

    # Epochs
    for epoch in range(start_epoch, epochs):
        # One epoch's training
        # Ensure dropout layers are in train mode
        encoder.train()

        start = time.time()




if __name__ == '__main__':
    main()
