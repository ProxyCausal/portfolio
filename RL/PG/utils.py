import logging
import os

LOG_DIR = "logs"

class Logger(object):
    def __init__(self, fname, seed, model=None):
        """Stores scores and saves models

        Args:
            model make this multiple args?
        """
        self.seed = seed
        logging.basicConfig(
            filename=os.path.join(LOG_DIR, os.path.splitext(os.path.basename(__file__))[0]),
            level=logging.INFO)
    
    def save_score(self, score, e):
        logging.info("Seed {}: Score of {} in {} epochs".format(self.seed, score, e))

    def save_model(self):
        pass

def add_arguments(parser):
    parser.add_argument('--train', type=bool, default=True, help="Train or load model")
    parser.add_argument('--env', type=str, default='CartPole-v1', help="Environment to train on")
    parser.add_argument('--epochs', type=int, default=100, help="Number of epochs to train for")
    parser.add_argument('--seed', type=int, default=None, help="Seed")
    parser.add_argument('--target', type=int, help="Stop training when target is reached")
    parser.add_argument('--repeat', type=int, default=1, help="Repeat training x times to compare results")
    parser.add_argument('--batch-size', type=int, default=5000, help="Batch size for updates")
    parser.add_argument('--save-every', type=int, default=1, help="Save model every x epochs")