import argparse


def get_args():
    parser = argparse.ArgumentParser(description="IRRA Args")
    ######################## mode ########################
    parser.add_argument("--simplified", default=False, action='store_true')
    parser.add_argument("--resume", default=False, action='store_true',
                        help="Resume training from checkpoint")

    args = parser.parse_args()

    return args