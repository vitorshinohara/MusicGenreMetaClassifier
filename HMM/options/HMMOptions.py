import argparse

class HMMOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument('--input', default='data/gtzan-ds_rp-feats_frames', type=str, help='Input folder to extract data em classify')

    def get_parser(self):
        return self.parser