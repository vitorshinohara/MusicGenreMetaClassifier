
class ParamParser:

    def __init__(self, path):
        self.path = path
        self.hmm_params = self.get_hmm_params()
        self.feature_params = self.get_feature_params()
        self.filename_pattern = self.get_filename_pattern()

    def get_hmm_params(self):
        return dict(
                    n_components=[3, 5, 7, 9, 11],
                    covariance_type=['full', 'diag'],
                    n_iter=[100, 150, 200, 250, 300]
                )

    def get_feature_params(self):
        hs = None
        if 'mel-feats' in self.path:
            hs = ['128']
        elif 'gtzan-feats' in self.path:
            hs = ['']
        else:
            hs = ['8', '26', '75', '100']

        return dict(
                   frames=['5', '20', '40'],
                   hs=hs,
                   folds_number= 3 if 'lmd-ds' in self.path else 10,
                )

    def get_filename_pattern(self):
        if 'mel-feats' in self.path or 'rp-feats' in self.path:
            return '{}{}-nf_linspace-fs_{}-nfeats_none-nl_True-noanova_True-delta_{}-f.predict.frames.txt'
        elif 'gtzan-feats' in self.path:
            return '{}{}-nf_linspace-fs_{}{}-f.predict.frames.txt'
        else:
            return '{}{}-nf_linspace-fs_{}-hs_{}-f.predict.frames.txt'