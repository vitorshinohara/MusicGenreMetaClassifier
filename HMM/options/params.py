hmm_params = dict(
        n_components=[3, 5, 7, 9, 11],
        covariance_type=['full', 'diag'],
        n_iter=[100, 150, 200, 250, 300]
)

feature_params = dict(
    frames=['5', '20', '40'],
    algorithms=['linspace'],
    hs=['8', '26', '75', '100'],
    folds_number=10
)

path = 'data/homburg-ds_gtzan-feats_frames'
