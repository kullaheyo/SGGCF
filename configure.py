CONFIG = {
    'name': '@likang',
    'path': './data',
    'log': './log',
    'visual': './visual',
    'gpu_id': "0",
    'note': 'some_note',
    'model': 'SGGCF',
    'dataset_name': 'Yelp',
    'task': 'tune',
    'eval_task': 'test',


    ## optimal hyperparameters 
    'lrs': [0.001],
    'message_dropouts': [0.2, 0.1, 0.05],
    'node_dropouts': [0],
    'decays': [1e-7, 1e-6],
    'alphas': [1.2, 1.1, 1, 0.9, 0.8],
    'cl_regs': [1e-4],
    'cl_2_regs': [1e-6],
    'cl_temps': [0.07],
    'cl_2_temps': [0.1],
    'drop_rates': [0.05],
    'aug_types': [4],


    ## hard negative sample and further train
    'sample': 'simple',
    #  'sample': 'hard',
    'hard_window': [0.7, 1.0], # top 30%
    'hard_prob': [0.4, 0.4], # probability 0.8
    'conti_train': 'model_file_from_simple_sample.pth',

    ## other settings
    'epochs': 80,
    'early': 50,
    'log_interval': 20,
    'test_interval': 1,
    'retry': 1,

    ## test path
    'test':['model_path_from_hard_sample']
}
