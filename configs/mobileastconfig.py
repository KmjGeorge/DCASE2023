mobileast_light1_config_silu = {
    # light1 使用silu   23.048 MMACs  52.960 K,  Quant: 23.804 MMACs  52.288 K,  Memory: 125.884765625 KB
    'spec_size': (256, 64),
    'kernel_size': (3, 3),
    'patch_size': (2, 2),
    'c1': 64,
    'c2': 64,
    'e1': 1,
    'e2': 1,
    'e3': 1,
    'mobileastblock': {
        'dim': 32,
        'heads': 4,
        'depth': 2,
        'mlp_dim': 64
    },
    'cout': 160,
    'act_func': 'silu'
}

mobileast_light1_config_relu = {
    # light1 使用relu 增大fc神经元数    23.507 MMACs 57.216 K,  Quant: 24.270 MMACs  56.488 K,  Memory: 125.47265625 KB
    'spec_size': (256, 64),
    'kernel_size': (3, 3),
    'patch_size': (2, 2),
    'c1': 64,
    'c2': 64,
    'e1': 1,
    'e2': 1,
    'e3': 1,
    'mobileastblock': {
        'dim': 32,
        'heads': 4,
        'depth': 2,
        'mlp_dim': 64
    },
    'cout': 216,
    'act_func': 'relu'
}

mobileast_light2_config_silu = {
    # light2 使用silu  26.914 MMACs  47.520 K  Quant: 27.867 MMACs  46.848 K  Memory: 124.609375 KB
    'spec_size': (256, 64),
    'kernel_size': (3, 3),
    'patch_size': (2, 2),
    'c1': 32,
    'c2': 32,
    'e1': 2,
    'e2': 1,
    'e3': 2,
    'mobileastblock': {
        'dim': 32,
        'heads': 4,
        'depth': 2,
        'mlp_dim': 64
    },
    'cout': 160,
    'act_func': 'silu'
}

mobileast_light2_config_relu = {
    # light2 使用relu 增大fc神经元数   27.439 MMACs  52.384 K   Quant: 28.400 MMACs   51.648 K,  Memory: 123.654296875 KB
    'spec_size': (256, 64),
    'kernel_size': (3, 3),
    'patch_size': (2, 2),
    'c1': 32,
    'c2': 32,
    'e1': 2,
    'e2': 1,
    'e3': 2,
    'mobileastblock': {
        'dim': 32,
        'heads': 4,
        'depth': 2,
        'mlp_dim': 64
    },
    'cout': 224,
    'act_func': 'relu'
}

mobileast_light3_config_silu = {  # light3 使用silu  12.350 MMACs 53.040K  Quant: 12.870 MMACs 52.592K  Memory: 125.375 KB
    'spec_size': (256, 64),
    'kernel_size': (3, 3),
    'patch_size': (2, 2),
    'c1': 32,
    'c2': 32,
    'e1': 1,
    'e2': 1,
    'e3': 1,
    'mobileastblock': {
        'dim': 48,
        'heads': 4,
        'depth': 2,
        'mlp_dim': 64
    },
    'cout': 128,
    'act_func': 'silu'
}

mobileast_light3_config_relu = {  # light3 使用relu 增大fc神经元数   12.875 MMACs 57.904 K   Quant: 13.403 MMACs  57.392 K  Memory: 125.650390625 KB
    'spec_size': (256, 64),
    'kernel_size': (3, 3),
    'patch_size': (2, 2),
    'c1': 32,
    'c2': 32,
    'e1': 1,
    'e2': 1,
    'e3': 1,
    'mobileastblock': {
        'dim': 48,
        'heads': 4,
        'depth': 2,
        'mlp_dim': 64
    },
    'cout': 192,
    'act_func': 'relu'
}

# 选定config
mobileast_light_config = mobileast_light1_config_relu
