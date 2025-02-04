import torch
import argparse
from vpnet import *
from ecg_tools import *

p = lambda lst: '-'.join(map(str, lst))

def vpnet_hermite(batch_size, lr, epoch, params_init, n_vp, n_hiddens, vp_penalty):
    name = f'vpnet-hermite_b{batch_size}_lr{lr}_e{epoch}_p{p(params_init)}_n{n_vp}_h{p(n_hiddens)}_r{vp_penalty}'
    model = VPNet(n_in, 0, n_channels, n_vp, VPTypes.FEATURES, params_init,
                  HermiteSystem(n_in, n_vp), n_hiddens, n_out,
                  device=device, dtype=dtype)
    criterion = VPLoss(torch.nn.CrossEntropyLoss(), vp_penalty)
    return name, model, criterion

def vpnet_hermite_rr(batch_size, lr, epoch, params_init, n_vp, n_hiddens, vp_penalty):
    name = f'vpnet-hermite-rr_b{batch_size}_lr{lr}_e{epoch}_p{p(params_init)}_n{n_vp}_h{p(n_hiddens)}_r{vp_penalty}'
    model = VPNet(n_in, n_rr, n_channels, n_vp, VPTypes.FEATURES, params_init,
                  HermiteSystem(n_in, n_vp), n_hiddens, n_out,
                  device=device, dtype=dtype)
    criterion = VPLoss(torch.nn.CrossEntropyLoss(), vp_penalty)
    return name, model, criterion

def vpnet_hermite2(batch_size, lr, epoch, params_init, n_vp, n_hiddens, vp_penalty):
    name = f'vpnet-hermite2_b{batch_size}_lr{lr}_e{epoch}_p{p(params_init)}_n{n_vp}_h{p(n_hiddens)}_r{vp_penalty}'
    model = VPNet(n_in, 0, n_channels, n_vp, VPTypes.FEATURES, params_init,
                  HermiteSystem2(n_in, n_vp), n_hiddens, n_out,
                  device=device, dtype=dtype)
    criterion = VPLoss(torch.nn.CrossEntropyLoss(), vp_penalty)
    return name, model, criterion

def vpnet_hermite2_rr(batch_size, lr, epoch, params_init, n_vp, n_hiddens, vp_penalty):
    name = f'vpnet-hermite2-rr_b{batch_size}_lr{lr}_e{epoch}_p{p(params_init)}_n{n_vp}_h{p(n_hiddens)}_r{vp_penalty}'
    model = VPNet(n_in, n_rr, n_channels, n_vp, VPTypes.FEATURES, params_init,
                  HermiteSystem2(n_in, n_vp), n_hiddens, n_out,
                  device=device, dtype=dtype)
    criterion = VPLoss(torch.nn.CrossEntropyLoss(), vp_penalty)
    return name, model, criterion

def vpunfold_hermite2(batch_size, lr, epoch, params_init, weight_init, weight_learn, n_vp, n_iterations,
                      n_iter_hiddens, n_hiddens, vp_penalty):
    name = f'vpunfold-hermite2_b{batch_size}_lr{lr}_e{epoch}_p{p(params_init)}_w{weight_init}_wl{weight_learn}_n{n_vp}_i{n_iterations}_ih{p(n_iter_hiddens)}_h{p(n_hiddens)}_r{vp_penalty}'
    model = VPDeepUnfold(n_in, 0, n_channels, n_vp, VPTypes.FEATURES,
                         params_init, weight_init, weight_learn,
                         HermiteSystem2(n_in, n_vp), n_iterations, n_iter_hiddens, n_hiddens, n_out,
                         device=device, dtype=dtype)
    criterion = VPLoss(torch.nn.CrossEntropyLoss(), vp_penalty)
    return name, model, criterion

def vpunfold_hermite2_rr(batch_size, lr, epoch, params_init, weight_init, weight_learn, n_vp, n_iterations,
                         n_iter_hiddens, n_hiddens, vp_penalty):
    name = f'vpunfold-hermite2-rr_b{batch_size}_lr{lr}_e{epoch}_p{p(params_init)}_w{weight_init}_wl{weight_learn}_n{n_vp}_i{n_iterations}_ih{p(n_iter_hiddens)}_h{p(n_hiddens)}_r{vp_penalty}'
    model = VPDeepUnfold(n_in, n_rr, n_channels, n_vp, VPTypes.FEATURES,
                         params_init, weight_init, weight_learn,
                         HermiteSystem2(n_in, n_vp), n_iterations, n_iter_hiddens, n_hiddens, n_out,
                         device=device, dtype=dtype)
    criterion = VPLoss(torch.nn.CrossEntropyLoss(), vp_penalty)
    return name, model, criterion

if __name__ == '__main__':
    # Arguments
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('dataset',
                        help='ECG dataset selector. Choices:\n'
                             '  balanced - Balanced subset of N and VEB (data/ecg_train.mat and data/ecg_test.mat)\n'
                             '  full - Full database, 5 class (data/mitdb_filt35_w300adapt_ds1_float.mat and *_ds2_float.mat)')
    parser.add_argument('-e', '--eval', action='append',
                        help='Model selector, multiple choices allowed. Choices:\n'
                        '  vpnet-hermite          Hermite VPNet\n'
                        '  vpnet-hermite-rr       Hermite VPNet + RR\n'
                        '  vpnet-hermite2         HermiteV2 VPNet\n'
                        '  vpnet-hermite2-rr      HermiteV2 VPNet + RR\n'
                        '  vpunfold-hermite2      HermiteV2 VPDeepUnfold \n'
                        '  vpunfold-hermite2-rr   HermiteV2 VPDeepUnfold + RR\n'
                        '(RR versions are supported with full dataset only)')
    args = parser.parse_args()

    # Data types
    dtype = torch.float
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # torch.use_deterministic_algorithms(True, warn_only=True)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
    random_init()

    # Dataset
    if args.dataset == 'full':
        train_set = ecg_dataset('data/mitdb_filt35_w300adapt_ds1_float.mat', device=device, dtype=dtype)
        test_set = ecg_dataset('data/mitdb_filt35_w300adapt_ds2_float.mat', device=device, dtype=dtype)
    elif args.dataset == 'balanced':
        train_set = ecg_dataset('data/ecg_train.mat', device=device, dtype=dtype)
        test_set = ecg_dataset('data/ecg_test.mat', device=device, dtype=dtype)
    else:
        raise argparse.ArgumentError(f'Unknown argument for dataset: {args.dataset}')

    samples0, rr0, labels0 = train_set[0]
    n_channels, n_in = samples0.shape
    _, n_rr = rr0.shape
    n_out, = labels0.shape

    # Configs
    configs = {
        'vpnet-hermite': {
            'model_builder': vpnet_hermite,
            'args': {
                'batch_size': [4096], 'lr': [1e-2, 1e-3], 'epoch': [50],
                'params_init': [[1.0, 0.0]],
                'n_vp': [4, 8, 16],
                'n_hiddens': [[8], [16], [8,8], [16,16]],
                'vp_penalty': [0.0, 0.1]
            }
        },
        'vpnet-hermite-rr': {
            'model_builder': vpnet_hermite_rr,
            'args': {
                'batch_size': [4096], 'lr': [1e-2, 1e-3], 'epoch': [50],
                'params_init': [[1.0, 0.0]],
                'n_vp': [4, 8, 16],
                'n_hiddens': [[8], [16], [8,8], [16,16]],
                'vp_penalty': [0.0, 0.1]
            }
        },
        'vpnet-hermite2': {
            'model_builder': vpnet_hermite2,
            'args': {
                'batch_size': [4096], 'lr': [1e-2, 1e-3], 'epoch': [50],
                'params_init': [[1.0, 0.0]],
                'n_vp': [4, 8, 16],
                'n_hiddens': [[8], [16], [8,8], [16,16]],
                'vp_penalty': [0.0, 0.1]
            }
        },
        'vpnet-hermite2-rr': {
            'model_builder': vpnet_hermite2_rr,
            'args': {
                'batch_size': [4096], 'lr': [1e-2, 1e-3], 'epoch': [50],
                'params_init': [[1.0, 0.0]],
                'n_vp': [4, 8, 16],
                'n_hiddens': [[8], [16], [8,8], [16,16]],
                'vp_penalty': [0.0, 0.1]
            }
        },
        'vpunfold-hermite2': {
            'model_builder': vpunfold_hermite2,
            'args': {
                'batch_size': [4096], 'lr': [1e-2, 1e-3], 'epoch': [50],
                'params_init': [[1.0, 0.0]],
                'weight_init': [1e-3, 1e-4],
                'weight_learn': [0],
                'n_vp': [4, 8, 16],
                'n_iterations': [1, 2, 3],
                'n_iter_hiddens': [[8], [16]],
                'n_hiddens': [[8], [16]],
                'vp_penalty': [0.0, 0.1]
            }
        },
        'vpunfold-hermite2-rr': {
            'model_builder': vpunfold_hermite2_rr,
            'args': {
                'batch_size': [4096], 'lr': [1e-2, 1e-3], 'epoch': [50],
                'params_init': [[1.0, 0.0]],
                'weight_init': [1e-3, 1e-4],
                'weight_learn': [0],
                'n_vp': [4, 8, 16],
                'n_iterations': [1, 2, 3],
                'n_iter_hiddens': [[8], [16]],
                'n_hiddens': [[8], [16]],
                'vp_penalty': [0.0, 0.1]
            }
        },
    }

    # Model evaluations
    if args.eval is None:
        raise argparse.ArgumentError(f'Missing model(s) --eval')
    warnings = ''
    best_models = {}
    for model in list(set(args.eval)):
        if model not in configs:
            warnings += f'Warning: unknown model {model}\n'
            continue
        if model.endswith('-rr') and args.dataset == 'balanced':
            warnings += f'Warning: unsupported model {model} for dataset balanced\n'
            continue
        best_models[model] = evaluate_models(configs[model]['model_builder'],
                                             train_set, test_set,
                                             **configs[model]['args'])

    print(warnings)
    if best_models:
        print('Best models:')
        for model in list(set(args.eval)):
            if model in best_models:
                accuracy, name = best_models[model]
                print(f'{accuracy:.2f}% {name}')
