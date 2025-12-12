import argparse

def get_args_parser():
    parser = argparse.ArgumentParser(description='Optimal Transport AutoEncoder training for AIST',
                                     add_help=True,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    ## dataloader  
    parser.add_argument('--dataname', type=str, default='kit', help='dataset directory')
    parser.add_argument('--batch-size', default=128, type=int, help='batch size')
    parser.add_argument('--window-size', type=int, default=64, help='training motion length')

    ## optimization
    parser.add_argument('--total-iter', default=200000, type=int, help='number of total iterations to run')
    parser.add_argument('--warm-up-iter', default=1000, type=int, help='number of total iterations for warmup')
    parser.add_argument('--lr', default=2e-4, type=float, help='max learning rate')
    parser.add_argument('--lr-scheduler', default=[50000, 400000], nargs="+", type=int, help="learning rate schedule (iterations)")
    parser.add_argument('--gamma', default=0.05, type=float, help="learning rate decay")

    parser.add_argument('--weight-decay', default=0.0, type=float, help='weight decay')
    # parser.add_argument("--commit", type=float, default=0.02, help="hyper-parameter for the commitment loss")
    parser.add_argument("--delete-mode", type=str, default='delete', choices=['delete', 'trim', 'del_super_posecode'], help="지울 semantic code(full name)")
    parser.add_argument("--target-del-semantic-code", type=str, default=None, help="지울 semantic code(full name)")
    parser.add_argument("--codes-folder-name", type=str, default='codes', help="code foler 이름")
    # parser.add_argument('--loss-recons', type=float, default=1.0, help='hyper-parameter for the main reconstruction loss')
    # parser.add_argument('--loss-vel', type=float, default=0.1, help='hyper-parameter for the velocity loss')
    # parser.add_argument('--loss-ortho', type=float, default=0.0, help='hyper-parameter for the codebook orthogonal loss')
    # parser.add_argument('--loss-gw', type=float, default=0.0, help='hyper-parameter for the group wise code reconstruction loss') #
    # parser.add_argument('--loss-plv', type=float, default=0.0, help='hyper-parameter for pelvis joint loss')
    # parser.add_argument('--loss-jt', type=float, default=0.0, help='hyper-parameter for normal joint group loss')
    # parser.add_argument('--loss-cjt', type=float, default=0.0, help='hyper-parameter for contact joint group loss')
    # parser.add_argument('--loss-plv-vel', type=float, default=0.0, help='hyper-parameter for pelvis joint loss')
    # parser.add_argument('--loss-jt-vel', type=float, default=0.0, help='hyper-parameter for normal joint group loss')
    # parser.add_argument('--loss-cjt-vel', type=float, default=0.0, help='hyper-parameter for contact joint group loss')
    # parser.add_argument('--recons-loss', type=str, default='l2', help='reconstruction loss')
    # parser.add_argument('--code-recons-loss', type=str, default='l1_smooth', help='code recons reconstruction loss')
    # parser.add_argument('--use-dynamic-weight', action='store_true', help='dynamic parameter for loss')
    # parser.add_argument('--norm-dynamic-weight', action='store_true', help='normalize dynamic parameter for loss to summation one')
    parser.add_argument('--disable-vel-loss', action='store_true', help='not to use velocity loss')
    parser.add_argument('--disable-align-root', action='store_true', help='align root joint to calculate mpjpe')
    parser.add_argument('--loss-format', choices=['joint', 'h3d'], default='h3d', help='h3d format or joint feature format')
    parser.add_argument('--loss-cfg-path', type=str, help='loss config file path')
    
    ## encdec arch
    parser.add_argument("--code-dim", type=int, default=512, help="embedding dimension")
    parser.add_argument("--nb-code", type=int, default=512, help="nb of embedding")
    parser.add_argument("--mu", type=float, default=0.99, help="exponential moving average to update the codebook")
    parser.add_argument("--down-t", type=int, default=2, help="downsampling rate")
    parser.add_argument("--stride-t", type=int, default=2, help="stride size")
    parser.add_argument("--width", type=int, default=512, help="width of the network")
    parser.add_argument("--depth", type=int, default=3, help="depth of the network")
    parser.add_argument("--dilation-growth-rate", type=int, default=3, help="dilation growth rate")
    parser.add_argument("--output-emb-width", type=int, default=512, help="output embedding width")
    parser.add_argument('--vq-act', type=str, default='relu', choices = ['relu', 'silu', 'gelu'], help='dataset directory')
    parser.add_argument('--vq-norm', type=str, default=None, help='dataset directory')
    
    ## resume
    parser.add_argument("--resume-pth", type=str, default=None, help='resume pth for VQ')
    parser.add_argument("--resume-gpt", type=str, default=None, help='resume pth for GPT')
    
    
    ## output directory 
    parser.add_argument('--out-dir', type=str, default='output_vqfinal/', help='output directory')
    parser.add_argument('--results-dir', type=str, default='visual_results/', help='output directory')
    parser.add_argument('--visual-name', type=str, default='baseline', help='output directory')
    parser.add_argument('--exp-name', type=str, default='exp_debug', help='name of the experiment, will create a file inside out-dir')
    
    ## other
    parser.add_argument('--print-iter', default=200, type=int, help='print frequency')
    parser.add_argument('--eval-iter', default=1000, type=int, help='evaluation frequency')
    parser.add_argument('--seed', default=123, type=int, help='seed for initializing training.')
    
    parser.add_argument('--vis-gt', action='store_true', help='whether visualize GT motions')
    parser.add_argument('--nb-vis', default=20, type=int, help='nb of visualizations')
    parser.add_argument('--num_workers', default=4, type=int, help='number of workers')


    parser.add_argument('--use-keywords', action='store_true', help='whether to use fine-grained keyword')
    parser.add_argument('--use-word-only', action='store_true', help='whether to use POS in word embedding')
    parser.add_argument('--cfg-cla-path', type=str, default=None, help='Configuration file path of category aggregator')
    parser.add_argument('--use-full-sequence', action='store_true', help='whether to use full sequence of motion sample in training')
    parser.add_argument('--val-shuffle', action='store_true', help='whether to shuffle validation set')
    parser.add_argument('--vel-loss-mode', choices=['v1', 'v2', 'v3'], default='v1', help='original vel loss or fixed vel loss')
    parser.add_argument('--meta-dir', type=str, default=None, help='meta file directory')
    parser.add_argument('--aggregate-mode', type=str, default=None, choices=[None, 'hadamard'])
    parser.add_argument('--randomize-window-size', action='store_true', help='randomize window size')


    parser.add_argument('--use-rvq', action='store_true', help='RVQ를 사용할지 여부 (사용 시 True)')
    parser.add_argument('--rvq-name', type=str, default='rptc')
    parser.add_argument('--rvq-num-quantizers', type=int, default=3)
    parser.add_argument('--rvq-shared-codebook', action='store_true')
    parser.add_argument('--rvq-quantize-dropout-prob', type=float, default=0.2)
    parser.add_argument('--rvq-quantize-dropout-cutoff-index', type=int, default=0)
    parser.add_argument('--rvq-nb-code', type=int, default=64)
    parser.add_argument('--rvq-mu', type=float, default=0.99)
    parser.add_argument('--rvq-commit', type=float, default=0.02)
    parser.add_argument('--rvq-resi-beta', type=float, default=1.0)
    parser.add_argument('--detach-p-latent', action='store_true')

    parser.add_argument('--rvq-vq-loss-beta', type=float, default=1.0)
    parser.add_argument('--rvq-quantizer-type', type=str, default='hard', choices=['hard', 'soft'])
    parser.add_argument('--params-soft-ent-loss', type=float, default=0.0)
    parser.add_argument('--unuse-ema', action='store_true', help='disable ema when using soft quantizer')
    parser.add_argument('--rvq-init-method', type=str, default='enc', choices=['enc', 'uniform', 'xavier'], help="codebook initialization method for RVQ")
    parser.add_argument('--pdrop-res', type=float, default=0.0, help='residual quantization을 drop할 확률 (0.0 ~ 1.0)')
    parser.add_argument("--dec-checkpoint-folder", type=str, default=None, help='./')
    parser.add_argument('--force-drop-residual-quantization', action='store_true', help='force drop residual quantization')
    parser.add_argument('--rvq-gated-residual', action='store_true', help='gated residual 사용 여부')
    
    parser.add_argument('--fcm-dsl-init-sigma', type=float, default=1.0,help='Decoder Side Lambda 초기 Gaussian sigma 값')
    parser.add_argument('--fcm-gaussian-blur', action='store_true', help='whether to use gaussian blur in FCM')

    return parser.parse_args()