import argparse


kv_dict = {
    # 현재 연결하려는 파일 명 : 식별 이름
    'exp_vis_anl_motion_reconstruction_all.py': 'anl_m_recons_addon',
    'exp_vis_k_hot_code_change_with_loss_by_frame.py': 'vis_code_with_loss',
    'exp_anl_motion_editing_v2.py': 'edit_motion'
}

def get_args_parser(exec_path):
    parser = argparse.ArgumentParser(description='mode selection parameters',
                                     add_help=True,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    id = kv_dict[exec_path]
    

    #### 추가 정의
    if id == 'anl_m_recons_addon':
        parser.add_argument('--output-loss-list', action='store_true', help='여러개 loss 출력할지 안할지') # 여러개 loss 출력할지 안할지
        parser.add_argument('--save-plot', action='store_true', help='') # 여러개 loss 출력할지 안할지
        parser.add_argument('--save-folder-path', type=str, required=True, help='') # 여러개 loss 출력할지 안할지
        parser.add_argument('--val-shuffle', action='store_true', help='') # 여러개 loss 출력할지 안할지
        parser.add_argument('--split', choices=['train', 'vaild', 'test'], required=True, help='') # 여러개 loss 출력할지 안할지
        parser.add_argument('--vel-loss-mode', choices=['v1', 'v2'], required=True, help='') # 여러개 loss 출력할지 안할지
    elif id == 'vis_code_with_loss':
        parser.add_argument('--save-folder-path', type=str, required=True, help='세이브 폴더 경로')
    elif id == 'edit_motion':
        parser.add_argument('--time-mode', choices=['all', 'even', 'odd', '20%', '50%'], required=True, help='편집할 시간대 비율')
        parser.add_argument('--edit-mode', choices=['replace_all', 'increase', 'decrease', 'reverse'], required=True, help='편집 모드')
        
    else:
        raise NotImplementedError()

    return parser.parse_args()


def print_parser_arguments(args):
    print("################### Selected Parameter List ###################")
    for key, value in vars(args).items():
        print(f"  {key}: {value}")