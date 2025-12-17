
def add_channel_system_args(parser):    
    # channel parameters
    parser.add_argument('--channel_type', type = str, default = 'DeepMIMO_O1',
                        choices = ['3GPP_38901', 'CDL-B', 'DeepMIMO_O1'])
    parser.add_argument('--training_samples', type = str, default = '90mats',
                        choices = ['50mats', '90mats', '100mats', '150mats'])
    parser.add_argument('--UE_velocity', type = str, default = '60kmh',
                        choices = ['20kmh', '40kmh', '60kmh', '80kmh', '100kmh'])
    parser.add_argument('--snr', type = str, default = '10dB',
                        choices = ['0dB', '5dB', '10dB', '15dB'])

    # pilot parameters
    parser.add_argument('--his_len', type = int, default = 8)
    parser.add_argument('--pre_len', type = int, default = 8)
    parser.add_argument('--T_his', type = float, default = 40.)
    parser.add_argument('--T_pre', type = float, default = 2.)
    parser.add_argument('--ifintepolate_pilot', type = eval, default = True)
    parser.add_argument('--intepolate_pilot_method', type = str, default = 'chebyshev',
                        choices = ['chebyshev', 'uniform', 'random', 'doppler'])
    parser.add_argument('--intepolation_points_num', type = int, default = 3)
    
    return parser




