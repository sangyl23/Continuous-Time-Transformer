
def add_model_args(parser):    
    # model parameters
    # parser.add_argument('--model_name', type = str, default = 'Vanilla_GRU',
    #                     choices = ['Vanilla_RNN', 'Vanilla_LSTM', 'Vanilla_GRU', 'Vanilla_Transformer', 'Neural_ODE', 'Latent_ODE', 'CT_Transformer'])
    # parser.add_argument('--dropout', type = float, default = 0.08)
    parser.add_argument('--input_mechanism', type = str, default = 'Channel_independent',
                        choices = ['Channel_independent', 'Channel_mixing'])
    parser.add_argument('--instance_norm', type = str, default = 'norm_max',
                        choices = ['norm_2', 'norm_max', 'without'])

    # RNN parameters
    parser.add_argument('--enc_in', type = int, default = 2, help = 'encoder input size')
    parser.add_argument('--rnn_i', type = int, default = 64, help = 'rnn input size')
    parser.add_argument('--rnn_h', type = int, default = 64, help = 'rnn hidden size')
    parser.add_argument('--rnn_layers', type = int, default = 2, help = 'rnn layer number')
    
    return parser




