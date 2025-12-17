
def add_model_args(parser):    
    # model parameters
    # parser.add_argument('--model_name', type = str, default = 'Neural_ODE',
    #                     choices = ['Vanilla_RNN', 'Vanilla_LSTM', 'Vanilla_GRU', 'Vanilla_Transformer', 'Neural_ODE', 'Latent_ODE', 'CT_Transformer'])
    #parser.add_argument('--dropout', type = float, default = 0.08)
    parser.add_argument('--input_mechanism', type = str, default = 'Channel_independent',
                        choices = ['Channel_independent', 'Channel_mixing'])
    parser.add_argument('--instance_norm', type = str, default = 'norm_max',
                        choices = ['norm_2', 'norm_max', 'without'])

    # RNN parameters
    parser.add_argument('--enc_in', type = int, default = 2, help = 'encoder input size')
    parser.add_argument('--rnn_i', type = int, default = 64, help = 'rnn input size')
    parser.add_argument('--rnn_h', type = int, default = 64, help = 'rnn hidden size')
    parser.add_argument('--rnn_layers', type = int, default = 2, help = 'rnn layer number')
    
    # ode parameters
    parser.add_argument('--ode_h', type = int, default = 64, help = 'decoder ode hidden size')
    parser.add_argument('--ode_hidden_layers', type = int, default = 2, help = 'decoder ode hidden layer number')    
    parser.add_argument('--ode_rnn_type', type = str, default = 'GRU',
                        choices = ['RNN', 'LSTM', 'GRU'])
    parser.add_argument('--dec_rtol', type = float, default = 1e-3)
    parser.add_argument('--dec_atol', type = float, default = 1e-4)
    parser.add_argument('--dec_method', type = str, default = 'rk4',
                        choices = ['euler', 'rk4', 'dopri5'])
    
    return parser




