
def add_model_args(parser):
    # model parameters
    # parser.add_argument('--model_name', type = str, default = 'Vanilla_Transformer',
    #                     choices = ['Vanilla_RNN', 'Vanilla_LSTM', 'Vanilla_GRU', 'Vanilla_Transformer', 'Neural_ODE', 'Latent_ODE', 'CT_Transformer'])
    parser.add_argument('--dropout', type = float, default = 0.08)
    parser.add_argument('--input_mechanism', type = str, default = 'Channel_independent',
                        choices = ['Channel_independent', 'Channel_mixing'])
    parser.add_argument('--instance_norm', type = str, default = 'without',
                        choices = ['norm_2', 'norm_max', 'without'])

    # Transformer parameters
    parser.add_argument('--enc_in', type = int, default = 2, help = 'encoder input size')
    parser.add_argument('--dec_in', type = int, default = 2, help = 'decoder input size')
    parser.add_argument('--c_out', type = int, default = 2, help = 'output size')
    parser.add_argument('--label_len', type = int, default= 3, help = 'start token length of Informer decoder')
    parser.add_argument('--d_model', type = int, default = 64, help = 'dimension of model')
    parser.add_argument('--n_heads', type = int, default = 8, help = 'num of heads')
    parser.add_argument('--e_layers', type = int, default = 2, help = 'num of encoder layers')
    parser.add_argument('--d_layers', type = int, default = 2, help = 'num of decoder layers')
    parser.add_argument('--d_ff', type = int, default = 64, help ='dimension of fcn')
    parser.add_argument('--factor', type = int, default = 5, help ='probsparse attn factor')
    parser.add_argument('--distil', action = 'store_false', help = 'whether to use distilling in encoder', default = True)
    parser.add_argument('--attn', type = str, default = 'full', help = 'attention used in encoder, options:[prob, full]')
    parser.add_argument('--embed', type = str, default = 'fixed', help = 'time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type = str, default = 'gelu', help = 'activation')
    parser.add_argument('--output_attention', action = 'store_true', help = 'whether to output attention in ecoder')
    
    return parser