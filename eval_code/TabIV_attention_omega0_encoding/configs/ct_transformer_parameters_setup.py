
def add_model_args(parser):
    # model parameters
    # parser.add_argument('--model_name', type = str, default = 'CT_Transformer',
    #                     choices = ['Vanilla_RNN', 'Vanilla_LSTM', 'Vanilla_GRU', 'Vanilla_Transformer', 'Neural_ODE', 'Latent_ODE', 'CT_Transformer'])
    parser.add_argument('--dropout', type = float, default = 0.08)
    parser.add_argument('--input_mechanism', type = str, default = 'Channel_independent',
                        choices = ['Channel_independent', 'Channel_mixing'])
    parser.add_argument('--instance_norm', type = str, default = 'without',
                        choices = ['norm_2', 'norm_max', 'without'])

    # Transformer parameters
    parser.add_argument('--time_emb_kind', type = str, default = 'HFTE',
                        choices = ['HFTE', 'PE', 'without'])
    parser.add_argument('--omega_0', type = float, default = 30.)
    parser.add_argument('--enc_ctsa', type = eval, default = False)
    parser.add_argument('--dec_ctsa', type = eval, default = True)
    parser.add_argument('--enc_in', type = int, default = 2, help = 'encoder input size')
    parser.add_argument('--dec_in', type = int, default = 2, help = 'decoder input size')
    parser.add_argument('--c_out', type = int, default = 2, help = 'output size')
    parser.add_argument('--label_len', type = int, default= 3, help = 'start token length of Informer decoder')
    parser.add_argument('--d_model', type = int, default = 64, help = 'dimension of model')
    parser.add_argument('--n_heads', type = int, default = 8, help = 'num of heads')
    parser.add_argument('--e_layers', type = int, default = 2, help = 'num of encoder layers')
    parser.add_argument('--d_layers', type = int, default = 2, help = 'num of decoder layers')
    parser.add_argument('--d_ff', type = int, default = 64, help ='dimension of fcn')
    parser.add_argument('--factor', type = int, default = 2, help ='probsparse attn factor')
    parser.add_argument('--distil', action = 'store_false', help = 'whether to use distilling in encoder', default = True)
    parser.add_argument('--attn', type = str, default = 'full', help = 'attention used in encoder, options:[prob, full]')
    parser.add_argument('--ct_attn', type = str, default = 'ct_full', help = 'CT attention kind, options:[ct_prob, ct_full]')
    parser.add_argument('--embed', type = str, default = 'fixed', help = 'time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type = str, default = 'gelu', help = 'activation')
    parser.add_argument('--output_attention', action = 'store_true', help = 'whether to output attention in ecoder')
    parser.add_argument(
        "--continuous_qkv_method",
        type = str,  
        nargs = 3,  
        default = ['interp', 'interp', 'ode'],  
        help = "A method list for extending q/k/v to continuous-time domain  (default: ['interp', 'interp', 'ode'])"
    )

    # ode parameters
    parser.add_argument('--dec_ode_h', type = int, default = 64, help = 'decoder ode hidden size')
    parser.add_argument('--dec_ode_hidden_layers', type = int, default = 0, help = 'decoder ode hidden layer number')
    parser.add_argument('--derivative_function_type', type = str, default = 'ConcatLinearNorm',
                        choices = ['ConcatLinear_v2', 'ConcatLinearNorm', 'ConcatSquashLinear'])
    parser.add_argument('--enc_rtol', type = float, default = 1e-3)
    parser.add_argument('--enc_atol', type = float, default = 1e-4)
    parser.add_argument('--enc_method', type = str, default = 'rk4',
                        choices = ['euler', 'rk4', 'dopri5'])
    parser.add_argument('--dec_rtol', type = float, default = 1e-3)
    parser.add_argument('--dec_atol', type = float, default = 1e-4)
    parser.add_argument('--dec_method', type = str, default = 'rk4',
                        choices = ['euler', 'rk4', 'dopri5'])
    
    return parser




