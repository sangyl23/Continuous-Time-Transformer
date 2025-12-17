
def add_training_args(parser):
    # training parameters
    parser.add_argument('--training_time', type = int, default = 1)
    parser.add_argument('--epoch_num', type = int, default = 300)
    parser.add_argument('--batch_size', type = int, default = 1)
    parser.add_argument('--gpu', type = str, default = 'cuda:0')
    parser.add_argument('--loss_function', type = str, default = 'NMSE',
                        choices = ['MSE', 'NMSE'])
    parser.add_argument('--lr', type = float, default = 5e-4)
    parser.add_argument('--seed', type = int, default = 7)
       
    return parser




