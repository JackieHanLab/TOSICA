import argparse, os
from datetime import datetime
import logging


logger = logging.getLogger()
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(filename)s %(lineno)d: %(message)s',
                    datefmt='%m-%d %H:%M:%S')
DATE_TIME = "%Y_%m_%d %H:%M:%S"


class ArgparseUtil(object):
    """
    参数解析工具类
    """
    def __init__(self):
        """ Basic args """
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument("--seed", default=1, type=int)
        self.parser.add_argument('--gpu_device_id', default=0, type=int, 
                                 help='the GPU NO. Use int because compare with gpu total count')
        self.parser.add_argument("--model_type", default='cnn', type=str)
        self.parser.add_argument("--data_type", default=None, type=str)
        self.parser.add_argument("--save_predicted_test", default=0, type=int, help="0 is false, 1 is true")
        self.parser.add_argument("--save_model_per_epoch", type=int, default=0, help="0 is false, 1 is true")
        self.parser.add_argument("--predict_on_separate_length", type=int, default=0, help="0 is false, 1 is true")
        self.parser.add_argument("--plot_proba_hist_kde", type=int, default=0, help="0 is false, 1 is true")

    def peptimizer(self):
        """ task args """
        self.parser.add_argument("--task_name", type=str, default='cpp', help="anti_inflammation, toxic")
        self.parser.add_argument("--enable_train", type=int, default=0, help="0 is false, 1 is true")
        self.parser.add_argument("--embedding_type", type=str, default='', help="vocab, topology")
        self.parser.add_argument("--epochs", type=int, default=300, help="")
        self.parser.add_argument("--warmup_steps", type=float, default=0.0,
            help="if warmup_steps < 1, it is used as int(warmup_ratio * max_interation_num) ")
        self.parser.add_argument("--learning_rate", type=float, default=0.0005, help="")
        self.parser.add_argument("--dropout", type=float, default=0.5, help="")
        self.parser.add_argument("--batch_size", type=int, default=50, help="")
        self.parser.add_argument("--val_split", type=float, default=0.2, help="")
        self.parser.add_argument("--data_split_seed", type=int, default=13, help="")
        args = self.parser.parse_args()
        return args

    def classifier_rnn(self):
        """ task args """
        self.parser.add_argument("--task_name", type=str, default='', help="anti_inflammation, toxic")
        self.parser.add_argument("--enable_train", type=int, default=1, help="0 is false, 1 is true")
        self.parser.add_argument("--embedding_type", type=str, default='', help="vocab, esm, t5")
        self.parser.add_argument("--read_dataset_cache", type=int, default=1, help="0 is false, overwite; 1 true")
        self.parser.add_argument("--basic_model", type=str, default='lstm', help="lstm, gru")
        self.parser.add_argument("--n_epoch", type=int, default=150, help="")
        self.parser.add_argument("--learning_rate", type=float, default=0.01, help="")
        self.parser.add_argument("--dropout", type=float, default=0.5, help="")
        self.parser.add_argument("--use_fixed_hyparameter", type=int, default=1, help="0 is false, 1 is true")
        self.parser.add_argument("--overwrite_hyperparameter_dict", type=int, default=0, help="0 is false, 1 is true")
        self.parser.add_argument(
            "--transformer_model_path", type=str,
            default="/mnt/sda/models/Rostlab_prot_t5_xl_uniref50",
            # default="/mnt/sda/models/Rostlab prot_bert_bfd"
        )
        args = self.parser.parse_args()
        return args

    def classifier_cnn(self):
        """ task args """
        self.parser.add_argument("--task_name", type=str, default='', help="anti_inflammation, toxic")
        self.parser.add_argument("--enable_train", type=int, default=1, help="0 is false, 1 is true")
        self.parser.add_argument("--embedding_type", type=str, default='', help="vocab, esm, t5")
        self.parser.add_argument("--read_dataset_cache", type=int, default=1, help="0 is false, overwite; 1 true")
        self.parser.add_argument("--basic_model", type=str, default='cnn', help="")
        self.parser.add_argument("--n_epoch", type=int, default=300, help="")
        self.parser.add_argument("--learning_rate", type=float, default=0.0005, help="")
        self.parser.add_argument("--dropout", type=float, default=0.5, help="")
        self.parser.add_argument("--use_fixed_hyparameter", type=int, default=1, help="0 is false, 1 is true")
        self.parser.add_argument(
            "--transformer_model_path", type=str,
            default="/mnt/sda/models/Rostlab_prot_t5_xl_uniref50",
            # default="/mnt/sda/models/Rostlab prot_bert_bfd"
        )
        args = self.parser.parse_args()
        return args

    def generator_rnn(self):
        """ task args """
        self.parser.add_argument("--task_name", type=str, default='', help="wound_heal, anti_inflammation, toxic")
        self.parser.add_argument("--enable_train", type=int, default=1, help="0 is false, 1 is true")
        self.parser.add_argument("--embedding_type", type=str, default='', help="vocab, esm, t5")
        self.parser.add_argument("--only_positive", type=int, default=1,
            help="Generator train/test only need positive samples")
        self.parser.add_argument("--read_dataset_cache", type=int, default=1, help="0 is false, overwite; 1 true")
        self.parser.add_argument("--basic_model", type=str, default='lstm', help="")

        self.parser.add_argument("--n_epoch", type=int, default=300, help="")
        self.parser.add_argument("--learning_rate", type=float, default=0.18, help="")
        self.parser.add_argument("--dropout", type=float, default=0.5, help="")
        self.parser.add_argument("--n_embedding", type=int, default=100, help="")
        self.parser.add_argument("--n_hidden", type=int, default=400, help="")
        self.parser.add_argument("--n_layers", type=int, default=2, help="")
        self.parser.add_argument("--momentum", type=float, default=0.9, help="")
        self.parser.add_argument("--batch_size", type=int, default=50, help="")

        self.parser.add_argument("--use_fixed_hyparameter", type=int, default=1, help="0 is false, 1 is true")
        self.parser.add_argument("--overwrite_hyperparameter_dict", type=int, default=0, help="0 is false, 1 is true")
        self.parser.add_argument("--generated_num", type=int, default=180_000, help="")
        self.parser.add_argument("--run_test", type=int, default=1, help="")
        args = self.parser.parse_args()
        return args

    def classifier_transformer_rnn(self):
        """ task args """
        self.parser.add_argument(
            "--transformer_model_path", type=str,
            default="/mnt/sda/models/Rostlab_prot_t5_xl_uniref50",
            # default="/mnt/sda/models/Rostlab prot_bert_bfd"
        )
        self.parser.add_argument("--transformer_mode", type=str, default='bert', help="bert, t5")
        self.parser.add_argument("--use_rnn", type=int, default=0, help="0 is false, 1 is true")
        self.parser.add_argument("--bert_lr", type=float, default=1e-5)
        self.parser.add_argument("--rnn_lr", type=float, default=1e-2)
        self.parser.add_argument("--output_lr", type=float, default=1e-5)
        self.parser.add_argument("--n_epoch", type=int, default=100)
        self.parser.add_argument("--test_steps_num", type=int, default=103)
        self.parser.add_argument("--adam_epsilon", type=float, default=1e-8)
        self.parser.add_argument("--weight_decay", default=0.01, type=float, help="Weight decay if we apply some.")
        self.parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
        # self.parser.add_argument("--use_fixed_hyparameter", type=int, default=1, help="0 is false, 1 is true")
        args = self.parser.parse_args()
        return args

    def predictor(self):
        """  """
        self.parser.add_argument("--read_cached_input_data", type=int, default=0, help="")
        args = self.parser.parse_args()
        return args


def save_args(args, output_dir='.', with_time_at_filename=False):
    os.makedirs(output_dir, exist_ok=True)

    t0 = datetime.now().strftime(DATE_TIME)
    if with_time_at_filename:
        out_file = os.path.join(output_dir, f"args-{t0}.txt")
    else:
        out_file = os.path.join(output_dir, f"args.txt")
    with open(out_file, "w", encoding='utf-8') as f:
        f.write(f'{t0}\n')
        for arg, value in vars(args).items():
            f.write(f"{arg}: {value}\n")


def log_args(args, logger):
    for arg, value in vars(args).items():
        logger.info(f"{arg}: {value}")
