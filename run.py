import scanpy as sc
import warnings
warnings.filterwarnings ("ignore")
from pathlib import Path
import TOSICA
from utils.log_util import logger
from utils.arg_util import ArgparseUtil, log_args
from utils.file_util import FileUtil
from app.postprocess.calc_performance_on_query import calc_accuracy
from app.data_preprocess.orig_train_test_data_reader import read_train_test_data


def read_train_config(data_type):
    """  """
    model_weight_paths = []
    logger.info('data_type %s', data_type)
    for file in Path('config').iterdir():
        if file.stem.startswith(data_type):
            configs = FileUtil.read_json(file)
            best_epoch = configs['best_epoch']
            model_dir_name = file.stem
            model_weight_path = f'model_files/{model_dir_name}/model-{best_epoch}.pth'
            model_weight_paths.append(model_weight_path)
    return model_weight_paths


args = ArgparseUtil().train_classifier()
data_type = args.data_type
project_path = args.project
ref_adata, query_adata = read_train_test_data(data_type)
project_dir = Path(f'projects/{project_path}')
project_dir.mkdir(exist_ok=1, parents=1)
log_args(args, logger)

if args.enable_train:
    TOSICA.train(
        ref_adata, gmt_path=args.gmt_path, data_type=data_type, project_path=project_dir,
        label_name=args.label_name,
        epochs=args.n_epoch,
        lr=args.learning_rate,
        embed_dim=args.embed_dim,
        depth=args.depth,
        num_heads=args.num_heads,
        batch_size=args.batch_size,
        data_seed=args.data_seed,
        val_data_ratio=args.val_data_ratio,
        seed=args.seed)
else:
    cached_prediction_file = project_dir / 'predicted_result.h5ad'
    if args.read_cached_prediction and cached_prediction_file.is_file():
        new_adata = sc.read(cached_prediction_file)
        logger.info(f'Loads predicted_result from {cached_prediction_file}')
        calc_accuracy(project_dir, query_adata, new_adata)
    else:
        # Support to compare the multi train config files
        model_weight_paths = read_train_config(data_type)
        if not model_weight_paths:
            logger.exception(f'There is no valid saved train config file for data type {data_type}')
        for model_weight_path in model_weight_paths:
            logger.info('trained model_weight_path %s', model_weight_path)
            new_adata = TOSICA.pre(query_adata, model_weight_path=model_weight_path, project_path=project_dir)
            new_adata.write(cached_prediction_file)
            calc_accuracy(project_dir, query_adata, new_adata)