import logging, os


_nameToLevel = {
#    'CRITICAL': logging.CRITICAL,
#    'FATAL': logging.FATAL,  # FATAL = CRITICAL
#    'ERROR': logging.ERROR,
#    'WARN': logging.WARNING,
   'INFO': logging.INFO,
   'DEBUG': logging.DEBUG,
}
fmt = '%(asctime)s %(filename)s %(lineno)d: %(message)s'
datefmt = '%y-%m-%d %H:%M'
# datefmt = '%y-%m-%d %H:%M:%S'


def get_logger(name=None, log_file=None, log_level=logging.DEBUG, log_level_name=''):
    """ default log level DEBUG """
    logger = logging.getLogger(name)
    logging.basicConfig(format=fmt, datefmt=datefmt)
    if log_file is not None:
        log_file_folder = os.path.split(log_file)[0]
        if log_file_folder:
            os.makedirs(log_file_folder, exist_ok=True)
        fh = logging.FileHandler(log_file, 'w', encoding='utf-8')
        fh.setFormatter(logging.Formatter(fmt, datefmt))
        logger.addHandler(fh)
    if log_level_name in _nameToLevel:
        log_level = _nameToLevel[log_level_name]
    logger.setLevel(log_level)
    return logger


def log_df_basic_info(df, log_func=None, comments=''):
    if log_func is None:
        log_func = logger
    if comments:
        log_func.info(f'comments {comments}')
    log_func.info(f'df.shape {df.shape}')
    log_func.info(f'df.columns {df.columns.to_list()}')
    log_func.info(f'df.head()\n{df.head()}')
    log_func.info(f'df.tail()\n{df.tail()}')


logger = get_logger()
