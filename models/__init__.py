"""Init file for models."""
from .db_bhcn.db_bhcn import DB_BHCN
from .db_bhcn_awx.db_bhcn_awx import DB_BHCN_AWX
from .db_achmcnn.db_achmcnn import DB_AC_HMCNN
from .db_ahmcnf.db_ahmcnf import DB_AHMCN_F
from .db_linear.db_linear import DB_Linear, get_metrics as get_linear_metrics
from .tfidf_hsgd.tfidf_hsgd import Tfidf_HSGD
from .tfidf_lsgd.tfidf_lsgd import Tfidf_LSGD

PYTORCH_MODEL_LIST = [
    'db_bhcn',
    'db_bhcn_awx',
    'db_ahmcnf',
    'db_achmcnn',
    'db_linear',
]
SKLEARN_MODEL_LIST = [
    'tfidf_hsgd',
    'tfidf_lsgd'
]
MODEL_LIST = PYTORCH_MODEL_LIST + SKLEARN_MODEL_LIST


__all__ = [
    'PYTORCH_MODEL_LIST',
    'SKLEARN_MODEL_LIST',
    'MODEL_LIST',
    'DB_BHCN',
    'DB_BHCN_AWX',
    'DB_AC_HMCNN',
    'DB_AHMCN_F',
    'DB_Linear',
    'get_linear_metrics',
    'Tfidf_HSGD',
    'Tfidf_LSGD'
]
