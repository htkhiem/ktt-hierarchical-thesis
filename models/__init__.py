"""Init file for models."""
from .db_bhcn.db_bhcn import DB_BHCN
from .db_bhcn_awx.db_bhcn_awx import DB_BHCN_AWX
from .db_achmcnn.db_achmcnn import DB_AC_HMCNN
from .db_ahmcnf.db_ahmcnf import DB_AHMCN_F
from .db_linear.db_linear import DB_Linear
from .tfidf_hsgd.tfidf_hsgd import Tfidf_HSGD
from .tfidf_lsgd.tfidf_lsgd import Tfidf_LSGD

PYTORCH_MODELS = {
    'db_bhcn': DB_BHCN,
    'db_bhcn_awx': DB_BHCN_AWX,
    'db_ahmcnf': DB_AHMCN_F,
    'db_achmcnn': DB_AC_HMCNN,
    'db_linear': DB_Linear,
}
SKLEARN_MODELS = {
    'tfidf_hsgd': Tfidf_HSGD,
    'tfidf_lsgd': Tfidf_LSGD
}

FRAMEWORKS = [PYTORCH_MODELS, SKLEARN_MODELS]

MODELS = {k: v for d in FRAMEWORKS for k, v in d.items()}

__all__ = [
    'PYTORCH_MODELS',
    'SKLEARN_MODELS',
    'FRAMEWORKS',
    'MODELS',
    'DB_BHCN',
    'DB_BHCN_AWX',
    'DB_AC_HMCNN',
    'DB_AHMCN_F',
    'DB_Linear',
    'Tfidf_HSGD',
    'Tfidf_LSGD'
]
