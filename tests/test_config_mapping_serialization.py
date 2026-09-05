from collections import UserDict
import json
from adapters.longlive_sparse.config import SparseHistoryConfig


def test_nested_parameter_mapping_is_normalized_before_persistence():
    wrapped=UserDict({'method':'transfer_vaware_hybrid_history',
                     'method_params':UserDict({'base_fraction':.7,'local_fraction':.15})})
    config=SparseHistoryConfig.from_mapping(wrapped)
    assert type(config.method_params) is dict
    assert json.loads(json.dumps(config.as_dict()))['method_params']=={'base_fraction':.7,'local_fraction':.15}
