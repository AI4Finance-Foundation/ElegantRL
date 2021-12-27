from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from pysc2.maps import lib
from smac.env.starcraft2.maps import smac_maps

map_param_registry = {
    "1o_10b_vs_1r": {
        "n_agents": 11,
        "n_enemies": 1,
        "limit": 50,
        "a_race": "Z",
        "b_race": "Z",
        "unit_type_bits": 2,
        "map_type": "overload_bane"
    },
    "1o_2r_vs_4r": {
        "n_agents": 3,
        "n_enemies": 4,
        "limit": 50,
        "a_race": "Z",
        "b_race": "Z",
        "unit_type_bits": 2,
        "map_type": "overload_roach"
    },
        "bane_vs_hM": {
        "n_agents": 3,
        "n_enemies": 2,
        "limit": 30,
        "a_race": "Z",
        "b_race": "T",
        "unit_type_bits": 2,
        "map_type": "bZ_hM"
    }
}


smac_maps.map_param_registry.update(map_param_registry)

def get_map_params(map_name):
    map_param_registry = smac_maps.get_smac_map_registry()
    return map_param_registry[map_name]


for name in map_param_registry.keys():
    globals()[name] = type(name, (smac_maps.SMACMap,), dict(filename=name))
