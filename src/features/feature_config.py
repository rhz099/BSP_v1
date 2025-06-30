# src/features/feature_config.py

FEATURE_CONFIGS = {
    "base": {
        "include_base": True,
        "include_structural": False,
        "include_typology": False,
        "include_temporal": False,
    },
    "base+structural": {
        "include_base": True,
        "include_structural": True,  
        "include_typology": False,
        "include_temporal": False,
    },
    "base+typology": {
        "include_base": True,
        "include_structural": False,
        "include_typology": True,     
        "include_temporal": False,
    },
    "base+structural+typology": {
        "include_base": True,
        "include_structural": True,
        "include_typology": True,
        "include_temporal": False,
    },
    "base+basic_temporal": {
        "include_base": True,
        "include_structural": False,
        "include_typology": False,
        "include_temporal": True,     
    },
    "base+basic_temporal+typology": {
        "include_base": True,
        "include_structural": False,
        "include_typology": True,
        "include_temporal": True,
    },
    "all": {
        "include_base": True,
        "include_structural": True,
        "include_typology": True,
        "include_temporal": True,
    }
}
