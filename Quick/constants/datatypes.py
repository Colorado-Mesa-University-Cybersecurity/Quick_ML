

BASE_TYPE_SCHEMA = [
    'name',
    'X_train',
    'y_train',
    'classes',
    'target_label'
]

MODEL_TYPE_SCHEMA = [
    *BASE_TYPE_SCHEMA,
    'model',
    'model_type',
    'X_test',
    'y_test',
    'to',
    'dls'   
]

COMPONENT_TYPE_SCHEMA = [
    *BASE_TYPE_SCHEMA,
    'Xy',
    'Components',
    'n_components'
]

TOPOLOGICAL_TYPE_SCHEMA = [
    *BASE_TYPE_SCHEMA,
    'clouds', 
    'clouds_y', 
    'clouds_y_decoded', 
    'persistence', 
    'features', 
    'Xy'
    'fig', 
]
