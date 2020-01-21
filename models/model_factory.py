from models.models import *


def model_select(activate_model):
    if activate_model == 'DHGNN_v1':
        return DHGNN_v1
    elif activate_model == 'DHGNN_v2':
        return DHGNN_v2
    else:
        raise ValueError