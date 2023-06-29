from .model import SpotNet, Config, SpotNetData 

from csbdeep.models import register_model, register_aliases, clear_models_and_aliases
# register pre-trained models and aliases (TODO: replace with updatable solution)
clear_models_and_aliases(SpotNet)
register_model(SpotNet,   'hybiss', 'https://www.dropbox.com/s/dcrn0f4yggytczs/hybiss_model_v2.zip?dl=1', '87b3e6cc0505f62926a925e1054866ac')
del register_model, register_aliases, clear_models_and_aliases