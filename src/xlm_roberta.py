from transformers.configuration_xlm_roberta import XLMRobertaConfig
from transformers.file_utils import add_start_docstrings
from transformers.modeling_roberta import (
    RobertaForCausalLM,
    RobertaForMaskedLM,
    RobertaForMultipleChoice,
    RobertaForQuestionAnswering,
    RobertaForSequenceClassification,
    RobertaForTokenClassification,
    RobertaModel,
)
from src.roberta_model import RobertaModel_trans

XLM_ROBERTA_START_DOCSTRING = r"""

    This model inherits from :class:`~transformers.PreTrainedModel`. Check the superclass documentation for the generic
    methods the library implements for all its model (such as downloading or saving, resizing the input embeddings,
    pruning heads etc.)

    This model is also a PyTorch `torch.nn.Module <https://pytorch.org/docs/stable/nn.html#torch.nn.Module>`__ subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general
    usage and behavior.

    Parameters:
        config (:class:`~transformers.XLMRobertaConfig`): Model configuration class with all the parameters of the
            model. Initializing with a config file does not load the weights associated with the model, only the configuration.
            Check out the :meth:`~transformers.PreTrainedModel.from_pretrained` method to load the model weights.
"""


@add_start_docstrings(
    "The bare XLM-RoBERTa Model transformer outputting raw hidden-states without any specific head on top.",
    XLM_ROBERTA_START_DOCSTRING,
)
class XLMRobertaModel_trans(RobertaModel_trans):
    """
    This class overrides :class:`~transformers.RobertaModel`. Please check the
    superclass for the appropriate documentation alongside usage examples.
    """

    config_class = XLMRobertaConfig