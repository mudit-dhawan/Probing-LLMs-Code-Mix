import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from transformers.modeling_outputs import MaskedLMOutput 

class XLMPredLayer(nn.Module):
    """
    Prediction layer (cross_entropy or adaptive_softmax).
    """

    def __init__(self, config):
        super().__init__()
        self.asm = config['asm']
        self.n_classes_lid = config['n_classes_lid']
        self.pad_index = config['pad_index']
        dim = config['emb_dim']

        if config.asm is False:
            self.proj = nn.Linear(dim, self.n_classes_lid, bias=True)
        else:
            self.proj = nn.AdaptiveLogSoftmaxWithLoss(
                in_features=dim,
                n_classes=self.n_classes_lid,
                cutoffs=config['asm_cutoffs'],
                div_value=config['asm_div_value'],
                head_bias=True,  # default is False
            )

    def forward(self, x, y=None):
        """Compute the loss, and optionally the scores."""
        outputs = ()
        if self.asm is False:
            scores = self.proj(x)
            outputs = (scores,) + outputs
            if y is not None:
                loss = F.cross_entropy(scores.view(-1, self.n_classes_lid), y.view(-1), reduction="elementwise_mean")
                outputs = (loss,) + outputs
        else:
            scores = self.proj.log_prob(x)
            outputs = (scores,) + outputs
            if y is not None:
                _, loss = self.proj(x, y)
                outputs = (loss,) + outputs

        return outputs



class XLM_model_lid(nn.Module):

    def __init__(self, xlm_config, pred_layer_config):
        super().__init__()

        self.xlm_mlm = XLMWithLMHeadModel(xlm_config)
        self.pred_layer = XLMPredLayer(pred_layer_config)


    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        langs=None,
        token_type_ids=None,
        position_ids=None,
        lengths=None,
        cache=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):

    xlm_outputs = self.xlm_mlm(
            input_ids,
            attention_mask=attention_mask,
            langs=langs,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            lengths=lengths,
            cache=cache,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

    last_hiddent_state = xlm_outputs.hidden_states[-1] ## (batch_size, sequence_length, hidden_size)

    ## If we use langs as the second input then the loss is calculated for all the tokens 
    outputs = self.pred_layer(last_hiddent_state, langs)  # (loss, logits) or (logits,) depending on if labels are provided.

    return MaskedLMOutput(
            total_loss=(xlm_outputs+outputs[0]) if labels is not None else None,
            logits=(xlm_outputs[0]+outputs[0]) if labels is None else (xlm_outputs[1], outputs[1]),
            hidden_states=xlm_outputs.hidden_states,
            attentions=xlm_outputs.attentions,
        )