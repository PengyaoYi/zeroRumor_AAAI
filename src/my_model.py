import torch
import torch.nn as nn
import transformers
from transformers.modeling_bert import BertPreTrainedModel, BertForSequenceClassification, BertModel, BertOnlyMLMHead
from transformers.modeling_roberta import RobertaForSequenceClassification, RobertaModel, RobertaLMHead, \
    RobertaClassificationHead
from src.xlm_roberta import XLMRobertaModel_trans  # , RobertaLMHead, RobertaClassificationHead
from transformers.modeling_outputs import SequenceClassifierOutput
import copy
import torch.nn.functional as F
from transformers import AutoTokenizer
from torch.autograd import Variable
import logging

logger = logging.getLogger(__name__)


class RelativePosition(nn.Module):

    def __init__(self, num_units, max_relative_position):
        super().__init__()
        self.num_units = num_units
        self.max_relative_position = max_relative_position

        self.embeddings_table = nn.Parameter(torch.Tensor(max_relative_position + 1, num_units))
        nn.init.xavier_uniform_(self.embeddings_table)


    def forward(self, relation_matrix):

        batch_size = relation_matrix.size(0)
        seq_len = relation_matrix.size(1)

        mask = relation_matrix.ge(1).unsqueeze(3).expand(batch_size, seq_len, seq_len, 768)

        embeddings = self.embeddings_table[relation_matrix]

        final_embeddings = (embeddings * mask)
        return final_embeddings


class LearnedPositionEncoder(nn.Module):
    """

    This set of codes would encode the structural information

    """

    def __init__(self, config, n_heads):
        super(LearnedPositionEncoder, self).__init__()

        self.config = config
        self.n_heads = n_heads
        self.d_emb_dim = config.hidden_size // self.n_heads
        self.n_pos = 5 + 1  # +1 for padding

        # <------------- Defining the position embedding ------------->
        self.structure_emb = nn.Embedding(self.n_pos, self.d_emb_dim)
        self.structure_emb.requries_grad = True

    def forward(self, src_seq):
        # <------------- Get the shape ------------->
        batch_size, num_posts, num_posts = src_seq.shape

        # <------------- Duplicate the src_seq based on the number of heads first ------------->
        src_seq = src_seq.repeat(self.n_heads, 1, 1)
        encoded_structure_features = self.structure_emb(src_seq)


        # <------------- Break into individual heads ------------->
        encoded_structure_features = encoded_structure_features.view(batch_size, self.n_heads, num_posts, num_posts, -1)

        return encoded_structure_features


class RobertaThreePart(BertPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.grad_loss = config.grad_loss
        self.cl_nn_size = config.cl_nn_size
        self.independen = config.independen
        self.alp = config.alp
        self.noise_norm = config.noise_norm
        self.activa = config.tem_activa
        self.position_mode = config.position

        self.template_tower = XLMRobertaModel_trans(config, add_pooling_layer=False)
        self.context_tower = XLMRobertaModel_trans(config, add_pooling_layer=False)
        self.fusion_tower = XLMRobertaModel_trans(config, fusion=True)
        self.pool_sent_limit = config.pooling_sent_limit
        self.fusion_embedding = nn.Embedding(self.pool_sent_limit, config.hidden_size) 


        if self.position_mode:#1128
            self.relation_encoder = RelativePosition(768, 5) #1128


        self.cl_nn = torch.nn.Linear(config.hidden_size, self.cl_nn_size, bias=False)

        self.init_weights()
        # These attributes should be assigned once the model is initialized
        self.model_args = None
        self.data_args = None
        self.label_word_list = None

        # For auto label search.
        self.return_full_softmax = None
        # for debug
        # self.tokenizer = AutoTokenizer.from_pretrained("./xlm-model")

        w = torch.empty((2, self.cl_nn_size))
        nn.init.xavier_uniform_(w)
        self.proto = nn.Parameter(w, requires_grad=True)
        self.num_classes = 2
        self.layer_norm = nn.LayerNorm(768)

    def forward(
            self,
            input_ids=None,
            template_input_ids=None,
            attention_mask=None,
            template_attention_mask=None,
            mask_pos=None,
            labels=None,
            comment_pos=None,
            relation_sent=None,
            abs_relation=None,

    ):

        batch_size = input_ids.size(0)
        if mask_pos is not None:
            mask_pos = mask_pos.squeeze()
        key_structure = None
        # Encode everything

        if self.position_mode:
           key_structure = self.relation_encoder(relation_sent)

        context_outputs = self.context_tower(
            input_ids,
            attention_mask=attention_mask,
        )
        template_outputs = self.template_tower(
            template_input_ids,
            attention_mask=template_attention_mask,
        )
        context_hidden_states = context_outputs[0].detach()
        template_hidden_states = template_outputs[0].detach()
        if not self.model_args.reverse:
            fusion_input = torch.cat((context_hidden_states, template_hidden_states), 1)
            fusion_attention_mask = torch.cat((attention_mask, template_attention_mask), 1)
        else:
            fusion_input = torch.cat((template_hidden_states, attention_mask), 1)
            fusion_attention_mask = torch.cat((template_attention_mask, attention_mask), 1)

        fusion_abs_pos = self.fusion_embedding(abs_relation)
        concat_input = fusion_input + fusion_abs_pos

        if self.position_mode:
            segment = input_ids.eq(2)
            if torch.cuda.is_available():
                index_tensor = torch.arange(0, len(input_ids[0])).expand(input_ids.shape).cuda()
            else:
                index_tensor = torch.arange(0, len(input_ids[0])).expand(input_ids.shape)
            sentence_pos = segment * index_tensor
            sentence_index = torch.nonzero(sentence_pos)
            batch_index = [[] for _ in range(batch_size)]
            for item in sentence_index:
                batch_index[item[0]].append(item[1])
            pooled_sent = None
            for i, batch_ in enumerate(batch_index):
                if batch_ == []:
                    if pooled_sent is None:
                        pooled_sent = torch.nn.functional.max_pool1d(
                            context_hidden_states[i].unsqueeze(0).transpose(1, 2),
                            context_hidden_states[i].size(0)).transpose(1, 2)
                    else:
                        b = torch.nn.functional.max_pool1d(context_hidden_states[i].unsqueeze(0).transpose(1, 2),
                                                           context_hidden_states[i].size(0)).transpose(1, 2)
                        pooled_sent = torch.cat((pooled_sent, b), 1)
                for j in range(0, len(batch_)):
                    if j == 0:
                        a = context_hidden_states[i][0:batch_[j]]

                        b = torch.nn.functional.max_pool1d(
                            a.unsqueeze(0).transpose(1, 2), a.size(0)).transpose(1, 2)
                        pooled_sent = b
                    else:
                        a = context_hidden_states[i][batch_[j - 1]:batch_[j]]
                        b = torch.nn.functional.max_pool1d(
                            a.unsqueeze(0).transpose(1, 2), a.size(0)).transpose(1, 2)
                        pooled_sent = torch.cat((pooled_sent, b), 1)

                while pooled_sent.size(1) < self.pool_sent_limit:
                    if torch.cuda.is_available():
                        pooled_sent = torch.cat((pooled_sent, torch.zeros(1, 1, 768).cuda()), 1)
                    else:
                        pooled_sent = torch.cat((pooled_sent, torch.zeros(1, 1, 768)), 1)
                if pooled_sent.size(1) > self.pool_sent_limit:
                    pooled_sent = pooled_sent[:, :self.pool_sent_limit, :]
                if i == 0:
                    batch_pool_sent = pooled_sent
                else:
                    batch_pool_sent = torch.cat((batch_pool_sent, pooled_sent), 0)
            batch_pool_sent_tensor = batch_pool_sent
        else:
            batch_pool_sent_tensor = None
            batch_index = None

        outputs = self.fusion_tower(
            concat_input,
            attention_mask=fusion_attention_mask,
            key_structure=key_structure,
            sentence_tensor=batch_pool_sent_tensor,
            batch_index=batch_index,
        )

        # Get <mask> token representation
        sequence_output, pooled_output = outputs[:2]
        sequence_mask_output = sequence_output[torch.arange(sequence_output.size(0)), mask_pos]

        # prediction_mask_scores = sequence_mask_output


        loss = None
        if labels is not None:
            if self.training:  # and self.cl_mode:
                embeds = [[] for _ in range(2)]

                for j in range(len(sequence_mask_output)):
                    label = labels[j]
                    embeds[label].append(sequence_mask_output[j])
                embeds = [torch.stack(e) for e in embeds]
                embeds = torch.stack(embeds)
                x = self.cl_nn(embeds)
                loss_cl, logits_cl, loss_cl_proto = self.cl_loss_format(x, self.cl_nn(sequence_mask_output))

                # bagging

                loss = loss_cl
                logits = logits_cl

            else:  # self.cl_mode:
                logits_cl = self.cal_logits(self.cl_nn(sequence_mask_output))
                logits = logits_cl

                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

        if self.training:

            x_ = concat_input
            if torch.cuda.is_available():
                index_tensor = torch.arange(0, len(input_ids[0])).expand(input_ids.shape).cuda()
            else:
                index_tensor = torch.arange(0, len(input_ids[0])).expand(input_ids.shape)
            tem_mask = torch.zeros_like(template_attention_mask)
            index_tensor = torch.cat((index_tensor, tem_mask), dim=1)
            mask_fill_tensor = comment_pos.unsqueeze(1).expand_as(concat_input[:, :, 0])
            mask_matric = index_tensor > mask_fill_tensor
            x_.retain_grad()  # we need to get gradient w.r.t low-resource embeddings
            loss_cl_proto.backward(retain_graph=True)
            unnormalized_noise = x_.grad.detach_()
            for p in self.parameters():
                if p.grad is not None:
                    p.grad.detach_()
                    p.grad.zero_()
            norm = unnormalized_noise.norm(p=2, dim=-1)
            normalized_noise = unnormalized_noise / (norm.unsqueeze(dim=-1) + 1e-10)  # add 1e-10 to avoid Nan

            noise_norm = self.noise_norm

            #target_noise = noise_norm * normalized_noise
            target_noise = normalized_noise
            noise_x_ = self.layer_norm(concat_input) + torch.mul(target_noise, mask_matric.unsqueeze(-1)) #+ fusion_abs_pos
            if self.position_mode:
                segment = input_ids.eq(2)
                if torch.cuda.is_available():
                    index_tensor = torch.arange(0, len(input_ids[0])).expand(input_ids.shape).cuda()
                else:
                    index_tensor = torch.arange(0, len(input_ids[0])).expand(input_ids.shape)
                sentence_pos = segment * index_tensor
                sentence_index = torch.nonzero(sentence_pos)
                batch_index = [[] for _ in range(batch_size)]
                for item in sentence_index:
                    batch_index[item[0]].append(item[1])
                pooled_sent = None
                for i, batch_ in enumerate(batch_index):
                    if batch_ == []:
                        if pooled_sent is None:
                            pooled_sent = torch.nn.functional.max_pool1d(noise_x_[i].unsqueeze(0).transpose(1, 2),
                                                                         noise_x_[i].size(0)).transpose(1, 2)
                        else:
                            b = torch.nn.functional.max_pool1d(noise_x_[i].unsqueeze(0).transpose(1, 2),
                                                               noise_x_[i].size(0)).transpose(1, 2)
                            pooled_sent = torch.cat((pooled_sent, b), 1)
                    for j in range(0, len(batch_)):
                        if j == 0:
                            a = noise_x_[i][0:batch_[j]]
                            b = torch.nn.functional.max_pool1d(
                                a.unsqueeze(0).transpose(1, 2), a.size(0)).transpose(1, 2)
                            pooled_sent = b
                        else:
                            a = noise_x_[i][batch_[j - 1]:batch_[j]]
                            b = torch.nn.functional.max_pool1d(
                                a.unsqueeze(0).transpose(1, 2), a.size(0)).transpose(1, 2)
                            pooled_sent = torch.cat((pooled_sent, b), 1)

                    while pooled_sent.size(1) < 128:
                        if torch.cuda.is_available():
                            pooled_sent = torch.cat((pooled_sent, torch.zeros(1, 1, 768).cuda()), 1)
                        else:
                            pooled_sent = torch.cat((pooled_sent, torch.zeros(1, 1, 768)), 1)
                        # pooled_sent = torch.cat((pooled_sent, torch.zeros(1, 1, 768)), 1)
                    if pooled_sent.size(1) > 128:
                        pooled_sent = pooled_sent[:, :128, :]
                    if i == 0:
                        batch_pool_sent = pooled_sent
                    else:
                        batch_pool_sent = torch.cat((batch_pool_sent, pooled_sent), 0)

                batch_pool_sent_tensor = batch_pool_sent
            else:
                batch_pool_sent_tensor = None
                batch_index = None

            noise_sequence_output = self.fusion_tower(noise_x_,
                                                      attention_mask=fusion_attention_mask,
                                                      key_structure=key_structure,
                                                      sentence_tensor=batch_pool_sent_tensor,
                                                      batch_index=batch_index,
                                                      )[0]
            noise_sequence_mask_output = noise_sequence_output[torch.arange(noise_sequence_output.size(0)), mask_pos]
            noise_embeds = [[] for _ in range(2)]

            for j in range(len(noise_sequence_mask_output)):
                label = labels[j]
                noise_embeds[label].append(noise_sequence_mask_output[j])
            noise_embeds = [torch.stack(e) for e in noise_embeds]
            noise_embeds = torch.stack(noise_embeds)
            noise_x = self.cl_nn(noise_embeds)
            noise_loss, _nologits = self.noise_cl_loss(noise_x, self.cl_nn(noise_sequence_mask_output), x)

            total_loss = self.alp * loss_cl + (1 - self.alp) * noise_loss

        else:
            total_loss = loss

        output = (logits, sequence_mask_output)
        if self.num_labels == 1:
            # Regression output
            output = (torch.exp(logits[..., 1].unsqueeze(-1)) * (self.ub - self.lb) + self.lb,)
        return ((total_loss,) + output) if total_loss is not None else output

    @staticmethod
    def sim(x, y):
        norm_x = F.normalize(x, dim=-1)
        norm_y = F.normalize(y, dim=-1)
        return torch.matmul(norm_x, norm_y.transpose(1, 0))

    def cal_logits(self, v_pro):
        sim_mat_agg = torch.exp(self.sim(v_pro, self.proto))
        logits = F.softmax(sim_mat_agg, dim=1)
        return logits

    def cl_loss_format(self, v_ins, v_pro):
        # instance-prototype loss
        logits = []
        sim_mat = torch.exp(self.sim(v_ins, self.proto))
        sim_mat_agg = torch.exp(self.sim(v_pro, self.proto))
        num = sim_mat.shape[1]
        loss = 0.
        for i in range(num):
            pos_score = torch.diag(sim_mat[:, i, :])
            neg_score = (sim_mat[:, i, :].sum(1) - pos_score)
            logits_ = pos_score / (pos_score + neg_score)
            loss += - torch.log(logits_).sum()

        loss = loss / (num * self.num_classes * self.num_classes)
        logits = F.softmax(sim_mat_agg, dim=1)
        # instance-instance loss

        loss_ins = 0.
        for i in range(v_ins.shape[0]):
            sim_instance = torch.exp(self.sim(v_ins, v_ins[i]))
            pos_ins = sim_instance[i]
            neg_ins = (sim_instance.sum(0) - pos_ins).sum(0)
            loss_ins += - torch.log(pos_ins / (pos_ins + neg_ins)).sum()
        loss_ins = loss_ins / (num * self.num_classes * num * self.num_classes)
        loss_ = loss + loss_ins

        return loss_, logits, loss

    def noise_cl_loss(self, v_ins, v_pro, v_ins_true):
        sim_mat = torch.exp(self.sim(v_ins, self.proto))
        sim_mat_agg = torch.exp(self.sim(v_pro, self.proto))
        num = sim_mat.shape[1]
        loss = 0.
        for i in range(num):
            pos_score = torch.diag(sim_mat[:, i, :])
            neg_score = (sim_mat[:, i, :].sum(1) - pos_score)
            logits_ = pos_score / (pos_score + neg_score)
            loss += - torch.log(logits_).sum()

        loss = loss / (num * self.num_classes * self.num_classes)
        logits = F.softmax(sim_mat_agg, dim=1)
        # instance-instance loss

        # loss_ins = 0.
        # for i in range(v_ins.shape[0]):
        #     sim_instance = torch.exp(self.sim(v_ins, v_ins_true[i]))
        #     pos_ins = sim_instance[i]
        #     neg_ins = (sim_instance.sum(0) - pos_ins).sum(0)
        #     loss_ins += - torch.log(pos_ins / (pos_ins + neg_ins)).sum()
        # loss_ins = loss_ins / (num * self.num_classes * num * self.num_classes)
        # loss = loss + loss_ins

        return loss, logits