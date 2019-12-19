from typing import Dict

import allennlp.modules.token_embedders.bert_token_embedder as bert_embedder
import torch
from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import Seq2VecEncoder
from allennlp.modules.feedforward import FeedForward
from allennlp.nn.util import get_text_field_mask
from allennlp.training.metrics import BooleanAccuracy
from allennlp.training.metrics import CategoricalAccuracy
from overrides import overrides


@Model.register("bert-split")
class BertSplit(Model):

    def __init__(self, vocab: Vocabulary,
                 first_stage_count) -> None:
        super(BertSplit, self).__init__(vocab)

        self.vocab = vocab

        self.model = bert_embedder.PretrainedBertModel().load('bert-base-uncased', cache_model=False)
        self.bert_init_embedder = self.model.embeddings
        self.bert_layers = None
        for layers in self.model.encoder.children():
            self.bert_layers = layers
        self.bert_pooler = self.model.pooler
        self.first_stage_count = first_stage_count

    @overrides
    def forward(self,
                premises: Dict[str, torch.LongTensor],
                attention: torch.Tensor) -> torch.Tensor:

        # Calculate Embeddings
        initial_embeddings = self.bert_init_embedder(premises['tokens'])

        # Calculate mask
        mask = get_text_field_mask({'tokens': premises['tokens']}) \
            .unsqueeze(1) \
            .unsqueeze(2) \
            .to(dtype=next(self.parameters()).dtype)
        mask = (1.0 - mask) * -10000.0

        # First Stage
        FIRST_STAGE_NUM_LAYERS = self.first_stage_count
        prev_output, curr_output = initial_embeddings, None
        for i in range(FIRST_STAGE_NUM_LAYERS):
            curr_output = self.bert_layers[i].forward(prev_output, mask)
            prev_output = curr_output
            del curr_output

        # Attention
        attention = attention.repeat(1, initial_embeddings.shape[1] * initial_embeddings.shape[2]).reshape(
            initial_embeddings.shape)

        # Apply Attention
        prev_output = prev_output * attention

        # Join
        num_tokens = prev_output.shape[0] * prev_output.shape[1]
        emb_size = prev_output.shape[2]
        prev_output = prev_output.reshape(num_tokens, emb_size).unsqueeze(0)
        mask = mask.reshape(-1, num_tokens)

        # Second Stage
        for i in range(FIRST_STAGE_NUM_LAYERS, len(self.bert_layers)):
            curr_output = self.bert_layers[i].forward(prev_output, mask)
            prev_output = curr_output
            del curr_output

        pooled = self.bert_pooler(prev_output)

        return pooled


@Model.register("bert-multee")
class Multee(Model):

    def __init__(self, vocab: Vocabulary,
                 attention_module: Model,
                 classifier: FeedForward) -> None:
        super(Multee, self).__init__(vocab)

        self.attention_module = attention_module
        self.branch_1 = BertSplit(vocab, 4)
        self.branch_2 = BertSplit(vocab, 6)
        self.classifier = classifier
        self.loss = torch.nn.MSELoss()
        self.accuracy = BooleanAccuracy()
        self.statistics = {
            "100%": 0,
            "75%": 0,
            "50%": 0,
            "25%": 0,
            "other": 0,
            "none": 0
        }

    @overrides
    def forward(self,
                premises: Dict[str, torch.LongTensor],
                hypotheses: Dict[str, torch.LongTensor],
                # paragraph: Dict[str, torch.LongTensor],
                question: Dict[str, torch.LongTensor],
                answer_correctness_mask: torch.IntTensor = None,
                relevance_presence_mask: torch.IntTensor = None) -> Dict[str, torch.Tensor]:

        num_hypothesis = hypotheses["tokens"].shape[1]
        attentions, attention_loss = [], None
        for i in range(num_hypothesis):
            attention_module_output = self.attention_module(premises,
                                                            {"tokens": hypotheses["tokens"][:, i, :]},
                                                            relevance_presence_mask)
            attentions.append(attention_module_output.get('logits'))
            loss_ = attention_module_output.get('loss', None)
            if loss_ is not None:
                if attention_loss is None:
                    attention_loss = loss_
                else:
                    attention_loss += loss_

        concatenated_pooled = []
        for i in range(num_hypothesis):
            branch_1_embeddings = self.branch_1(premises, attentions[i])
            branch_2_embeddings = self.branch_2(premises, attentions[i])
            concatenated_pooled.append(torch.cat((branch_1_embeddings, branch_2_embeddings), 0).reshape(1, -1))

        concatenated_pooled = torch.cat(concatenated_pooled, 0)

        logits = self.classifier(concatenated_pooled)
        output_dict = {"logits": logits}

        if answer_correctness_mask is not None:
            answer_correctness_mask = answer_correctness_mask.float().reshape(-1, 1)
            loss = self.loss(logits, answer_correctness_mask)
            if attention_loss is not None:
                loss += attention_loss
            output_dict["loss"] = loss
            result = logits.round()

            num_correct = 0
            for i in range(result.shape[0]):
                num_correct += result[i, 0] == answer_correctness_mask[i, 0]
            percentage_correct = (num_correct * 1.0) / (result.shape[0] * 1.0)
            if percentage_correct >= 0.99:
                self.statistics["100%"] += 1
            elif percentage_correct >= 0.75:
                self.statistics["75%"] += 1
            elif percentage_correct >= 0.50:
                self.statistics["50%"] += 1
            elif percentage_correct >= 0.25:
                self.statistics["25%"] += 1
            elif percentage_correct == 0.0:
                self.statistics["none"] += 1
            else:
                self.statistics["other"] += 1

            self.accuracy(result, answer_correctness_mask)

        return output_dict


    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = self.statistics
        metrics["accuracy"] = self.accuracy.get_metric(reset)
        return metrics
