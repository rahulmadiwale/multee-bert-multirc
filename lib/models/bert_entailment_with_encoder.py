from typing import Dict

import torch
from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import TextFieldEmbedder, Seq2VecEncoder
from allennlp.modules.feedforward import FeedForward
from allennlp.nn.util import get_text_field_mask
from allennlp.training.metrics import BooleanAccuracy
from overrides import overrides


@Model.register("bert-entailment-with-encoder")
class BertEntailmentWithEncoder(Model):

    def __init__(self, vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 classifier: FeedForward,
                 encoder: Seq2VecEncoder) -> None:
        super(BertEntailmentWithEncoder, self).__init__(vocab)
        self.text_field_embedder = text_field_embedder
        self.classifier = classifier
        self.encoder = encoder
        self.loss = torch.nn.MSELoss()
        self.accuracy = BooleanAccuracy()
        self.vocab = vocab

    @overrides
    def forward(self,
                premises: Dict[str, torch.LongTensor],
                # hypotheses: Dict[str, torch.LongTensor],
                # paragraph: Dict[str, torch.LongTensor],
                question: Dict[str, torch.LongTensor],
                # answer_correctness_mask: torch.IntTensor = None,
                relevance_presence_mask: torch.Tensor = None) -> Dict[str, torch.Tensor]:

        def squeeze_tokens(tokens):
            for key in tokens.keys():
                tokens[key] = torch.squeeze(tokens[key])

        # squeeze_tokens(question)
        squeeze_tokens(premises)
        # squeeze_tokens(hypotheses)

        #print("question: ", question.keys(), [x.shape for _, x in question.items()])
        #print("premises: ", premises.keys(), [x.shape for _, x in premises.items()])
        # print("hypotheses: ", hypotheses.keys(), [x.shape for _, x in hypotheses.items()])
        # print("answer_correctness_mask", answer_correctness_mask.shape, answer_correctness_mask)
        #print("relevance_presence_mask", relevance_presence_mask.shape, relevance_presence_mask)

        question_mask = get_text_field_mask({'tokens': question['tokens']})
        premises_mask = get_text_field_mask({'tokens': premises['tokens']})
        # hypotheses_mask = get_text_field_mask({'tokens': hypotheses['tokens']})

        #print(question_mask.shape, premises_mask.shape)

        premises_with_questions = torch.cat(
            (premises['tokens'], question['tokens'][:, 1:].repeat(premises['tokens'].shape[0], 1)), 1)
        premises_with_questions_mask = torch.cat(
            (premises_mask, question_mask[:, 1:].repeat(premises['tokens'].shape[0], 1)), 1)
        #print("CONCAT: ", premises_with_questions.shape)
        #print("CONCAT MASK: ", premises_with_questions_mask.shape)

        premises_embedding = self.text_field_embedder({
            'tokens': premises_with_questions,
            'mask': premises_with_questions_mask
        })
        # print("Premises Embedding: ", premises_embedding.shape)
        encoded_premises = self.encoder(premises_embedding, premises_with_questions_mask)
        # print(encoded_premises.shape)

        logits = self.classifier(encoded_premises)
        #print("LOGITS: ", logits.shape, logits.reshape(1, -1))
        #print("Labels: ", relevance_presence_mask.shape, relevance_presence_mask)
        output_dict = {"logits": logits}

        # print(logits)

        if relevance_presence_mask is not None:
            loss = self.loss(logits, relevance_presence_mask.reshape(-1, 1))
            output_dict["loss"] = loss
            # print("\nLogits: ", logits.reshape(1, -1))
            # print("\nLABELS: ", relevance_presence_mask.reshape(1, -1))
            # print("\nLOSS: ", loss)
            self.accuracy(logits.round(), relevance_presence_mask.reshape(-1, 1))

        return output_dict

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {
            "accuracy": self.accuracy.get_metric(reset)
        }