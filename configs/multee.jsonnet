{
  "dataset_reader": {
    "type": "multiple_correct_mcq_multee",
    "token_indexers": {
            "tokens": {
                "type": "bert-pretrained",
                "pretrained_model": "bert-base-cased",
                "max_pieces": 100,
                "do_lowercase": false
            },
        }
  },
  "train_data_path": "data/train-sample.jsonl",
  "validation_data_path": "data/dev-sample.jsonl",
  "model": {
    "type": "bert-multee",
    "attention_module": {
        "type": "bert-entailment-with-encoder",
        "text_field_embedder": {
          "token_embedders": {
            "tokens": {
              "type": "bert-pretrained",
              "pretrained_model": "bert-base-cased",
              "top_layer_only": true
            }
          },
          "allow_unmatched_keys": true
        },
        "classifier": {
          "input_dim": 400,
          "num_layers": 3,
          "hidden_dims": [128, 64, 1],
          "activations": ["linear", "linear", 'sigmoid']
        },
        "encoder": {
          "type": "lstm",
          "bidirectional": true,
          "input_size": 768,
          "hidden_size": 200,
          "num_layers": 1,
        }
     },
     "classifier": {
      "input_dim": 1536,
      "num_layers": 3,
      "hidden_dims": [512, 256, 1],
      "activations": ["linear", "linear", 'sigmoid']
    }
  },
  "iterator": {
    "type": "bucket",
    "batch_size": 1,
    "sorting_keys": [
            [
                "premises",
                "list_num_tokens"
            ],
            [
                "hypotheses",
                "list_num_tokens"
            ],

        ],
  },
  "trainer": {
    "num_epochs": 1,
    "patience": 1,
    "cuda_device": -1,
    "grad_clipping": 5.0,
    "validation_metric": "+accuracy",
    "optimizer": {
      "type": "adagrad"
    }
  }
}
