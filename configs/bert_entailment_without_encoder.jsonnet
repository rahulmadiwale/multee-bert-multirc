{
  "dataset_reader": {
    "type": "multiple_correct_mcq_entailment",
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
    "type": "bert-entailment",
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
      "input_dim": 768,
      "num_layers": 3,
      "hidden_dims": [256, 64, 1],
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
                "question",
                "num_tokens"
            ],

        ],
  },
  "trainer": {
    "num_epochs": 1,
    "patience": 5,
    "cuda_device": -1,
    "grad_clipping": 5.0,
    "validation_metric": "+accuracy",
    "optimizer": {
      "type": "adagrad"
    }
  }
}
