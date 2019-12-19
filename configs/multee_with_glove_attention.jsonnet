{
  "dataset_reader": {
    "type": "multiple_correct_mcq_multee_with_glove_attention",
    "token_indexers": {
            "tokens": {
                "type": "bert-pretrained",
                "pretrained_model": "bert-base-cased",
                "max_pieces": 100,
                "do_lowercase": false
            },
        },
    "glove_token_indexers": {
            "tokens": {
                "type": "single_id",
                "lowercase_tokens": false,
                "namespace": "glove"
            }
        }
  },
  "train_data_path": "data/train-sample.jsonl",
  "validation_data_path": "data/dev-sample.jsonl",
  "model": {
    "type": "bert-multee-glove",
    "attention_module": {
        "type": "glove-entailment",
        "text_field_embedder": {
      "tokens": {
        "type": "embedding",
        "pretrained_file": "https://s3-us-west-2.amazonaws.com/allennlp/datasets/glove/glove.6B.100d.txt.gz",
        "vocab_namespace": "glove",
        "embedding_dim": 100,
        "trainable": true
      },
          "allow_unmatched_keys": true
        },
        "classifier": {
          "input_dim": 128,
          "num_layers": 3,
          "hidden_dims": [64, 32, 1],
          "activations": ["linear", "linear", 'sigmoid']
        },
        "encoder": {
          "type": "lstm",
          "bidirectional": true,
          "input_size": 100,
          "hidden_size": 64,
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
