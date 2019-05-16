# bert-cosine-sim
Fine-tune BERT to generate sentence embedding for cosine similarity. Most of the code is based on [huggingface's bert project](https://github.com/huggingface/pytorch-pretrained-BERT).

# Main point
Add a FC layer + tanh activation on the `CLS` token to generate sentence embedding. Fine-tune the model on the STS-B dataset, by reducing the cosine similarity loss. With a embedding size of 1024, and trained 20 epochs, it can achieve 0.830 Pearson score.

```python
 class BertPairSim(BertPreTrainedModel):
    def __init__(self, config, emb_size=1024):
        super(BertPairSim, self).__init__(config)
        self.emb_size = emb_size
        self.bert = BertModel(config)
        self.emb = nn.Linear(config.hidden_size, emb_size)
        self.activation = nn.Tanh()
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, attention_mask):
        _, pooled_output = self.bert(input_ids, None, attention_mask,
                                     output_all_encoded_layers=False)
        emb = self.activation(self.emb(pooled_output))
        return emb
```
