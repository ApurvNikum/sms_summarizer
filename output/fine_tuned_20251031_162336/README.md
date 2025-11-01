---
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- dense
- generated_from_trainer
- dataset_size:9820
- loss:CustomClassificationLoss
base_model: sentence-transformers/all-MiniLM-L6-v2
widget:
- source_sentence: आपली आवडती geet/name कॉलरट्यून्स म्हणून सेट करा. क्लिक bit.ly/vitunes5
  sentences:
  - incoming calls and otps will be stopped from tonight on your airtel number. to
    continue with your services, recharge now https://u.airtel.in/fdp1
  - your a/c xx8415 is credited for inr 215.00 on 06-04-23 14:44 through upi.available
    bal inr 11084.12 (upi ref id 346235864936).download pnb one-pnb
  - dear customer,your a/c xxxxxxxx8415 is credited for rs 70.00 on 08-05-22 18.59.56
    through upi.available bal rs 6642.99 (upi ref id 212818129738)-pnb
- source_sentence: last-call:00:00:15, charge:rs0.00, main-bal:rs9.52, ulpack-exp:05-dec-2022
  sentences:
  - '[part-time recruitment] happy new year 2021, you can get a commission of 2000rs
    every day. come with me and give you 100rs now whats-app wa.me/919593004532'
  - apurv, we found india's favourite offer! buy 1, get 3 free @ myntra right to fashion
    sale shop now with free shipping http://smsd.in/bcsplclrx
  - a/c xx8415 debited inr 1252.00 dt 28-08-24 13:09 thru upi:424172944254.bal inr
    10275.55 not u?fwd this sms to 9264092640 to block upi.download pnb one-pnb
- source_sentence: enjoy the india vs australia final match without interruption!
    recharge with rs.49 & get 6gb data for 24 hrs. don't miss a moment! https://phon.pe/vimg
  sentences:
  - a/c xx8415 debited inr 160.00 dt 15-12-22 13:37 thru upi:234980913676.bal inr
    1100.16 not u?fwd this sms to 9264092640 to block upi.download pnb one-pnb
  - dear apurv, participate in our scholarship test & get up to 100% scholarship on
    all courses. register now weurl.co/gr5ts1 -coding ninjas
  - 'your amazon web services (aws) verification code is: 4481'
- source_sentence: congratulations for claiming 3 month of airtel xstream play premium.
    happy streaming on your favourite movies and shows.
  sentences:
  - your a/c xx8415 is credited for inr 50.00 on 31-03-23 12:47 through upi.available
    bal inr 6938.12 (upi ref id 309018918287).download pnb one-pnb
  - 'alert 50%: data is consumed. get 2gb at rs33 till midnight. recharge now i.airtel.in/dtpck'
  - hi, the reference no. 31-1000005334125 related to your airtel number 7770051925
    has been closed. for queries, please contact customer care at 121 or visit airtel
    thanks app u.airtel.in/c_srtrck
- source_sentence: dear customer,your a/c xxxxxxxx8415 is credited for rs 80.00 on
    24-05-22 20.08.06 through upi.available bal rs 3948.17 (upi ref id 214473135622)-pnb
  sentences:
  - there's no end to learning! access the best for your business with eunimart, hubbler,
    fiskl, and vodafone tuesday offers, only on the vi app https://www.myvi.in/vi-app
  - माझा मतदारसंघ माझं व्हिजन मा.श्री.गंगाधर विठ्ठल बधे चिन्ह- एअरकंडिशनर अपक्ष उमेदवार
    हडपसर विधानसभा मतदारसंघ विकास गंगाधर बधे
  - data alert! 500 mb left on your daily data pack. click bit.ly/vidatarc or dial
    *121# to buy a data pack.
pipeline_tag: sentence-similarity
library_name: sentence-transformers
---

# SentenceTransformer based on sentence-transformers/all-MiniLM-L6-v2

This is a [sentence-transformers](https://www.SBERT.net) model finetuned from [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2). It maps sentences & paragraphs to a 384-dimensional dense vector space and can be used for semantic textual similarity, semantic search, paraphrase mining, text classification, clustering, and more.

## Model Details

### Model Description
- **Model Type:** Sentence Transformer
- **Base model:** [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) <!-- at revision c9745ed1d9f207416be6d2e6f8de32d1f16199bf -->
- **Maximum Sequence Length:** 256 tokens
- **Output Dimensionality:** 384 dimensions
- **Similarity Function:** Cosine Similarity
<!-- - **Training Dataset:** Unknown -->
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Documentation:** [Sentence Transformers Documentation](https://sbert.net)
- **Repository:** [Sentence Transformers on GitHub](https://github.com/huggingface/sentence-transformers)
- **Hugging Face:** [Sentence Transformers on Hugging Face](https://huggingface.co/models?library=sentence-transformers)

### Full Model Architecture

```
SentenceTransformer(
  (0): Transformer({'max_seq_length': 256, 'do_lower_case': False, 'architecture': 'BertModel'})
  (1): Pooling({'word_embedding_dimension': 384, 'pooling_mode_cls_token': False, 'pooling_mode_mean_tokens': True, 'pooling_mode_max_tokens': False, 'pooling_mode_mean_sqrt_len_tokens': False, 'pooling_mode_weightedmean_tokens': False, 'pooling_mode_lasttoken': False, 'include_prompt': True})
  (2): Normalize()
)
```

## Usage

### Direct Usage (Sentence Transformers)

First install the Sentence Transformers library:

```bash
pip install -U sentence-transformers
```

Then you can load this model and run inference.
```python
from sentence_transformers import SentenceTransformer

# Download from the 🤗 Hub
model = SentenceTransformer("sentence_transformers_model_id")
# Run inference
sentences = [
    'dear customer,your a/c xxxxxxxx8415 is credited for rs 80.00 on 24-05-22 20.08.06 through upi.available bal rs 3948.17 (upi ref id 214473135622)-pnb',
    'data alert! 500 mb left on your daily data pack. click bit.ly/vidatarc or dial *121# to buy a data pack.',
    "there's no end to learning! access the best for your business with eunimart, hubbler, fiskl, and vodafone tuesday offers, only on the vi app https://www.myvi.in/vi-app",
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 384]

# Get the similarity scores for the embeddings
similarities = model.similarity(embeddings, embeddings)
print(similarities)
# tensor([[ 1.0000, -0.2178, -0.3075],
#         [-0.2178,  1.0000,  0.6798],
#         [-0.3075,  0.6798,  1.0000]])
```

<!--
### Direct Usage (Transformers)

<details><summary>Click to see the direct usage in Transformers</summary>

</details>
-->

<!--
### Downstream Usage (Sentence Transformers)

You can finetune this model on your own dataset.

<details><summary>Click to expand</summary>

</details>
-->

<!--
### Out-of-Scope Use

*List how the model may foreseeably be misused and address what users ought not to do with the model.*
-->

<!--
## Bias, Risks and Limitations

*What are the known or foreseeable issues stemming from this model? You could also flag here known failure cases or weaknesses of the model.*
-->

<!--
### Recommendations

*What are recommendations with respect to the foreseeable issues? For example, filtering explicit content.*
-->

## Training Details

### Training Dataset

#### Unnamed Dataset

* Size: 9,820 training samples
* Columns: <code>sentence_0</code> and <code>label</code>
* Approximate statistics based on the first 1000 samples:
  |         | sentence_0                                                                        | label                                                                                                                                      |
  |:--------|:----------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------------------------------------------------------------|
  | type    | string                                                                            | int                                                                                                                                        |
  | details | <ul><li>min: 3 tokens</li><li>mean: 56.6 tokens</li><li>max: 256 tokens</li></ul> | <ul><li>0: ~11.80%</li><li>1: ~2.50%</li><li>2: ~2.10%</li><li>3: ~6.50%</li><li>4: ~32.20%</li><li>5: ~41.70%</li><li>6: ~3.20%</li></ul> |
* Samples:
  | sentence_0                                                                                                                                                                                | label          |
  |:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:---------------|
  | <code>activating dnd ( do not disturb) service allows you to avoid telemarketing sms or call from registered telemarketers , to activate please click here https://vi.app.link/dnd</code> | <code>4</code> |
  | <code>complete our survey to receive a 500 rupee flipkart gift card. click on the link to begin: https://ql.tc/b/6nuveyiw</code>                                                          | <code>3</code> |
  | <code>a/c xx8415 debited inr 220.00 dt 13-05-23 21:59 thru upi:313385043830.bal inr 6600.85 not u?fwd this sms to 9264092640 to block upi.download pnb one-pnb</code>                     | <code>5</code> |
* Loss: <code>__main__.CustomClassificationLoss</code>

### Training Hyperparameters
#### Non-Default Hyperparameters

- `per_device_train_batch_size`: 16
- `per_device_eval_batch_size`: 16
- `multi_dataset_batch_sampler`: round_robin

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `overwrite_output_dir`: False
- `do_predict`: False
- `eval_strategy`: no
- `prediction_loss_only`: True
- `per_device_train_batch_size`: 16
- `per_device_eval_batch_size`: 16
- `per_gpu_train_batch_size`: None
- `per_gpu_eval_batch_size`: None
- `gradient_accumulation_steps`: 1
- `eval_accumulation_steps`: None
- `torch_empty_cache_steps`: None
- `learning_rate`: 5e-05
- `weight_decay`: 0.0
- `adam_beta1`: 0.9
- `adam_beta2`: 0.999
- `adam_epsilon`: 1e-08
- `max_grad_norm`: 1
- `num_train_epochs`: 3
- `max_steps`: -1
- `lr_scheduler_type`: linear
- `lr_scheduler_kwargs`: {}
- `warmup_ratio`: 0.0
- `warmup_steps`: 0
- `log_level`: passive
- `log_level_replica`: warning
- `log_on_each_node`: True
- `logging_nan_inf_filter`: True
- `save_safetensors`: True
- `save_on_each_node`: False
- `save_only_model`: False
- `restore_callback_states_from_checkpoint`: False
- `no_cuda`: False
- `use_cpu`: False
- `use_mps_device`: False
- `seed`: 42
- `data_seed`: None
- `jit_mode_eval`: False
- `bf16`: False
- `fp16`: False
- `fp16_opt_level`: O1
- `half_precision_backend`: auto
- `bf16_full_eval`: False
- `fp16_full_eval`: False
- `tf32`: None
- `local_rank`: 0
- `ddp_backend`: None
- `tpu_num_cores`: None
- `tpu_metrics_debug`: False
- `debug`: []
- `dataloader_drop_last`: False
- `dataloader_num_workers`: 0
- `dataloader_prefetch_factor`: None
- `past_index`: -1
- `disable_tqdm`: False
- `remove_unused_columns`: True
- `label_names`: None
- `load_best_model_at_end`: False
- `ignore_data_skip`: False
- `fsdp`: []
- `fsdp_min_num_params`: 0
- `fsdp_config`: {'min_num_params': 0, 'xla': False, 'xla_fsdp_v2': False, 'xla_fsdp_grad_ckpt': False}
- `fsdp_transformer_layer_cls_to_wrap`: None
- `accelerator_config`: {'split_batches': False, 'dispatch_batches': None, 'even_batches': True, 'use_seedable_sampler': True, 'non_blocking': False, 'gradient_accumulation_kwargs': None}
- `parallelism_config`: None
- `deepspeed`: None
- `label_smoothing_factor`: 0.0
- `optim`: adamw_torch
- `optim_args`: None
- `adafactor`: False
- `group_by_length`: False
- `length_column_name`: length
- `project`: huggingface
- `trackio_space_id`: trackio
- `ddp_find_unused_parameters`: None
- `ddp_bucket_cap_mb`: None
- `ddp_broadcast_buffers`: False
- `dataloader_pin_memory`: True
- `dataloader_persistent_workers`: False
- `skip_memory_metrics`: True
- `use_legacy_prediction_loop`: False
- `push_to_hub`: False
- `resume_from_checkpoint`: None
- `hub_model_id`: None
- `hub_strategy`: every_save
- `hub_private_repo`: None
- `hub_always_push`: False
- `hub_revision`: None
- `gradient_checkpointing`: False
- `gradient_checkpointing_kwargs`: None
- `include_inputs_for_metrics`: False
- `include_for_metrics`: []
- `eval_do_concat_batches`: True
- `fp16_backend`: auto
- `push_to_hub_model_id`: None
- `push_to_hub_organization`: None
- `mp_parameters`: 
- `auto_find_batch_size`: False
- `full_determinism`: False
- `torchdynamo`: None
- `ray_scope`: last
- `ddp_timeout`: 1800
- `torch_compile`: False
- `torch_compile_backend`: None
- `torch_compile_mode`: None
- `include_tokens_per_second`: False
- `include_num_input_tokens_seen`: no
- `neftune_noise_alpha`: None
- `optim_target_modules`: None
- `batch_eval_metrics`: False
- `eval_on_start`: False
- `use_liger_kernel`: False
- `liger_kernel_config`: None
- `eval_use_gather_object`: False
- `average_tokens_across_devices`: True
- `prompts`: None
- `batch_sampler`: batch_sampler
- `multi_dataset_batch_sampler`: round_robin
- `router_mapping`: {}
- `learning_rate_mapping`: {}

</details>

### Training Logs
| Epoch  | Step | Training Loss |
|:------:|:----:|:-------------:|
| 0.8143 | 500  | 1.913         |
| 1.6287 | 1000 | 1.7427        |
| 2.4430 | 1500 | 1.6479        |


### Framework Versions
- Python: 3.12.6
- Sentence Transformers: 5.1.2
- Transformers: 4.57.1
- PyTorch: 2.5.1+cu121
- Accelerate: 1.11.0
- Datasets: 4.3.0
- Tokenizers: 0.22.1

## Citation

### BibTeX

#### Sentence Transformers
```bibtex
@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/1908.10084",
}
```

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--
## Model Card Contact

*Provides a way for people who have updates to the Model Card, suggestions, or questions, to contact the Model Card authors.*
-->