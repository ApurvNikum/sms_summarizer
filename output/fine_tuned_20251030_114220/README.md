---
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- dense
- generated_from_trainer
- dataset_size:9805
- loss:CustomClassificationLoss
base_model: sentence-transformers/all-MiniLM-L6-v2
widget:
- source_sentence: dear student, why wait for cap rounds book your seat now. link
    https //url_mask mitaoe, alandi, pune phonenum_mask thanks
  sentences:
  - hi apurv, make a smart career move with career camp! switch to a tech job role
    and get up to 200% hike. apply now weurl.co/nsehla -coding ninjas
  - your a/c xx8415 is credited for inr 35.00 on 15-10-22 22 55 through upi.available
    bal inr 1365.97 upi ref id 228876626349 .download pnb one-pnb
  - otp for aadhaar xx0998 is 269326 valid for 10 mins . to update aadhaar, upload
    documents on myaadhaar.uidai.gov.in or visit aadhaar center. call 1947 for info.
    -uidai
- source_sentence: yami gautam is back with a new thriller movie - the lost on vi
    app. watch it now click https //url_mask
  sentences:
  - dear student, your 24mcb0028 otp for vtop mobile application is 991891 - vit
  - introducing screen it only on pvr inox app! create your own show, share it with
    friends, communities earn amazing rewards. the more you share, the more you earn
    https //pvr.im/inoxmo/y9dq0z8x
  - . . . ..
- source_sentence: ', /sms/ https //sancharsaathi.gov.in -rbi'
  sentences:
  - your 2022 wrapped! celebrate this year s most loved movies series on vi movies
    tv. click https //url_mask
  - a/c xx8415 debited inr 94.17 dt 01-03-23 12 47 thru upi 342672950487.bal inr 5952.18
    not u?fwd this sms to phonenum_mask to block upi.download pnb one-pnb
  - vi exclusive! offer only for you!719 2gb/d 10gb 28d ul,84d recharge dial 121 or
    url_mask
- source_sentence: bindaas english bolna seekho! 3 din tak free live classes book
    karo sirf vi app par! https //url_mask
  sentences:
  - hi apurv ,were celebrating the bond between science skincare get upto 30% off
    on best sellers code bestyou shop - https //u1.mnge.co/5mdykle thedermaco
  - dear customer,your a/c xxxxxxxx8415 is debited for rs 80.00 on 24-04-22 21.09.40
    through upi.available bal rs 7059.65 upi ref no 211421030865 .if not done by you,pl
    forward this sms from registered mobile to phonenum_mask to report unauthorized
    txn block upi. download pnb one-pnb
  - a/c xx8415 debited inr 1.00 dt 10-07-22 13 41 thru upi 219113604944.bal inr 5392.11
    not u?fwd this sms to phonenum_mask to block upi.download pnb one-pnb
- source_sentence: https //amazon.in/a/c/r/e0y48jc1bztbpzcge9ljpkg90 amazon sign-in
    from mh, in. tap link to respond.
  sentences:
  - admission notice for b.tech/b.e other courses in top 10 pvt. engineering colleges
    of pune under management nri quota . call/whatsapp-phonenum_mask
  - can you call back later?
  - hello! we are now on whatsapp! pay your bills, change your tariff, activate services
    do much more. simply click wa.me/919654297000?text hi say hi
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
    'https //amazon.in/a/c/r/e0y48jc1bztbpzcge9ljpkg90 amazon sign-in from mh, in. tap link to respond.',
    'hello! we are now on whatsapp! pay your bills, change your tariff, activate services do much more. simply click wa.me/919654297000?text hi say hi',
    'admission notice for b.tech/b.e other courses in top 10 pvt. engineering colleges of pune under management nri quota . call/whatsapp-phonenum_mask',
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 384]

# Get the similarity scores for the embeddings
similarities = model.similarity(embeddings, embeddings)
print(similarities)
# tensor([[ 1.0000, -0.2913,  0.3579],
#         [-0.2913,  1.0000,  0.5938],
#         [ 0.3579,  0.5938,  1.0000]])
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

* Size: 9,805 training samples
* Columns: <code>sentence_0</code> and <code>label</code>
* Approximate statistics based on the first 1000 samples:
  |         | sentence_0                                                                         | label                                                                                                                                                                         |
  |:--------|:-----------------------------------------------------------------------------------|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
  | type    | string                                                                             | int                                                                                                                                                                           |
  | details | <ul><li>min: 3 tokens</li><li>mean: 49.35 tokens</li><li>max: 256 tokens</li></ul> | <ul><li>0: ~0.70%</li><li>1: ~6.00%</li><li>2: ~1.40%</li><li>3: ~4.90%</li><li>4: ~5.60%</li><li>5: ~2.60%</li><li>6: ~32.90%</li><li>7: ~45.70%</li><li>8: ~0.20%</li></ul> |
* Samples:
  | sentence_0                                                                                                                                                                                                                                                | label          |
  |:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:---------------|
  | <code>someday we ll go to the cinema again. until then, get latest movies shows delivered home safely, with entertainment plus 699 postpaid. stream endless movies on amazon prime disney hotstar vip for 1 full year! upgrade now, click url_mask</code> | <code>5</code> |
  | <code>thanks for ordering from blrpulse, it is confirmed. here is the receipt- https //url_mask blr pulse by graymatter</code>                                                                                                                            | <code>0</code> |
  | <code>alert 50% data is consumed. get 2gb at rs33 till midnight. recharge now i.airtel.in/dtpck</code>                                                                                                                                                    | <code>7</code> |
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
| 0.8157 | 500  | 2.1616        |
| 1.6313 | 1000 | 1.9701        |
| 2.4470 | 1500 | 1.8606        |


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