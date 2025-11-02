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
- source_sentence: exclusive offer! order above rs.499 on blinkit-10 minute delivery
    app and get a free rs.100 zomato voucher. order:https://blinkit.to/blnkit/z/cz67aflt
    -blinkit
  sentences:
  - missing your favorite songs. get it back with 3months ad free music in hd quality
    with 6gb data for 15days. recharge now at rs108 vi.app.link/vi18e
  - movies? web series? sports? gaming? you're guaranteed to miss nothing with entertainment
    plus 699 postpaid! enjoy unlimited data-no questions asked & stream endless amazon
    prime & disney+ hotstar vip for 1 full year! upgrade now, click bit.ly/vipp699
  - hi, to fulfil your new airtel xstream fiber request we have assigned a representative
    aniket aher with mobile number 07947491646. expect a visit soon. please dont pay
    activation fees in cash.
- source_sentence: 'note: we have launched a part-time project, you can earn 100-5000
    rupees at home every day, please contact me wa.me/919574196172 webnrt'
  sentences:
  - congrats 77700519xx, 1500 bonus, ind vs nz, t20 match on my11circle!. first prize
    - 47lacs win now - http://1kx.in/vfk4yyudlcw
  - binge, play, surf all day. vi's data packs lead the way! 1)58=3gb,28d; 2)118=12gb,28d;
    3)418=100gb,56d; click to recharge bit.ly/vidatarc
  - get 50gb extra this republic day! recharge with rs1449 & get 1.5gb/day + 50gb
    extra + ul calls & more; validity 180 days! get now https://bit.ly/viextradata
- source_sentence: vi नंबर की do not disturb (dnd) सेवा सक्रिय करने से आप रजिस्टर्ड
    टेलीमार्केटर्स से टेलीमार्केटिंग एसएमएस या आनेवाले कॉल से बच सकते हैं, कृपया यहां
    क्लिक करें https://vi.app.link/dnd
  sentences:
  - dear sbi user, your a/c x8415-credited by rs.540 on 23aug25 transfer from miss
    shreya vishwas chavan ref no 523575617432 -sbi
  - 'hi apurv , you love pampering your skin and we love pampering you! bag our best
    sellers get upto 30% off code: bestyou shop - u1.mnge.co/dq2yqlb thedermaco.'
  - your a/c xx8415 is credited for inr 494.00 on 18-10-25 18:49:40 through upi.available
    bal inr 2316.67 (upi ref id 565721755611).download pnb one-pnb
- source_sentence: आपले नाव आपली कॉलरट्यून्स म्हणून सेट करा. bit.ly/nametune वर क्लिक
    करा
  sentences:
  - dear customer, rs. 30 free wallet cash expiring soon. use it now to save on your
    next zepto order. https://s.zepto.co.in/tc90/fts
  - dear sbi upi user, ur a/cx8415 credited by rs210 on 26may22 by (ref no 214614028252)
  - a/c xx8415 debited inr 103.00 dt 08-08-22 18:35 thru upi:222021967803.bal inr
    11108.07 not u?fwd this sms to 9264092640 to block upi.download pnb one-pnb
- source_sentence: a/c xx8415 debited inr 150.00 dt 21-05-23 22:42 thru upi:314165858288.bal
    inr 4170.18 not u?fwd this sms to 9264092640 to block upi.download pnb one-pnb
  sentences:
  - dear sbi upi user, ur a/cx8415 credited by rs10000 on 02jul22 by (ref no 218341471010)
  - vivo ipl mein aaj dekhiye rajasthan royals vs punjab kings ka zabardast muqabla
    live disney+ hotstar vip par. 401 ke recharge pe pao 1yr disney+ hotstar vip pack
    + 100gb data + ul calls;28 days. bit.ly/vihotstar1
  - win 12gb data free! play vi games fest from 23rd - 30th apr to with 12gb data
    free daily! play on vi app https://vi.app.link/smsgam2
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
    'a/c xx8415 debited inr 150.00 dt 21-05-23 22:42 thru upi:314165858288.bal inr 4170.18 not u?fwd this sms to 9264092640 to block upi.download pnb one-pnb',
    'win 12gb data free! play vi games fest from 23rd - 30th apr to with 12gb data free daily! play on vi app https://vi.app.link/smsgam2',
    'dear sbi upi user, ur a/cx8415 credited by rs10000 on 02jul22 by (ref no 218341471010)',
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 384]

# Get the similarity scores for the embeddings
similarities = model.similarity(embeddings, embeddings)
print(similarities)
# tensor([[ 1.0000, -0.2765,  0.9923],
#         [-0.2765,  1.0000, -0.2955],
#         [ 0.9923, -0.2955,  1.0000]])
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
  |         | sentence_0                                                                         | label                                                                                                                                      |
  |:--------|:-----------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------------------------------------------------------------|
  | type    | string                                                                             | int                                                                                                                                        |
  | details | <ul><li>min: 3 tokens</li><li>mean: 57.13 tokens</li><li>max: 256 tokens</li></ul> | <ul><li>0: ~11.10%</li><li>1: ~2.90%</li><li>2: ~1.60%</li><li>3: ~7.60%</li><li>4: ~31.90%</li><li>5: ~42.10%</li><li>6: ~2.80%</li></ul> |
* Samples:
  | sentence_0                                                                                                                                                           | label          |
  |:---------------------------------------------------------------------------------------------------------------------------------------------------------------------|:---------------|
  | <code>dear nikum apurv prabhakar, your otp for login is 19813. naacit</code>                                                                                         | <code>5</code> |
  | <code>a/c xx8415 debited inr 190.00 dt 01-06-23 12:22 thru upi:315220568602.bal inr 910.07 not u?fwd this sms to 9264092640 to block upi.download pnb one-pnb</code> | <code>5</code> |
  | <code>kidhar hain !??</code>                                                                                                                                         | <code>2</code> |
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
| 0.8143 | 500  | 1.9084        |
| 1.6287 | 1000 | 1.7343        |
| 2.4430 | 1500 | 1.6394        |


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