---
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- dense
- generated_from_trainer
- dataset_size:43562
- loss:MarginInfoNCELossWithStats
base_model: BAAI/bge-small-en-v1.5
widget:
- source_sentence: 'Represent this web action for retrieving relevant DOM elements:
    [ACTION] Click the "Dougray Scott" actor link. [HISTORY] The page was loaded,
    and the "More Film Information and Actions" link was clicked to reveal the actors.'
  sentences:
  - '[TAG] a [TEXT] Dougray Scott'
  - '[TAG] a [CLASS] cyljGc [HREF] travel worlds table [TEXT] World’s Table'
  - '[TAG] a [CLASS] flex-next [TEXT] Next'
- source_sentence: 'Represent this web action for retrieving relevant DOM elements:
    [ACTION] Click the "Add to Wishlist" button for "Where''s Spot?" [HISTORY] The
    page was loaded, and the user clicked on the title "Where''s Spot?".'
  sentences:
  - '[TAG] a [CLASS] aps-btn-boxed aps-add-compare [TEXT] Add to Compare'
  - '[TAG] div [CLASS] calendar-body__cell [TEXT] 16'
  - '[TAG] div [CLASS] add-to-wishlist add-to-wishlist-btn [TEXT] NONE'
- source_sentence: 'Represent this web action for retrieving relevant DOM elements:
    [ACTION] Navigate to Biological Physics'
  sentences:
  - '[TAG] a [ID] physics.bio-ph [HREF] list physics [TEXT] Biological Physics'
  - '[TAG] a [TEXT] Senior Product Manager'
  - '[TAG] button [CLASS] btn btn-tool btn-mini [TEXT] Apply'
- source_sentence: 'Represent this web action for retrieving relevant DOM elements:
    [ACTION] Click "Magic" link [HISTORY] N/A'
  sentences:
  - '[TAG] a [HREF] business media telecom [TEXT] Media & Telecom'
  - '[TAG] a [CLASS] AnchorLink clr-black [TEXT] Magic'
  - '[TAG] button [CLASS] kmx-menu-item-button [TEXT] $99 or less'
- source_sentence: 'Represent this web action for retrieving relevant DOM elements:
    [ACTION] Click on the "REFINE YOUR SEARCH" button. [HISTORY] The user viewed the
    search results and is now clicking the refine search button.'
  sentences:
  - '[TAG] button [CLASS] btn btn-primary green-hover refine-search-btn [TEXT] REFINE
    YOUR SEARCH'
  - '[TAG] a [CLASS] lnkInquiry btnInq btn [TEXT] Inquiry'
  - '[TAG] a [TEXT] Tohoku Univ'
pipeline_tag: sentence-similarity
library_name: sentence-transformers
---

# SentenceTransformer based on BAAI/bge-small-en-v1.5

This is a [sentence-transformers](https://www.SBERT.net) model finetuned from [BAAI/bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5). It maps sentences & paragraphs to a 256-dimensional dense vector space and can be used for semantic textual similarity, semantic search, paraphrase mining, text classification, clustering, and more.

## Model Details

### Model Description
- **Model Type:** Sentence Transformer
- **Base model:** [BAAI/bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5) <!-- at revision 5c38ec7c405ec4b44b94cc5a9bb96e735b38267a -->
- **Maximum Sequence Length:** 256 tokens
- **Output Dimensionality:** 256 dimensions
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
  (2): Dense({'in_features': 384, 'out_features': 256, 'bias': True, 'activation_function': 'torch.nn.modules.activation.Tanh'})
  (3): Normalize()
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
    'Represent this web action for retrieving relevant DOM elements: [ACTION] Click on the "REFINE YOUR SEARCH" button. [HISTORY] The user viewed the search results and is now clicking the refine search button.',
    '[TAG] button [CLASS] btn btn-primary green-hover refine-search-btn [TEXT] REFINE YOUR SEARCH',
    '[TAG] a [TEXT] Tohoku Univ',
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 256]

# Get the similarity scores for the embeddings
similarities = model.similarity(embeddings, embeddings)
print(similarities)
# tensor([[ 1.0000,  0.9617, -0.0978],
#         [ 0.9617,  1.0000, -0.0918],
#         [-0.0978, -0.0918,  1.0000]])
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

* Size: 43,562 training samples
* Columns: <code>anchor</code> and <code>positive</code>
* Approximate statistics based on the first 1000 samples:
  |         | anchor                                                                              | positive                                                                           |
  |:--------|:------------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------|
  | type    | string                                                                              | string                                                                             |
  | details | <ul><li>min: 19 tokens</li><li>mean: 52.34 tokens</li><li>max: 239 tokens</li></ul> | <ul><li>min: 6 tokens</li><li>mean: 20.55 tokens</li><li>max: 143 tokens</li></ul> |
* Samples:
  | anchor                                                                                                                                                                                                                                                                                                      | positive                                                               |
  |:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-----------------------------------------------------------------------|
  | <code>Represent this web action for retrieving relevant DOM elements: [ACTION] Search for a flight on May 9 and return on May 16 from Tel Aviv to Venice with plus type fare option. [HISTORY] [span]  Venice Treviso -> CLICK; [div]  May -> CLICK; [generic]  9 -> CLICK</code>                           | <code>[TAG] div [CLASS] calendar-body__cell [TEXT] 16</code>           |
  | <code>Represent this web action for retrieving relevant DOM elements: [ACTION] Click on the link for "Kentavious Caldwell-Pope" to view his profile. [HISTORY] The user has clicked on all the column headers and clicked on George Hill's, Dwight Powell's and Shake Milton's links.</code>                | <code>[TAG] a [CLASS] C(#333) [TEXT] Kentavious Caldwell-Pope</code>   |
  | <code>Represent this web action for retrieving relevant DOM elements: [ACTION] find schedule for Long Island Rail Road & Metro-North Railroad from Bay Shore to Breakneck Ridge on Thu, Mar 23, 08:37 AM. [HISTORY] [select]  1 -> SELECT: 8; [select]  00 -> SELECT: 37; [select]  AM -> SELECT: AM</code> | <code>[TAG] button [CLASS] find-schedules [TEXT] Find Schedules</code> |
* Loss: <code>margin_infonce_loss.MarginInfoNCELossWithStats</code> with these parameters:
  ```json
  {
      "temperature": 0.05,
      "margin": 0.3
  }
  ```

### Training Hyperparameters
#### Non-Default Hyperparameters

- `eval_strategy`: epoch
- `per_device_train_batch_size`: 128
- `learning_rate`: 2e-05
- `num_train_epochs`: 40
- `warmup_ratio`: 0.1
- `fp16`: True
- `dataloader_drop_last`: True
- `load_best_model_at_end`: True

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `overwrite_output_dir`: False
- `do_predict`: False
- `eval_strategy`: epoch
- `prediction_loss_only`: True
- `per_device_train_batch_size`: 128
- `per_device_eval_batch_size`: 8
- `per_gpu_train_batch_size`: None
- `per_gpu_eval_batch_size`: None
- `gradient_accumulation_steps`: 1
- `eval_accumulation_steps`: None
- `torch_empty_cache_steps`: None
- `learning_rate`: 2e-05
- `weight_decay`: 0.0
- `adam_beta1`: 0.9
- `adam_beta2`: 0.999
- `adam_epsilon`: 1e-08
- `max_grad_norm`: 1.0
- `num_train_epochs`: 40
- `max_steps`: -1
- `lr_scheduler_type`: linear
- `lr_scheduler_kwargs`: {}
- `warmup_ratio`: 0.1
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
- `fp16`: True
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
- `dataloader_drop_last`: True
- `dataloader_num_workers`: 0
- `dataloader_prefetch_factor`: None
- `past_index`: -1
- `disable_tqdm`: False
- `remove_unused_columns`: True
- `label_names`: None
- `load_best_model_at_end`: True
- `ignore_data_skip`: False
- `fsdp`: []
- `fsdp_min_num_params`: 0
- `fsdp_config`: {'min_num_params': 0, 'xla': False, 'xla_fsdp_v2': False, 'xla_fsdp_grad_ckpt': False}
- `fsdp_transformer_layer_cls_to_wrap`: None
- `accelerator_config`: {'split_batches': False, 'dispatch_batches': None, 'even_batches': True, 'use_seedable_sampler': True, 'non_blocking': False, 'gradient_accumulation_kwargs': None}
- `parallelism_config`: None
- `deepspeed`: None
- `label_smoothing_factor`: 0.0
- `optim`: adamw_torch_fused
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
- `multi_dataset_batch_sampler`: proportional
- `router_mapping`: {}
- `learning_rate_mapping`: {}

</details>

### Training Logs
<details><summary>Click to expand</summary>

| Epoch    | Step     | Training Loss |
|:--------:|:--------:|:-------------:|
| 0.2941   | 50       | 9.4657        |
| 0.5882   | 100      | 6.804         |
| 0.8824   | 150      | 3.8678        |
| 1.0      | 170      | -             |
| 1.1765   | 200      | 2.9304        |
| 1.4706   | 250      | 2.6245        |
| 1.7647   | 300      | 2.4865        |
| 2.0      | 340      | -             |
| 2.0588   | 350      | 2.4415        |
| 2.3529   | 400      | 2.3301        |
| 2.6471   | 450      | 2.2199        |
| 2.9412   | 500      | 2.2165        |
| 3.0      | 510      | -             |
| 3.2353   | 550      | 2.1846        |
| 3.5294   | 600      | 2.1014        |
| 3.8235   | 650      | 1.9946        |
| 4.0      | 680      | -             |
| 4.1176   | 700      | 1.987         |
| 4.4118   | 750      | 1.9541        |
| 4.7059   | 800      | 1.9296        |
| 5.0      | 850      | 1.8766        |
| 5.2941   | 900      | 1.7951        |
| 5.5882   | 950      | 1.8032        |
| 5.8824   | 1000     | 1.7315        |
| 6.0      | 1020     | -             |
| 6.1765   | 1050     | 1.7671        |
| 6.4706   | 1100     | 1.6529        |
| 6.7647   | 1150     | 1.7324        |
| 7.0      | 1190     | -             |
| 7.0588   | 1200     | 1.7029        |
| 7.3529   | 1250     | 1.6183        |
| 7.6471   | 1300     | 1.6502        |
| 7.9412   | 1350     | 1.6581        |
| 8.0      | 1360     | -             |
| 8.2353   | 1400     | 1.5964        |
| 8.5294   | 1450     | 1.5672        |
| 8.8235   | 1500     | 1.6265        |
| 9.0      | 1530     | -             |
| 9.1176   | 1550     | 1.538         |
| 9.4118   | 1600     | 1.4796        |
| 9.7059   | 1650     | 1.5746        |
| 10.0     | 1700     | 1.5104        |
| 10.2941  | 1750     | 1.4728        |
| 10.5882  | 1800     | 1.4376        |
| 10.8824  | 1850     | 1.5122        |
| 11.0     | 1870     | -             |
| 11.1765  | 1900     | 1.4209        |
| 11.4706  | 1950     | 1.4511        |
| 11.7647  | 2000     | 1.4123        |
| 12.0     | 2040     | -             |
| 12.0588  | 2050     | 1.5277        |
| 12.3529  | 2100     | 1.3963        |
| 12.6471  | 2150     | 1.4161        |
| 12.9412  | 2200     | 1.4373        |
| 13.0     | 2210     | -             |
| 13.2353  | 2250     | 1.3954        |
| 13.5294  | 2300     | 1.4321        |
| 13.8235  | 2350     | 1.3792        |
| 14.0     | 2380     | -             |
| 14.1176  | 2400     | 1.3839        |
| 14.4118  | 2450     | 1.3955        |
| 14.7059  | 2500     | 1.3353        |
| 15.0     | 2550     | 1.3609        |
| 15.2941  | 2600     | 1.3501        |
| 15.5882  | 2650     | 1.3858        |
| 15.8824  | 2700     | 1.3344        |
| 16.0     | 2720     | -             |
| 16.1765  | 2750     | 1.372         |
| 16.4706  | 2800     | 1.3384        |
| 16.7647  | 2850     | 1.2786        |
| 17.0     | 2890     | -             |
| 17.0588  | 2900     | 1.3122        |
| 17.3529  | 2950     | 1.2673        |
| 17.6471  | 3000     | 1.3234        |
| 17.9412  | 3050     | 1.3598        |
| 18.0     | 3060     | -             |
| 18.2353  | 3100     | 1.3129        |
| 18.5294  | 3150     | 1.3045        |
| 18.8235  | 3200     | 1.2582        |
| 19.0     | 3230     | -             |
| 19.1176  | 3250     | 1.2768        |
| 19.4118  | 3300     | 1.2519        |
| 19.7059  | 3350     | 1.2944        |
| 20.0     | 3400     | 1.298         |
| 20.2941  | 3450     | 1.2654        |
| 20.5882  | 3500     | 1.2875        |
| 20.8824  | 3550     | 1.2123        |
| 21.0     | 3570     | -             |
| 21.1765  | 3600     | 1.2933        |
| 21.4706  | 3650     | 1.2585        |
| 21.7647  | 3700     | 1.2717        |
| 22.0     | 3740     | -             |
| 22.0588  | 3750     | 1.2712        |
| 22.3529  | 3800     | 1.215         |
| 22.6471  | 3850     | 1.2223        |
| 22.9412  | 3900     | 1.2683        |
| 23.0     | 3910     | -             |
| 23.2353  | 3950     | 1.194         |
| 23.5294  | 4000     | 1.2168        |
| 23.8235  | 4050     | 1.2353        |
| 24.0     | 4080     | -             |
| 24.1176  | 4100     | 1.2475        |
| 24.4118  | 4150     | 1.1421        |
| 24.7059  | 4200     | 1.2397        |
| 25.0     | 4250     | 1.2076        |
| 25.2941  | 4300     | 1.1889        |
| 25.5882  | 4350     | 1.1889        |
| 25.8824  | 4400     | 1.2577        |
| 26.0     | 4420     | -             |
| 26.1765  | 4450     | 1.2357        |
| 26.4706  | 4500     | 1.179         |
| 26.7647  | 4550     | 1.208         |
| 27.0     | 4590     | -             |
| 27.0588  | 4600     | 1.2482        |
| 27.3529  | 4650     | 1.232         |
| 27.6471  | 4700     | 1.1592        |
| 27.9412  | 4750     | 1.2439        |
| **28.0** | **4760** | **-**         |
| 28.2353  | 4800     | 1.1787        |
| 28.5294  | 4850     | 1.2107        |
| 28.8235  | 4900     | 1.2052        |
| 29.0     | 4930     | -             |
| 29.1176  | 4950     | 1.1634        |
| 29.4118  | 5000     | 1.1619        |
| 29.7059  | 5050     | 1.2188        |
| 30.0     | 5100     | 1.2166        |
| 30.2941  | 5150     | 1.1605        |
| 30.5882  | 5200     | 1.1925        |
| 30.8824  | 5250     | 1.1628        |
| 31.0     | 5270     | -             |
| 31.1765  | 5300     | 1.2036        |
| 31.4706  | 5350     | 1.184         |
| 31.7647  | 5400     | 1.1504        |
| 32.0     | 5440     | -             |
| 32.0588  | 5450     | 1.1495        |
| 32.3529  | 5500     | 1.1397        |
| 32.6471  | 5550     | 1.1891        |
| 32.9412  | 5600     | 1.2152        |
| 33.0     | 5610     | -             |
| 33.2353  | 5650     | 1.1835        |
| 33.5294  | 5700     | 1.1684        |
| 33.8235  | 5750     | 1.1759        |
| 34.0     | 5780     | -             |
| 34.1176  | 5800     | 1.2087        |
| 34.4118  | 5850     | 1.1634        |
| 34.7059  | 5900     | 1.1247        |
| 35.0     | 5950     | 1.1828        |
| 35.2941  | 6000     | 1.1366        |
| 35.5882  | 6050     | 1.1365        |
| 35.8824  | 6100     | 1.1672        |
| 36.0     | 6120     | -             |
| 36.1765  | 6150     | 1.178         |
| 36.4706  | 6200     | 1.1536        |
| 36.7647  | 6250     | 1.1738        |
| 37.0     | 6290     | -             |
| 37.0588  | 6300     | 1.1614        |
| 37.3529  | 6350     | 1.1303        |
| 37.6471  | 6400     | 1.1604        |
| 37.9412  | 6450     | 1.1592        |
| 38.0     | 6460     | -             |
| 38.2353  | 6500     | 1.1525        |
| 38.5294  | 6550     | 1.1311        |
| 38.8235  | 6600     | 1.1754        |
| 39.0     | 6630     | -             |
| 39.1176  | 6650     | 1.1533        |
| 39.4118  | 6700     | 1.1844        |
| 39.7059  | 6750     | 1.1194        |
| 40.0     | 6800     | 1.1695        |

* The bold row denotes the saved checkpoint.
</details>

### Framework Versions
- Python: 3.12.11
- Sentence Transformers: 5.2.0
- Transformers: 4.57.1
- PyTorch: 2.9.0+cu128
- Accelerate: 1.12.0
- Datasets: 4.4.2
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