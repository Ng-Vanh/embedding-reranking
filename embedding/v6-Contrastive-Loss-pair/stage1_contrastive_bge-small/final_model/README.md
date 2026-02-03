---
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- dense
- generated_from_trainer
- dataset_size:43562
- loss:BatchContrastiveLoss
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
# tensor([[1.0000, 0.9999, 0.9883],
#         [0.9999, 1.0000, 0.9882],
#         [0.9883, 0.9882, 1.0000]])
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
* Loss: <code>contrastive_loss.BatchContrastiveLoss</code> with these parameters:
  ```json
  {
      "margin": 1.0,
      "distance_metric": "euclidean",
      "batch_mode": true
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
| 0.2941   | 50       | 7.1429        |
| 0.5882   | 100      | 3.6587        |
| 0.8824   | 150      | 0.9827        |
| 1.0      | 170      | -             |
| 1.1765   | 200      | 0.6122        |
| 1.4706   | 250      | 0.4976        |
| 1.7647   | 300      | 0.4538        |
| 2.0      | 340      | -             |
| 2.0588   | 350      | 0.4302        |
| 2.3529   | 400      | 0.4127        |
| 2.6471   | 450      | 0.3924        |
| 2.9412   | 500      | 0.3847        |
| 3.0      | 510      | -             |
| 3.2353   | 550      | 0.378         |
| 3.5294   | 600      | 0.3671        |
| 3.8235   | 650      | 0.3536        |
| 4.0      | 680      | -             |
| 4.1176   | 700      | 0.3513        |
| 4.4118   | 750      | 0.3431        |
| 4.7059   | 800      | 0.3392        |
| 5.0      | 850      | 0.3329        |
| 5.2941   | 900      | 0.3245        |
| 5.5882   | 950      | 0.325         |
| 5.8824   | 1000     | 0.3154        |
| 6.0      | 1020     | -             |
| 6.1765   | 1050     | 0.321         |
| 6.4706   | 1100     | 0.3052        |
| 6.7647   | 1150     | 0.3158        |
| 7.0      | 1190     | -             |
| 7.0588   | 1200     | 0.3092        |
| 7.3529   | 1250     | 0.3019        |
| 7.6471   | 1300     | 0.3038        |
| 7.9412   | 1350     | 0.3033        |
| 8.0      | 1360     | -             |
| 8.2353   | 1400     | 0.2961        |
| 8.5294   | 1450     | 0.2944        |
| 8.8235   | 1500     | 0.2995        |
| 9.0      | 1530     | -             |
| 9.1176   | 1550     | 0.2896        |
| 9.4118   | 1600     | 0.2841        |
| 9.7059   | 1650     | 0.2913        |
| 10.0     | 1700     | 0.2862        |
| 10.2941  | 1750     | 0.282         |
| 10.5882  | 1800     | 0.276         |
| 10.8824  | 1850     | 0.2856        |
| 11.0     | 1870     | -             |
| 11.1765  | 1900     | 0.2733        |
| 11.4706  | 1950     | 0.2773        |
| 11.7647  | 2000     | 0.272         |
| 12.0     | 2040     | -             |
| 12.0588  | 2050     | 0.2842        |
| 12.3529  | 2100     | 0.27          |
| 12.6471  | 2150     | 0.2698        |
| 12.9412  | 2200     | 0.2712        |
| 13.0     | 2210     | -             |
| 13.2353  | 2250     | 0.2689        |
| 13.5294  | 2300     | 0.2683        |
| 13.8235  | 2350     | 0.2664        |
| 14.0     | 2380     | -             |
| 14.1176  | 2400     | 0.2654        |
| 14.4118  | 2450     | 0.2644        |
| 14.7059  | 2500     | 0.2599        |
| 15.0     | 2550     | 0.261         |
| 15.2941  | 2600     | 0.2586        |
| 15.5882  | 2650     | 0.2637        |
| 15.8824  | 2700     | 0.258         |
| 16.0     | 2720     | -             |
| 16.1765  | 2750     | 0.2624        |
| 16.4706  | 2800     | 0.2581        |
| 16.7647  | 2850     | 0.2508        |
| 17.0     | 2890     | -             |
| 17.0588  | 2900     | 0.2541        |
| 17.3529  | 2950     | 0.2474        |
| 17.6471  | 3000     | 0.254         |
| 17.9412  | 3050     | 0.2574        |
| 18.0     | 3060     | -             |
| 18.2353  | 3100     | 0.2542        |
| 18.5294  | 3150     | 0.2519        |
| 18.8235  | 3200     | 0.2471        |
| **19.0** | **3230** | **-**         |
| 19.1176  | 3250     | 0.2481        |
| 19.4118  | 3300     | 0.2456        |
| 19.7059  | 3350     | 0.2496        |
| 20.0     | 3400     | 0.2484        |
| 20.2941  | 3450     | 0.2464        |
| 20.5882  | 3500     | 0.2459        |
| 20.8824  | 3550     | 0.2406        |
| 21.0     | 3570     | -             |
| 21.1765  | 3600     | 0.2473        |
| 21.4706  | 3650     | 0.2415        |
| 21.7647  | 3700     | 0.2439        |
| 22.0     | 3740     | -             |
| 22.0588  | 3750     | 0.2442        |
| 22.3529  | 3800     | 0.2371        |
| 22.6471  | 3850     | 0.2388        |
| 22.9412  | 3900     | 0.2433        |
| 23.0     | 3910     | -             |
| 23.2353  | 3950     | 0.2339        |
| 23.5294  | 4000     | 0.2384        |
| 23.8235  | 4050     | 0.2393        |
| 24.0     | 4080     | -             |
| 24.1176  | 4100     | 0.242         |
| 24.4118  | 4150     | 0.2291        |
| 24.7059  | 4200     | 0.239         |
| 25.0     | 4250     | 0.2346        |
| 25.2941  | 4300     | 0.2316        |
| 25.5882  | 4350     | 0.2326        |
| 25.8824  | 4400     | 0.2379        |
| 26.0     | 4420     | -             |
| 26.1765  | 4450     | 0.2394        |
| 26.4706  | 4500     | 0.2305        |
| 26.7647  | 4550     | 0.2336        |
| 27.0     | 4590     | -             |
| 27.0588  | 4600     | 0.2361        |
| 27.3529  | 4650     | 0.2371        |
| 27.6471  | 4700     | 0.2296        |
| 27.9412  | 4750     | 0.2364        |
| 28.0     | 4760     | -             |
| 28.2353  | 4800     | 0.2293        |
| 28.5294  | 4850     | 0.2322        |
| 28.8235  | 4900     | 0.2329        |
| 29.0     | 4930     | -             |
| 29.1176  | 4950     | 0.231         |
| 29.4118  | 5000     | 0.2273        |
| 29.7059  | 5050     | 0.2326        |
| 30.0     | 5100     | 0.2313        |
| 30.2941  | 5150     | 0.2298        |
| 30.5882  | 5200     | 0.2277        |
| 30.8824  | 5250     | 0.2263        |
| 31.0     | 5270     | -             |
| 31.1765  | 5300     | 0.2308        |
| 31.4706  | 5350     | 0.2286        |
| 31.7647  | 5400     | 0.2266        |
| 32.0     | 5440     | -             |
| 32.0588  | 5450     | 0.2239        |
| 32.3529  | 5500     | 0.2245        |
| 32.6471  | 5550     | 0.2297        |
| 32.9412  | 5600     | 0.2313        |
| 33.0     | 5610     | -             |
| 33.2353  | 5650     | 0.2288        |
| 33.5294  | 5700     | 0.2236        |
| 33.8235  | 5750     | 0.2272        |
| 34.0     | 5780     | -             |
| 34.1176  | 5800     | 0.2289        |
| 34.4118  | 5850     | 0.2227        |
| 34.7059  | 5900     | 0.223         |
| 35.0     | 5950     | 0.2259        |
| 35.2941  | 6000     | 0.2243        |
| 35.5882  | 6050     | 0.223         |
| 35.8824  | 6100     | 0.2246        |
| 36.0     | 6120     | -             |
| 36.1765  | 6150     | 0.2264        |
| 36.4706  | 6200     | 0.2251        |
| 36.7647  | 6250     | 0.2259        |
| 37.0     | 6290     | -             |
| 37.0588  | 6300     | 0.225         |
| 37.3529  | 6350     | 0.2224        |
| 37.6471  | 6400     | 0.2257        |
| 37.9412  | 6450     | 0.2255        |
| 38.0     | 6460     | -             |
| 38.2353  | 6500     | 0.222         |
| 38.5294  | 6550     | 0.2202        |
| 38.8235  | 6600     | 0.228         |
| 39.0     | 6630     | -             |
| 39.1176  | 6650     | 0.2253        |
| 39.4118  | 6700     | 0.2274        |
| 39.7059  | 6750     | 0.2192        |
| 40.0     | 6800     | 0.2253        |

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