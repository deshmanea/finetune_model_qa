# finetune_model_qa
Model - DistilBert, DataSet - SQuAD

As part of training model below setting used,

**Model** - distilbert-base-uncased

**Dataset** - squad (HF)

**cpu** - i5-1035G1 CPU @ 1.00GHz (1.19 GHz)

# Steps :
**Step - 1** - Select base model
Depending on uses purpose we can choose base model in this case https://huggingface.co/distilbert/distilbert-base-uncased is used as a base model since my intention was to train it for QA
Reason for choosing it - As part of ML learning I have challenged myself to train model with old GPU which is not made with ML in mind. Due to which I have ended up with student version of BERT. It's smaller, faster and as it's teacher model is BERT so it's very close to standard like BERT compressed _(with little loss which might not be noticed unless encountered with rarecase)_.

**Step - 2** - Prepare Dataset
With strict hardware restrictions decided to train on 'SQuAD' QA dataset. The data is split into Train (1500 samples) and Validate (500 Samples).

**Step - 3** - Tokenizer
Used same tokenizer which is used by BERT team, to avoid incorrect token generation.

**Step - 4** - Offset understanding
Felt most tricy part of the entrire finetuning if this goes wrong, everything collapse - Lesson Learned : Pay attention to mapping

**Step - 5** - Training arg 
Used 3e-5 learning rate considering the CPU capacity and time. The selected learning rate is generally considered in good learning rate range.

**Step - 6** - Save and Tokenizer
Post finetuning Save the model and along with used tokenizer

**Step - 7** - Check few example for inference
Before evalution quickly cross check for inference with some samples

**Step - 8** - Evaluation and F1  [Exact Match and F1]
Postprocessing predictions...
EM: 57.80%
F1: 64.92%

_**Note - Considering 2K Sample size this is good score and training it further with full dataset will increase EM and F1**_


# Training time
{'train_runtime': 11126.6732, 'train_samples_per_second': 0.554, 'train_steps_per_second': 0.069, 'train_loss': 2.1886439812013303, 'epoch': 3.0}
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 771/771 [3:05:26<00:00, 14.43s/it]

# Training, Evalution and F1 score
<img width="1533" height="900" alt="image" src="https://github.com/user-attachments/assets/6fcc71b5-b35d-4925-9a6e-4be7f30287de" />
