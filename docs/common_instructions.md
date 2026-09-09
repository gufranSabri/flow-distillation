I will slowly build out this project

The rules i want you to follow are that at every phase of the project, I will try a different approach.
For each approach, I want a model file and trainer file and config file.
model and trainer file go inside src folder

The model class should be such that it saves as a huggingface model.
So when i want to benchmark using lm-evaluation-harness, I can just load the model using AutoModelForCausalLM.from_pretrained() and run the evaluation.
