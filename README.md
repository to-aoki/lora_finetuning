# LoRA instruct script for Japanese Language models.

| models                   | tested platform                              |
|--------------------------|----------------------------------------------|
| rinna gptneox            | jetson agx orin                              |
| cyberagent opencalm      | jetson agx orin                              |
| meta llama2              | jetson agx orin (70B not work need more VRAM) |
| line-corporation gptneox | jetson agx orin                              |
| stabilityai stablelm     | jetson agx orin                              |
| matsuo-lab gptneox       | jetson agx orin                              |

**example**
```bash
nohup python3 lora_finetune.py --model_name="HUGGINGFACE_MODEL_NAME" > logging.out &
```

