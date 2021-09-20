from trainer import *

with open('configs/test_config.yaml') as file:
    config = yaml.full_load(file)

trainer = Trainer(config)
trainer.train()