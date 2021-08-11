from trainer import *

with open('configs/mac_test.yaml.yaml') as file:
    config = yaml.full_load(file)

trainer = Trainer(config)
trainer.train()