from trainer import *


# load models
save_path = './models/pretrained/generic_sfm.pth'
checkpoint = torch.load(save_path)

# init dataset
with open('configs/basic_config.yaml') as file:
    config = yaml.full_load(file)

trainer    = Trainer(config)
dataloader = trainer.train_loader
data       = next(dataloader)

print(data)





