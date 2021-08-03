from trainer import *


# load checkpoint
save_path = './models/pretrained/generic_sfm.pth'
checkpoint = torch.load(save_path)
depth_model_state_dict = checkpoint['dpth_mdl_state_dict']

# init dataset
with open('configs/basic_config.yaml') as file:
    config = yaml.full_load(file)

trainer    = Trainer(config)
dataloader = trainer.train_loader
data       = next(iter(dataloader))

# load a model
depth_model = trainer.depth_model
depth_model.load_state_dict(depth_model_state_dict)
depth_model.eval()

# test input image
input_imgs = data['tgt'].to(trainer.device)

# test and plot
output = depth_model(input_imgs*255)
img    = output[0][0].squeeze().cpu().detach().numpy()

imga = Image.fromarray(img)
imga.show()







