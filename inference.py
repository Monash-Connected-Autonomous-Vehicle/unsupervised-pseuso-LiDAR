from os import sep
from trainer import *
from models.depth.disp_net import DispNetS

from utils.transforms import UnNormalize
from geometry.pose_geometry import *

# load checkpoint
save_path = './pretrained/generic_sfm.pth'
checkpoint = torch.load(save_path)
depth_model_state_dict = checkpoint['dpth_mdl_state_dict']
pose_model_state_dict  = checkpoint['pose_mdl_state_dict']

# init dataset
with open('configs/test_config.yaml') as file:
    config = yaml.full_load(file)

trainer    = Trainer(config)
dataloader = trainer.train_loader

one_sample = trainer.dataset.__getitem__(0)


# load a depth model
depth_model = trainer.depth_model
# depth_model.load_state_dict(depth_model_state_dict)
depth_model.eval()

# load a pose model
pose_model = trainer.pose_model
# pose_model.load_state_dict(pose_model_state_dict)
pose_model.eval()

# test input image
input_imgs = one_sample['tgt'].unsqueeze(0).to(trainer.device)
ref_imgs   = [img.unsqueeze(0).to(trainer.device) for img in one_sample['ref_imgs']]

with torch.no_grad():
    # test and plot
    depth = depth_model(input_imgs)
    pose  = pose_model(input_imgs, ref_imgs)

# split poses
poses_t_0 = pose[:, 0, :]

# figure out all outputs
img        = (1/depth[0][0].squeeze()).cpu().detach().numpy()

plt.imshow(img)
plt.show()

