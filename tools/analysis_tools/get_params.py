import torch

file_path = "/data/zhh/zhh/MapTR/ckpts/maptr_tiny_r50_24e.pth"
model = torch.load(file_path, map_location="cpu")
all = 0
for key in list(model["state_dict"].keys()):
    all += model["state_dict"][key].nelement()
print(all)

# smaller 63374123
# v4 69140395

# RoadMapNet 57317631
# maptr 36246771