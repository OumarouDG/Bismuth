import torch
from torchvision.models.segmentation import deeplabv3_resnet101

model = deeplabv3_resnet101(pretrained=True)   #ok so to my understandign this only needs to be run ONCE and it gets it into onnx
model.eval()         
dummy_input = torch.randn(1, 3, 512, 512)  #yk what we only need 512 by 512, this a mvp done in like oh god 5 hours are left
torch.onnx.export(model, dummy_input, "deeplabv3.onnx", input_names=["input"], output_names=["output"])
#AYYYY IT WORKED ON THE FIRST TRY, note to self do not run this again