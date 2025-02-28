import os
import argparse
import paddle
import paddle.distributed as dist
from paddle.io import Dataset
import paddle.vision.transforms as T
from PIL import Image
from ppcls.arch.backbone.legendary_models.swin_transformer import SwinTransformer_large_patch4_window7_224
from ppclsxcit.arch.backbone.model_zoo.xcit import XCiT_large_12_p16
import torch
import numpy as np
from adv.autoattack_ddp_paddle import AutoAttack
import random

MEAN_ZERO = (0.0, 0.0, 0.0)
STA_ONE = (1.0, 1.0, 1.0)
MEAN_IMAGENET = [0.485, 0.456, 0.406]
STD_IMAGENET = [0.229, 0.224, 0.225]

_ = paddle.seed(0)
np.random.seed(0)
random.seed(0)

def get_args_parser():
    parser = argparse.ArgumentParser('Robust training script', add_help=False)


    # Model parameters
    parser.add_argument('--model_name', default='', type=str, metavar='MODEL', help='Name of model to train')
    parser.add_argument('--num_classes', default=1000, type=int, help='number of classes')

    # data parameters
    parser.add_argument('--input_size', default=224, type=int, help='images input size')
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--crop_pct', default=0.875, type=float, metavar='N', help='Input image center crop percent (for validation only)')
    parser.add_argument('--interpolation', default="bicubic")
    parser.add_argument('--imagenet_val_path', default='', type=str, help='path to imagenet validation dataset')

    # attack info
    parser.add_argument('--attack_types', type=str, nargs='*', default=('autoattack',), help='autoattack, pgd')
    parser.add_argument('--norm', type=str, default='Linf', help='You can choose norm for aa attack', choices=['Linf', 'L2', 'L1'])
    
    return parser


class ImageNet(Dataset):
    def __init__(self, root, meta_file='./src_data/imagenet_1k.txt', transform=None):
        super(ImageNet, self).__init__()
        self.data_dir = root
        self.meta_file = meta_file
        self.transform = transform
        self._indices = []

        for line in open(meta_file, encoding="utf-8"):
            temp_names=line.strip().split(' ')
            img_path, label = temp_names[0], temp_names[1]
            self._indices.append((os.path.join(self.data_dir, img_path), label))


    def __getitem__(self, index):
        img_path, label = self._indices[index]
        img = Image.open(img_path).convert('RGB')
        label = int(label)
        if self.transform is not None:
            img = self.transform(img)
        label=paddle.to_tensor([label], dtype='int64')
        return img, label

    def __len__(self):
        return len(self._indices)

class NormalizeByChannelMeanStd(paddle.nn.Layer):
    def __init__(self, mean, std):
        super().__init__()
        mean = paddle.unsqueeze(paddle.to_tensor(mean), axis=[0,2,3])
        std = paddle.unsqueeze(paddle.to_tensor(std), axis=[0,2,3])
        
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, inputs):
        return (inputs-self.mean)/self.std


args = get_args_parser().parse_args()

size = int(args.input_size/args.crop_pct)
transform=T.Compose([T.Resize(size, interpolation=args.interpolation), T.CenterCrop(args.input_size), T.ToTensor()])
eval_dataset=ImageNet(root=args.imagenet_val_path, meta_file='./src_data/imagenet_1k.txt', transform=transform)

dist.init_parallel_env()

dataloader = paddle.io.DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False)

if args.model_name=='swinl_at':
    model=SwinTransformer_large_patch4_window7_224(pretrained=False)
    torch_state_dict=torch.load('./src_ckpt/swinl_at_checkpoint.pth', map_location='cpu')
    paddle_state_dict = {}
    for torch_key in torch_state_dict:
        paddle_key = torch_key

        if 'weight' in paddle_key and (not 'patch_embed.proj.weight'==paddle_key):
            paddle_state_dict[paddle_key] = paddle.to_tensor(torch_state_dict[torch_key].cpu().numpy().transpose())
        else:
            paddle_state_dict[paddle_key] = paddle.to_tensor(torch_state_dict[torch_key].cpu().numpy())
    mean = MEAN_IMAGENET
    std = STD_IMAGENET
else:
    model=XCiT_large_12_p16(pretrained=False)
    torch_state_dict=torch.load('./src_ckpt/xcitl_sota_checkpoint.pth', map_location='cpu')
    paddle_state_dict = {}
    for torch_key in torch_state_dict:
        paddle_key = torch_key

        if 'running_mean' in paddle_key:
            new_paddle_key=paddle_key.replace('running_mean', '_mean')
            paddle_state_dict[new_paddle_key] = paddle.to_tensor(torch_state_dict[torch_key].cpu().numpy())
        elif 'running_var' in paddle_key:
            new_paddle_key=paddle_key.replace('running_var', '_variance')
            paddle_state_dict[new_paddle_key] = paddle.to_tensor(torch_state_dict[torch_key].cpu().numpy())
        elif ('cls_attn_blocks.0.attn' in paddle_key or 'cls_attn_blocks.1.attn' in paddle_key) and (not 'proj' in paddle_key):
            continue
        elif 'pos_embed.token_projection' in paddle_key:
            new_paddle_key=paddle_key.replace('pos_embed', 'pos_embeder')
            paddle_state_dict[new_paddle_key] = paddle.to_tensor(torch_state_dict[torch_key].cpu().numpy())
        elif 'weight' in paddle_key and (not 'patch_embed' in paddle_key) and (not 'conv' in paddle_key):
            paddle_state_dict[paddle_key] = paddle.to_tensor(torch_state_dict[torch_key].cpu().numpy().transpose())
        else:
            paddle_state_dict[paddle_key] = paddle.to_tensor(torch_state_dict[torch_key].cpu().numpy())
    
    # cls_attn_blocks qkv
    for id in ['0', '1']:
        for wb in ['weight', 'bias']:
            temp_list=[]
            for qkv in ['q', 'k', 'v']:
                temp_list.append(torch_state_dict[f'cls_attn_blocks.{id}.attn.{qkv}.{wb}'].cpu().numpy())
            temp_np=np.concatenate(temp_list, axis=0).transpose()
            paddle_state_dict[f'cls_attn_blocks.{id}.attn.qkv.{wb}']=paddle.to_tensor(temp_np)
    
    mean = MEAN_ZERO
    std = STA_ONE


model.set_state_dict(paddle_state_dict)
normalize = NormalizeByChannelMeanStd(mean = mean, std = std)
model = paddle.nn.Sequential(normalize, model)
model = paddle.DataParallel(model)
model.eval()


# clean acc
total_acc = 0
total_num = 0
for batch_id, data in enumerate(dataloader()):
    x_data = data[0]
    y_data = data[1]

    batch_size=x_data.shape[0]

    predicts = model(x_data)

    acc = paddle.metric.accuracy(predicts, y_data).numpy()

    total_acc += acc * batch_size
    total_num += batch_size

    print("batch_id: {}, acc is: {}".format(batch_id, acc))

print("Total acc on clean examples is: {}".format(total_acc / total_num))

# adv acc
total_acc = 0
total_num = 0
eps=4/255
for batch_id, data in enumerate(dataloader()):
    x_data = data[0]
    y_data = data[1]

    batch_size=x_data.shape[0]

    attacker = AutoAttack(model, norm=args.norm, eps=eps, version='standard')
    x_adv_data=attacker.run_standard_evaluation(x_data, y_data, bs=batch_size)
    predicts = model(x_adv_data)

    acc = paddle.metric.accuracy(predicts, y_data).numpy()

    total_acc += acc * batch_size
    total_num += batch_size

    print("batch_id: {}, acc is: {}".format(batch_id, acc))

print("Total acc on adversarial examples is: {}".format(total_acc / total_num))
