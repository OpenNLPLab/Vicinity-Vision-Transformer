'''turn the model pth file to weight only, no optimizer states'''
import torch
import argparse

parser = argparse.ArgumentParser(description='Hyperparams')
parser.add_argument('filename', nargs='?', type=str, default=None)

args = parser.parse_args()

pth = torch.load(args.filename, map_location=torch.device('cpu'))
checkpoint = pth['model']
torch.save(checkpoint, args.filename.replace(".pth", "_.pth"))
print("finished")
