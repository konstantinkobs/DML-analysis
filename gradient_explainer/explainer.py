from utils import convert_to_valid_path
from matplotlib import pyplot as plt
from tqdm import tqdm
import numpy as np
import torch
from PIL import Image

class Explainer:
	def __init__(self, model, distance, dataset, device="cpu", std_dev_spread=0.15, num_samples=25):
		self.model = model
		self.distance = distance
		self.dataset = dataset
		self.device = device

		assert hasattr(self.dataset, "transform"), "Dataset has no transform attribute"
		self.base_embedding = self.generate_base_embedding()

		# SmoothGrad parameters
		self.std_dev_spread = std_dev_spread
		self.num_samples = num_samples
		
	def generate_base_embedding(self):
		# Creates a black image which is transformed/normalized
		base_img = self.dataset.transform(Image.fromarray(np.zeros((500, 500, 3)), mode="RGB")).unsqueeze(0).to(self.device)
		base_embedding = self.model(base_img).detach()

		return base_embedding

	def calc_grads(self, idx):
		item = self.dataset[idx]
		data = item["data"].to(self.device)
		
		img = item["data"].detach().numpy()

		# add fake batch dimension
		data.unsqueeze_(0)
		data = data.repeat(self.num_samples, 1, 1, 1)
		data += torch.randn_like(data) * self.std_dev_spread * (data.max() - data.min())
		_ = data.requires_grad_()

		out = self.model(data)
		loss = self.distance(out, self.base_embedding).mean()
		loss.backward()

		grads = data.grad.mean(dim=1).detach().cpu()

		return img, grads

	@staticmethod
	def process_grads(grads, percentile=99):
		grads = grads.abs().sum(dim=0).detach().numpy()
		v_max = np.percentile(grads, percentile)
		v_min = np.min(grads)
		grads = np.clip((grads - v_min) / (v_max - v_min), 0, 1)

		return grads

	@staticmethod
	def visualize_grads(grads_processed, title="", save_path=None):
		plt.imshow(grads_processed, cmap="gray")
		plt.axis('off')

		plt.title(title)

		if save_path:
			plt.savefig(save_path, bbox_inches="tight")

		plt.show()

	@staticmethod
	def visualize_img(img, title="", save_path=None):
		img = np.moveaxis(img, 0, -1)
		img = img[:, :, ::-1]
		img = (img - img.min()) / (img.max() - img.min())

		plt.imshow(img)
		plt.axis('off')

		plt.title(title)

		if save_path:
			plt.savefig(save_path, bbox_inches="tight")

		plt.show()



if __name__ == "__main__":
	import sys
	sys.path.append("..")
	from utils import loss_distances, get_dataset

	import argparse
	import os
	import pickle
	
	from reality_check.dataset import transform
	from reality_check.model import Network


	parser = argparse.ArgumentParser()
	parser.add_argument("--loss", type=str, help="Loss name")
	parser.add_argument("--dataset", type=str, help="Dataset name")
	parser.add_argument("--fold", type=int, help="Fold number")
	config = vars(parser.parse_args())

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(f"Using {device}")

	results_path = f"results/{convert_to_valid_path(config['loss'])}/{convert_to_valid_path(config['dataset'])}/"
	os.makedirs(results_path, exist_ok=True)

	dataset = get_dataset(config["dataset"], transform, mode="test")
	distance = loss_distances[config["loss"]]
	model = Network(config["loss"], dataset=config["dataset"], fold=config["fold"]).eval().to(device)
	explainer = Explainer(model, distance, dataset, std_dev_spread=0.15, num_samples=50, device=device)

	grads_list = []
	for i in tqdm(range(len(dataset))):
		img, grads = explainer.calc_grads(i)
		grads_processed = Explainer.process_grads(grads)
		grads_list.append(grads_processed)
		# Explainer.visualize_grads(img, grads_processed)

	with open(results_path + f"{config['fold']}.p", "wb") as f:
		pickle.dump(grads_list, f)
