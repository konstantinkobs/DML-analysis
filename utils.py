import sys
sys.path.append("..")
from pytorch_metric_learning import distances
import os

all_losses = ["Contrastive", "Triplet", "NTXent", "ProxyNCA", "Margin", "Margin / class", "Normalized Softmax", "CosFace", "ArcFace", "FastAP", "SNR Contrastive", "Multi Similarity", "Multi Similarity + Miner", "SoftTriple", "None"]
all_datasets = ["Cars196", "CUB200", "SOP"]
all_folds = [0, 1, 2, 3]

all_attributes = {
	"model": "Car model",
	"rotation_z": "Car rotation",
	"color_hue": "Car color hue",
	"color_sat": "Car color saturation",
	"color_val": "Car color value",
	"bg_color_hue": "Background color hue",
	"bg_color_sat": "Background color saturation",
	"bg_color_val": "Background color value",
	"camera_loc_z": "Camera height",
	"sun_elevation": "Sun elevation",
	"sun_rotation": "Sun rotation"
}

loss_metadata = {
	"Contrastive": {
		"year": 2006,
		"distance": "Euclidean power 1",
		"type": "Ranking"
	},
	"Triplet": {
		"year": 2006,
		"distance": "Euclidean power 1",
		"type": "Ranking"
	},
	"NTXent": {
		"year": 2016,
		"distance": "Cosine Similarity",
		"type": "Ranking"
	},
	"ProxyNCA": {
		"year": 2017,
		"distance": "Euclidean power 2",
		"type": "Classification"
	},
	"Margin": {
		"year": 2017,
		"distance": "Euclidean power 1",
		"type": "Ranking"
	},
	"Margin / class": {
		"year": 2017,
		"distance": "Euclidean power 1",
		"type": "Ranking"
	},
	"Normalized Softmax": {
		"year": 2017,
		"distance": "Dot Product Similarity",
		"type": "Classification"
	},
	"CosFace": {
		"year": 2018,
		"distance": "Cosine Similarity",
		"type": "Classification"
	},
	"ArcFace": {
		"year": 2019,
		"distance": "Cosine Similarity",
		"type": "Classification"
	},
	"FastAP": {
		"year": 2019,
		"distance": "Euclidean power 2",
		"type": "Ranking"
	},
	"SNR Contrastive": {
		"year": 2019,
		"distance": "SNR Distance",
		"type": "Ranking"
	},
	"Multi Similarity": {
		"year": 2019,
		"distance": "Cosine Similarity",
		"type": "Ranking"
	},
	"Multi Similarity + Miner": {
		"year": 2019,
		"distance": "Cosine Similarity",
		"type": "Ranking"
	},
	"SoftTriple": {
		"year": 2019,
		"distance": "Cosine Similarity",
		"type": "Classification"
	},
	"None": {
		"year": 0,
		"distance": "None",
		"type": "None"
	}
}

loss_distances = {
	# Euclidean
	"None": distances.LpDistance(normalize_embeddings=True, p=2, power=1),
	"Contrastive": distances.LpDistance(normalize_embeddings=True, p=2, power=1),
	"Triplet": distances.LpDistance(normalize_embeddings=True, p=2, power=1),
	"Margin": distances.LpDistance(normalize_embeddings=True, p=2, power=1),
	"Margin / class": distances.LpDistance(normalize_embeddings=True, p=2, power=1),
	# Euclidean with power of 2
	"FastAP": distances.LpDistance(normalize_embeddings=True, p=2, power=2),
	"ProxyNCA": distances.LpDistance(normalize_embeddings=True, p=2, power=2),
	# Cosine Similarity
	"CosFace": distances.CosineSimilarity(),
	"Multi Similarity": distances.CosineSimilarity(),
	"Multi Similarity + Miner": distances.CosineSimilarity(),
	"NTXent": distances.CosineSimilarity(),
	"SoftTriple": distances.CosineSimilarity(),
	"ArcFace": distances.CosineSimilarity(),
	# Dot Product Similarity
	"Normalized Softmax": distances.DotProductSimilarity(),
	# SNR Distance
	"SNR Contrastive": distances.SNRDistance(),
}

def get_dataset(dataset, transform, mode="train"):
	if dataset == "Cars196":
		from datasets.cars196 import Cars196
		return Cars196("../data", mode=mode, transform=transform, download=True)
	elif dataset == "CUB200":
		from datasets.cub200 import CUB200
		return CUB200("../data", mode=mode, transform=transform, download=True)
	elif dataset == "SOP":
		from datasets.stanford_online_products import StanfordOnlineProducts
		return StanfordOnlineProducts("../data", mode=mode, transform=transform, download=True)
	else:
		raise Exception("No valid dataset name")

def get_model_path(loss, dataset, fold):
	paths = {
		"Contrastive": {
			"Cars196": f"models/cars_contrastive46_reproduction0/Test50_50_Partitions4_{fold}/saved_models",
			"CUB200": f"models/cub_contrastive41_reproduction0/Test50_50_Partitions4_{fold}/saved_models",
			"SOP": f"models/sop_contrastive21_reproduction0/Test50_50_Partitions4_{fold}/saved_models",
		},
		"Triplet": {
			"Cars196": f"models/cars_triplet46_reproduction0/Test50_50_Partitions4_{fold}/saved_models",
			"CUB200": f"models/cub_triplet37_reproduction0/Test50_50_Partitions4_{fold}/saved_models",
			"SOP": f"models/sop_triplet34_reproduction0/Test50_50_Partitions4_{fold}/saved_models",
		},
		"NTXent": {
			"Cars196": f"models/cars_ntxent49_reproduction0/Test50_50_Partitions4_{fold}/saved_models",
			"CUB200": f"models/cub_ntxent28_reproduction0/Test50_50_Partitions4_{fold}/saved_models",
			"SOP": f"models/sop_ntxent23_reproduction0/Test50_50_Partitions4_{fold}/saved_models",
		},
		"ProxyNCA": {
			"Cars196": f"models/cars_proxy_nca5_reproduction0/Test50_50_Partitions4_{fold}/saved_models",
			"CUB200": f"models/cub_proxy_nca25_reproduction0/Test50_50_Partitions4_{fold}/saved_models",
			"SOP": f"models/sop_proxy_nca23_reproduction0/Test50_50_Partitions4_{fold}/saved_models",
		},
		"Margin": {
			"Cars196": f"models/cars_margin_no_weight_decay_reproduction0/Test50_50_Partitions4_{fold}/saved_models",
			"CUB200": f"models/cub_margin_no_weight_decay15_reproduction0/Test50_50_Partitions4_{fold}/saved_models",
			"SOP": f"models/sop_margin_no_weight_decay30_reproduction0/Test50_50_Partitions4_{fold}/saved_models",
		},
		"Margin / class": {
			"Cars196": f"models/cars_margin_param_per_class_no_weight_decay20_reproduction0/Test50_50_Partitions4_{fold}/saved_models",
			"CUB200": f"models/cub_margin_param_per_class_no_weight_decay24_reproduction0/Test50_50_Partitions4_{fold}/saved_models",
			"SOP": f"models/sop_margin_param_per_class_no_weight_decay35_reproduction0/Test50_50_Partitions4_{fold}/saved_models",
		},
		"Normalized Softmax": {
			"Cars196": f"models/cars_normalized_softmax45_reproduction0/Test50_50_Partitions4_{fold}/saved_models",
			"CUB200": f"models/cub_normalized_softmax15_reproduction0/Test50_50_Partitions4_{fold}/saved_models",
			"SOP": f"models/sop_normalized_softmax47_reproduction0/Test50_50_Partitions4_{fold}/saved_models",
		},
		"CosFace": {
			"Cars196": f"models/cars_cosface15_reproduction0/Test50_50_Partitions4_{fold}/saved_models",
			"CUB200": f"models/cub_cosface20_reproduction0/Test50_50_Partitions4_{fold}/saved_models",
			"SOP": f"models/sop_cosface8_reproduction0/Test50_50_Partitions4_{fold}/saved_models",
		},
		"ArcFace": {
			"Cars196": f"models/cars_arcface49_reproduction0/Test50_50_Partitions4_{fold}/saved_models",
			"CUB200": f"models/cub_arcface36_reproduction0/Test50_50_Partitions4_{fold}/saved_models",
			"SOP": f"models/sop_arcface49_reproduction0/Test50_50_Partitions4_{fold}/saved_models",
		},
		"FastAP": {
			"Cars196": f"models/cars_fastap27_reproduction0/Test50_50_Partitions4_{fold}/saved_models",
			"CUB200": f"models/cub_fastap17_reproduction0/Test50_50_Partitions4_{fold}/saved_models",
			"SOP": f"models/sop_fastap16_reproduction0/Test50_50_Partitions4_{fold}/saved_models",
		},
		"SNR Contrastive": {
			"Cars196": f"models/cars_snr_contrastive33_reproduction0/Test50_50_Partitions4_{fold}/saved_models",
			"CUB200": f"models/cub_snr_contrastive5_reproduction0/Test50_50_Partitions4_{fold}/saved_models",
			"SOP": f"models/sop_snr_contrastive29_reproduction0/Test50_50_Partitions4_{fold}/saved_models",
		},
		"Multi Similarity": {
			"Cars196": f"models/cars_multi_similarity22_reproduction0/Test50_50_Partitions4_{fold}/saved_models",
			"CUB200": f"models/cub_multi_similarity9_reproduction0/Test50_50_Partitions4_{fold}/saved_models",
			"SOP": f"models/sop_multi_similarity6_reproduction0/Test50_50_Partitions4_{fold}/saved_models",
		},
		"Multi Similarity + Miner": {
			"Cars196": f"models/cars_multi_similarity_with_ms_miner38_reproduction0/Test50_50_Partitions4_{fold}/saved_models",
			"CUB200": f"models/cub_multi_similarity_with_ms_miner45_reproduction0/Test50_50_Partitions4_{fold}/saved_models",
			"SOP": f"models/sop_multi_similarity_with_ms_miner36_reproduction0/Test50_50_Partitions4_{fold}/saved_models",
		},
		"SoftTriple": {
			"Cars196": f"models/cars_soft_triple14_reproduction0/Test50_50_Partitions4_{fold}/saved_models",
			"CUB200": f"models/cub_soft_triple46_reproduction0/Test50_50_Partitions4_{fold}/saved_models",
			"SOP": f"models/sop_soft_triple18_reproduction0/Test50_50_Partitions4_{fold}/saved_models",
		},
	}

	assert fold in range(4), "No valid fold index specified"
	assert loss in paths.keys(), "No valid loss function specified"
	assert dataset in paths["Contrastive"].keys(), "No valid dataset name specified"

	current_dir = os.path.dirname(os.path.realpath(__file__))

	base_path = current_dir + "/" + paths[loss][dataset]
	trunk_path = base_path + "/trunk_best.pth"
	embedder_path = base_path + "/embedder_best.pth"

	return trunk_path, embedder_path

def convert_to_valid_path(string, lower=False):
	if lower:
		string = string.lower()
	return "".join(x for x in string if x.isalnum())