import os
import glob
import json
import wandb
import torch
import argparse
import torchaudio
import pedalboard
import numpy as np
import pytorch_lightning as pl

from typing import Callable
from torchmetrics import Accuracy

from tqdm import tqdm

from st_ito.utils import (
    load_param_model,
    load_clap_model,
    load_fx_encoder_model,
    get_fx_encoder_embeds,
    load_deepafx_st_model,
    get_deepafx_st_embeds,
    get_param_embeds,
    get_clap_embeds,
    load_wav2vec2_model,
    get_wav2vec2_embeds,
    load_wav2clip_model,
    get_wav2clip_embeds,
    load_vggish_model,
    get_vggish_embeds,
    load_mfcc_feature_extractor,
    get_mfcc_feature_embeds,
    load_mir_feature_extractor,
    get_mir_feature_embeds,
    load_beats_model,
    get_beats_embeds,
)


class StyleDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir: str, mode: str, sample_rate: int = 48000):
        self.root_dir = root_dir
        self.mode = mode
        self.sample_rate = sample_rate

        assert mode in ["train", "val", "test"]  # only these modes are supported

        # find all filepaths for the dataset
        filepaths = glob.glob(
            os.path.join(root_dir, f"{mode}", "**", "*.wav"), recursive=True
        )

        if len(filepaths) == 0:
            raise ValueError(f"No files found in {root_dir}")

        # define label indices
        self.label_to_idx = {
            "bright": 0,
            "broadcast": 1,
            "neutral": 2,
            "telephone": 3,
            "warm": 4,
        }
        self.idx_to_label = {v: k for k, v in self.label_to_idx.items()}

        self.examples = []
        # read audio into memory
        for filepath in tqdm(filepaths):
            x, sr = torchaudio.load(filepath, backend="soundfile")

            if sr != self.sample_rate:
                x = torchaudio.functional.resample(x, sr, self.sample_rate)

            if x.shape[0] == 1:
                x = x.repeat(2, 1)  # make dual mono

            label = os.path.basename(filepath).split("_")[1]
            if label not in self.label_to_idx:
                raise ValueError(f"Unknown label: {label}")
            self.examples.append((x, label))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        audio, label = self.examples[idx]

        return audio, self.label_to_idx[label]


class StyleClassifierLightningModule(pl.LightningModule):
    def __init__(
        self,
        root_dir: str,
        model: torch.nn.Module,
        embed_fn: Callable,
        lr: float = 1e-4,
        max_epochs: int = 100,
        num_classes: int = 5,
        batch_size: int = 32,
        sample_rate: int = 48000,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["embed_fn", "model"])

        # set up model and linear probe
        self.model = model
        self.embed_fn = embed_fn
        self.probe = torch.nn.Linear(model.embed_dim, num_classes)

        # track classification accuracy
        self.accuracy = Accuracy(task="multiclass", num_classes=num_classes)

    def forward(self, x: torch.Tensor):
        bs, chs, seq_len = x.shape

        # extract features with encoder
        with torch.no_grad():  # do not update the encoder
            mid_feats = self.embed_fn(x, self.model, self.hparams.sample_rate)

        # classify
        logits = self.probe(mid_feats)

        return logits

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.probe.parameters(),
            lr=self.hparams.lr,
        )
        ms1 = int(self.hparams.max_epochs * 0.8)
        ms2 = int(self.hparams.max_epochs * 0.95)
        print(
            "Learning rate schedule:",
            f"0 {self.hparams.lr:0.2e} -> ",
            f"{ms1} {self.hparams.lr*0.1:0.2e} -> ",
            f"{ms2} {self.hparams.lr*0.01:0.2e}",
        )
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[ms1, ms2],
            gamma=0.1,
        )
        return [optimizer], [lr_scheduler]

    def common_step(self, batch, batch_idx, mode: str = "train"):
        x, y = batch
        logits = self(x)
        loss = torch.nn.functional.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = self.accuracy(preds, y)
        self.log(f"{mode}_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log(f"{mode}_accuracy", acc, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self.common_step(batch, batch_idx, mode="train")

    def validation_step(self, batch, batch_idx):
        return self.common_step(batch, batch_idx, mode="val")

    def test_step(self, batch, batch_idx):
        return self.common_step(batch, batch_idx, mode="test")

    def train_dataloader(self):
        train_dataset = StyleDataset(root_dir=self.hparams.root_dir, mode="train")
        return torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            drop_last=True,
        )

    def val_dataloader(self):
        val_dataset = StyleDataset(root_dir=self.hparams.root_dir, mode="val")
        return torch.utils.data.DataLoader(
            val_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
        )

    def test_dataloader(self):
        test_dataset = StyleDataset(root_dir=self.hparams.root_dir, mode="test")
        return torch.utils.data.DataLoader(
            test_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
        )


def run_probe_task(
    root_dir: str,
    save_dir: str,
    models: dict,
    max_epochs: int = 100,
    batch_size: int = 32,
    lr: float = 1e-3,
):
    for dataset_size in [5, 10, 100]:
        for model_name, model_dict in models.items():
            checkpoint_callback = pl.callbacks.ModelCheckpoint(
                monitor="val_accuracy",
                mode="max",
                save_top_k=1,
                save_last=True,
            )
            logger = logger = pl.loggers.WandbLogger(
                save_dir=save_dir,
                log_model=False,
                project="lcap-cls",
                name=f"{mode}-{model_name}-{dataset_size}",
            )
            trainer = pl.Trainer(
                max_epochs=max_epochs,
                devices=1,
                logger=logger,
                callbacks=[checkpoint_callback],
                check_val_every_n_epoch=5,
            )
            lightning_module = StyleClassifierLightningModule(
                model=model_dict["model"],
                embed_fn=model_dict["embed_fn"],
                num_classes=5,
                batch_size=batch_size,
                root_dir=os.path.join(root_dir, dataset_path),
                max_epochs=max_epochs,
                lr=lr,
            )
            trainer.fit(lightning_module)
            trainer.test(ckpt_path="best")
            logger.finalize("success")
            wandb.finish()


def run_zero_shot_task(
    root_dir: str,
    dataset_path: str,
    models: dict,
    output_dir: str,
    num_examples: int = 10,
):
    style_names = ["neutral", "warm", "bright", "broadcast", "telephone"]
    style_filepaths = {}
    for style_name in style_names:
        # find all style filepaths
        filepaths = glob.glob(
            os.path.join(root_dir, dataset_path, "test", style_name, "*.wav"),
            recursive=True,
        )
        style_filepaths[style_name] = filepaths

    results = {}
    # create test examples
    for style_name in style_names:
        print("Evaluting style:", style_name)
        for _ in tqdm(range(num_examples)):
            input_filepath = np.random.choice(style_filepaths[style_name])
            input_audio, input_sr = torchaudio.load(input_filepath, backend="soundfile")
            input_audio = input_audio.unsqueeze(0)
            bs, chs, seq_len = input_audio.shape
            if chs == 1:
                input_audio = input_audio.repeat(1, 2, 1)

            input_audio /= input_audio.abs().max()

            # create the reference examples
            ref_examples = []
            for style_name_ref in style_names:
                ref_filepath = np.random.choice(style_filepaths[style_name_ref])
                ref_audio, ref_sr = torchaudio.load(ref_filepath, backend="soundfile")
                ref_audio = ref_audio.unsqueeze(0)
                bs, chs, seq_len = ref_audio.shape
                if chs == 1:
                    ref_audio = ref_audio.repeat(1, 2, 1)
                ref_audio /= ref_audio.abs().max()
                ref_examples.append((style_name_ref, ref_audio))

            # classify with each model
            for model_name, model_dict in models.items():
                model = model_dict["model"]
                embed_fn = model_dict["embed_fn"]

                # extract features
                input_feats = embed_fn(input_audio, model, input_sr)

                similarities = []
                for style_name_ref, ref_audio in ref_examples:
                    ref_feats = embed_fn(ref_audio, model, ref_sr)

                    tmp_similarities = []
                    for ref_feat_name, ref_feat in ref_feats.items():
                        input_feat = input_feats[ref_feat_name]

                        # compute similarity
                        similarity = torch.nn.functional.cosine_similarity(
                            input_feat, ref_feat
                        )
                        tmp_similarities.append(similarity.item())

                    similarities.append(np.mean(tmp_similarities))

                # find the most similar style
                most_similar_idx = np.argmax(similarities)
                most_similar_style = style_names[most_similar_idx]

                if model_name not in results:
                    results[model_name] = {}

                if style_name not in results[model_name]:
                    results[model_name][style_name] = {
                        "avg": 0,
                        "correct": [],
                    }

                correct = most_similar_style == style_name
                results[model_name][style_name]["correct"].append(correct)
                # print results
                print(
                    f"""{model_name}: {np.mean(results[model_name][style_name]["correct"])}"""
                )

    # compute aggregate metrics
    for model_name, model_results in results.items():
        avg_correct = []
        for style_name in style_names:
            correct = np.mean(model_results[style_name]["correct"])
            avg_correct.append(correct)
            results[model_name][style_name]["avg"] = correct
        results[model_name]["overall"] = np.mean(avg_correct)

    # save result to json file
    json_filepath = os.path.join(output_dir, "zero_shot_results.json")
    with open(json_filepath, "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_gpu", action="store_true", help="Use GPU")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument(
        "--root_dir",
        type=str,
        default="/import/c4dm-datasets/deepafx2",
        help="Root directory of the dataset",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="/import/c4dm-datasets-ext/lcap/cls",
        help="Save directory",
    )
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--max_epochs", type=int, default=100, help="Max epochs")
    parser.add_argument(
        "--test_type", type=str, choices=["probe", "zero-shot"], default="zero-shot"
    )
    parser.add_argument(
        "--num_examples", type=int, default=10, help="Number of examples"
    )
    args = parser.parse_args()

    modes = ["speech", "music"]
    pl.seed_everything(42)

    # set up the pretrained models

    # load models
    print("Loading models...")
    models = {
        # --------------- self-supervised parameter models --------------- #
        "param-panns-l2-concat": {
            "model": load_param_model(
                "/import/c4dm-datasets-ext/lcap/param/lcap-param/myjhyqlz/checkpoints/last.ckpt",
                use_gpu=args.use_gpu,
            ),
            "embed_fn": get_param_embeds,
        },
        "param-clap": {
            "model": load_param_model(
                "/import/c4dm-datasets-ext/lcap/param/lcap-param/cxdl1pga/checkpoints/last.ckpt",
                use_gpu=args.use_gpu,
            ),
            "embed_fn": get_param_embeds,
        },
        # ---------------- style transfer models ---------------- #
        "deepafx-st": {
            "model": load_deepafx_st_model(
                "/import/c4dm-datasets-ext/lcap/style/lcap-style/k7c29vsa/checkpoints/last.ckpt",
                use_gpu=args.use_gpu,
                encoder_only=True,
            ),
            "embed_fn": get_deepafx_st_embeds,
        },
        "deepafx-st+": {
            "model": load_deepafx_st_model(
                "/import/c4dm-datasets-ext/lcap/style/lcap-style/87h3sb6n/checkpoints/last.ckpt",
                use_gpu=args.use_gpu,
                encoder_only=True,
            ),
            "embed_fn": get_deepafx_st_embeds,
        },
        # --------------- audio features --------------- #
        "mfccs": {  # these are mfccs
            "model": load_mfcc_feature_extractor(use_gpu=args.use_gpu),
            "embed_fn": get_mfcc_feature_embeds,
        },
        "mir-features": {
            "model": load_mir_feature_extractor(use_gpu=args.use_gpu),
            "embed_fn": get_mir_feature_embeds,
        },
        # --------------- pretrained models --------------- #
        "fx_encoder": {
            "model": load_fx_encoder_model(
                use_gpu=args.use_gpu,
            ),
            "embed_fn": get_fx_encoder_embeds,
        },
        "clap": {
            "model": load_clap_model(use_gpu=args.use_gpu),
            "embed_fn": get_clap_embeds,
        },
        "wav2vec2": {
            "model": load_wav2vec2_model(use_gpu=args.use_gpu),
            "embed_fn": get_wav2vec2_embeds,
        },
        "wav2clip": {
            "model": load_wav2clip_model(use_gpu=args.use_gpu),
            "embed_fn": get_wav2clip_embeds,
        },
        "vggish": {
            "model": load_vggish_model(use_gpu=args.use_gpu),
            "embed_fn": get_vggish_embeds,
        },
        "beats": {
            "model": load_beats_model(use_gpu=args.use_gpu),
            "embed_fn": get_beats_embeds,
        },
    }

    # train linear classifier on top of each
    # pretrained model for each mode
    for mode in modes:
        # create dataset
        if mode == "speech":
            dataset_path = "daps_24000_styles_100"
        elif mode == "music":
            dataset_path = "musdb18_44100_styles_100"

        output_dir = os.path.join("output", "cls", mode)
        os.makedirs(output_dir, exist_ok=True)

        if args.test_type == "probe":
            run_probe_task(
                args.root_dir,
                args.save_dir,
                models,
                max_epochs=args.max_epochs,
                batch_size=args.batch_size,
                lr=args.lr,
            )
        elif args.test_type == "zero-shot":
            run_zero_shot_task(
                args.root_dir,
                dataset_path,
                models,
                output_dir=output_dir,
                num_examples=args.num_examples,
            )
        else:
            raise ValueError(f"Unknown test type: {args.test_type}")
