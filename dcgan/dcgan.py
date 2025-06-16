from dataclasses import dataclass

import torch
from dcgan_consts import ADAM_BETA1, ADAM_LEARNING_RATE, LATENT_DIM
from dcgan_discriminator import Discriminator
from dcgan_generator import Generator
from torch import nn
from torch.optim.adam import Adam


@dataclass
class TrainingObjects:
    D: Discriminator
    G: Generator
    D_opt: Adam
    G_opt: Adam
    loss_D: nn.BCELoss
    loss_G: nn.BCELoss


def setup():
    # get my objects!
    D = Discriminator(device="mps")
    G = Generator(device="mps")
    objs = TrainingObjects(
        D=D,
        G=G,
        D_opt=Adam(D.parameters(), lr=ADAM_LEARNING_RATE, betas=(ADAM_BETA1, 0.999)),
        G_opt=Adam(G.parameters(), lr=ADAM_LEARNING_RATE, betas=(ADAM_BETA1, 0.999)),
        loss_D=nn.BCELoss(),
        loss_G=nn.BCELoss(),
    )
    return objs


def single_inference_step(
    objs: TrainingObjects, num_samples: int, detach: bool = False
):
    input_noise = torch.randn(num_samples, LATENT_DIM, device="mps")
    if detach:
        gen_imgs = objs.G(input_noise).detach()
    else:
        gen_imgs = objs.G(input_noise)
    return gen_imgs


@dataclass
class TrainingResults:
    discriminator_loss: float
    generator_loss: float
    mean_score_on_real: float
    mean_score_on_fake: float


def _train_discriminator(objs, img_batch, gen_imgs):
    # discriminator training steps.
    noise_std = 0.05
    img_batch += torch.randn_like(img_batch) * noise_std
    gen_imgs += torch.randn_like(gen_imgs) * noise_std

    gen_preds = objs.D(gen_imgs)
    true_preds = objs.D(img_batch)

    all_false = torch.zeros(len(gen_preds), 1, device="mps", requires_grad=False)

    # add random noise to the true labels
    all_true = torch.clamp(
        torch.normal(mean=0.8, std=0.15, size=(len(true_preds), 1), device="mps"),
        min=0.7,
        max=1,
    )
    flip_mask = torch.rand(len(true_preds), 1, device="mps") < 0.05  # 5% chance
    all_true[flip_mask] = 0.1  # flip some real labels to fake

    discriminator_loss = 0.5 * (
        objs.loss_D(gen_preds, all_false) + objs.loss_D(true_preds, all_true)
    )

    # compute grad and update weights with optimizer
    objs.D_opt.zero_grad()
    discriminator_loss.backward()
    objs.D_opt.step()
    return gen_preds, true_preds, discriminator_loss


def _train_generator(objs, gen_imgs):
    discriminator_scores = objs.D(gen_imgs)
    all_true = torch.ones(len(gen_imgs), 1, device="mps", requires_grad=False)
    gen_loss = objs.loss_G(discriminator_scores, all_true)

    # compute grad and update weights
    objs.G_opt.zero_grad()
    gen_loss.backward()
    objs.G_opt.step()
    return gen_loss


def single_training_step(
    objs: TrainingObjects, num_samples: int, img_batch
) -> TrainingResults:

    # discriminator training steps.
    gen_imgs = single_inference_step(objs, num_samples, detach=True)

    gen_preds, true_preds, discriminator_loss = _train_discriminator(
        objs, img_batch, gen_imgs
    )

    # generator training
    for _ in range(2):
        # train twice
        gen_imgs = single_inference_step(objs, num_samples, detach=False)
        gen_loss = _train_generator(objs, gen_imgs)

    return TrainingResults(
        discriminator_loss=discriminator_loss.item(),
        generator_loss=gen_loss.item(),
        mean_score_on_real=true_preds.mean().item(),
        mean_score_on_fake=gen_preds.mean().item(),
    )
