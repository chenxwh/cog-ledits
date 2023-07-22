import os
from torch import autocast, inference_mode
from diffusers import StableDiffusionPipeline
from diffusers import DDIMScheduler
from cog import BasePredictor, Input, Path

from inversion_utils import *
from modified_pipeline_semantic_stable_diffusion import SemanticStableDiffusionPipeline


class Predictor(BasePredictor):
    def setup(self):
        sd_model_id = "runwayml/stable-diffusion-v1-5"
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.sd_pipe = StableDiffusionPipeline.from_pretrained(
            sd_model_id, cache_dir="models_cache"
        ).to(device)
        self.sd_pipe.scheduler = DDIMScheduler.from_config(
            sd_model_id, subfolder="scheduler", cache_dir="models_cache"
        )
        self.sega_pipe = SemanticStableDiffusionPipeline.from_pretrained(
            sd_model_id, cache_dir="models_cache"
        ).to(device)
        # pass

    def predict(
        self,
        input_image: Path = Input(description="Input image"),
        concept_to_add: str = Input(
            description="concept to add in the target image", default=""
        ),
        guidance_scale: float = Input(
            description="guidance scale to add the above concept", default=15
        ),
        concept_to_remove: str = Input(
            description="concept to remove from the original image", default=""
        ),
        neg_guidance_scale: float = Input(
            description="guidance scale to remove the above concept", default=7
        ),
        num_diffusion_steps: int = Input(
            description="number of diffusion steps", default=100
        ),
        target_prompt: str = Input(
            description="DDPM Inversion Prompt. Can help with global changes, modify to what you would like to see.",
            default="",
        ),
        target_guidance_scale: float = Input(
            description="guidance scale for the target_prompt provided above",
            default=20,
        ),
        skip_steps: int = Input(description="DDPM denoising steps to skip", default=36),
        seed: int = Input(
            description="Random seed. Leave blank to randomize the seed", default=None
        ),
    ) -> Path:
        """Run a single prediction on the model"""

        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")
        torch.manual_seed(seed)

        source_prompt = ""
        source_guidance_scale = 3.5

        # SEGA only params
        edit_concepts = [concept_to_remove, concept_to_add]
        edit_guidance_scales = [neg_guidance_scale, guidance_scale]
        warmup_steps = [1, 1]
        reverse_editing = [True, False]
        thresholds = [0.95, 0.95]

        # Invert with ddpm
        x0 = load_512(str(input_image), device="cuda")
        # noise maps and latents
        zs, wts = invert(
            self.sd_pipe,
            x0=x0,
            prompt_src=source_prompt,
            num_diffusion_steps=num_diffusion_steps,
            cfg_scale_src=source_guidance_scale,
        )

        # edit with the pre-computed latents and noise maps
        sega_ddpm_edited_img = edit(
            self.sega_pipe,
            wts,
            zs,
            tar_prompt=target_prompt,
            steps=num_diffusion_steps,
            skip=skip_steps,
            tar_cfg_scale=target_guidance_scale,
            edit_concept=edit_concepts,
            guidnace_scale=edit_guidance_scales,
            warmup=warmup_steps,
            neg_guidance=reverse_editing,
            threshold=thresholds,
        )

        output = "/tmp/out.png"
        sega_ddpm_edited_img.save(output)

        return Path(output)


def invert(
    sd_pipe,
    x0: torch.FloatTensor,
    prompt_src: str = "",
    num_diffusion_steps=100,
    cfg_scale_src=3.5,
    eta=1,
):

    #  inverts a real image according to Algorihm 1 in https://arxiv.org/pdf/2304.06140.pdf,
    #  based on the code in https://github.com/inbarhub/DDPM_inversion

    #  returns wt, zs, wts:
    #  wt - inverted latent
    #  wts - intermediate inverted latents
    #  zs - noise maps

    sd_pipe.scheduler.set_timesteps(num_diffusion_steps)

    # vae encode image
    with autocast("cuda"), inference_mode():
        w0 = (sd_pipe.vae.encode(x0).latent_dist.mode() * 0.18215).float()

    # find Zs and wts - forward process
    wt, zs, wts = inversion_forward_process(
        sd_pipe,
        w0,
        etas=eta,
        prompt=prompt_src,
        cfg_scale=cfg_scale_src,
        prog_bar=True,
        num_inference_steps=num_diffusion_steps,
    )
    return zs, wts


def edit(
    sega_pipe,
    wts,
    zs,
    tar_prompt="",
    steps=100,
    skip=36,
    tar_cfg_scale=15,
    edit_concept="",
    guidnace_scale=7,
    warmup=1,
    neg_guidance=False,
    threshold=0.95,
):

    # SEGA
    # parse concepts and neg guidance
    editing_args = dict(
        editing_prompt=edit_concept,
        reverse_editing_direction=neg_guidance,
        edit_warmup_steps=warmup,
        edit_guidance_scale=guidnace_scale,
        edit_threshold=threshold,
        edit_momentum_scale=0.5,
        edit_mom_beta=0.6,
        eta=1,
    )
    latnets = wts[skip].expand(1, -1, -1, -1)
    sega_out = sega_pipe(
        prompt=tar_prompt,
        latents=latnets,
        guidance_scale=tar_cfg_scale,
        num_images_per_prompt=1,
        num_inference_steps=steps,
        use_ddpm=True,
        wts=wts,
        zs=zs[skip:],
        **editing_args
    )
    return sega_out.images[0]
