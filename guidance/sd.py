from diffusers import DDIMScheduler, StableDiffusionPipeline

import torch
import torch.nn as nn


class StableDiffusion(nn.Module):
    def __init__(self, args, t_range=[0.02, 0.98]):
        super().__init__()

        self.device = args.device
        self.dtype = args.precision
        print(f'[INFO] loading stable diffusion...')

        model_key = "stabilityai/stable-diffusion-2-1-base"
        pipe = StableDiffusionPipeline.from_pretrained(
            model_key, torch_dtype=self.dtype,
        )

        pipe.to(self.device)
        self.vae = pipe.vae
        self.tokenizer = pipe.tokenizer
        self.text_encoder = pipe.text_encoder
        self.unet = pipe.unet
        self.scheduler = DDIMScheduler.from_pretrained(
            model_key, subfolder="scheduler", torch_dtype=self.dtype,
        )

        del pipe

        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.t_range = t_range
        self.min_step = int(self.num_train_timesteps * t_range[0])
        self.max_step = int(self.num_train_timesteps * t_range[1])
        self.alphas = self.scheduler.alphas_cumprod.to(self.device) # for convenience

        print(f'[INFO] loaded stable diffusion!')

    @torch.no_grad()
    def get_text_embeds(self, prompt):
        inputs = self.tokenizer(prompt, padding='max_length', max_length=self.tokenizer.model_max_length, return_tensors='pt')
        embeddings = self.text_encoder(inputs.input_ids.to(self.device))[0]

        return embeddings
    
    
    def get_noise_preds(self, latents_noisy, t, text_embeddings, guidance_scale=100):
        latent_model_input = torch.cat([latents_noisy] * 2)
            
        tt = torch.cat([t] * 2)
        noise_pred = self.unet(latent_model_input, tt, encoder_hidden_states=text_embeddings).sample

        noise_pred_uncond, noise_pred_pos = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_pos - noise_pred_uncond)
        
        return noise_pred


    def get_sds_loss(
        self, 
        latents,
        text_embeddings, 
        guidance_scale=100, 
        grad_scale=1,
    ):
        
        # TODO: Implement the loss function for SDS
        # raise NotImplementedError("SDS is not implemented yet.")

        # Randomly sample timesteps
        t = torch.randint(self.min_step, self.max_step + 1, [latents.shape[0]], device=self.device)
        
        # Add noise to the latents according to the noise magnitude at each timestep
        noise = torch.randn_like(latents)
        latents_noisy = self.scheduler.add_noise(latents, noise, t)
        
        # Predict the noise residual
        noise_pred = self.get_noise_preds(
            latents_noisy, t, text_embeddings, guidance_scale=guidance_scale
        )
        
        # Calculate the loss as the difference between predicted and target noise
        loss = grad_scale * (noise_pred - noise)
        loss = (loss - latents).detach() + latents
        loss = (loss ** 2).mean()
        
        return loss

    
    def get_pds_loss(
        self, src_latents, tgt_latents, 
        src_text_embedding, tgt_text_embedding,
        guidance_scale=7.5, 
        grad_scale=1,
    ):
        
        # TODO: Implement the loss function for PDS
        # raise NotImplementedError("PDS is not implemented yet.")
        
        t = torch.randint(self.min_step, self.max_step, (1,)).to(self.device)
        alpha_t_curr = self.alphas[t]
        alpha_t_prev = self.alphas[t-1]
        var = ((1 - alpha_t_prev) / (1 - alpha_t_curr) * (1 - alpha_t_curr / alpha_t_prev)).sqrt()

        noise_t = torch.randn_like(src_latents)
        noise_t_prev = torch.randn_like(src_latents)

        def compute_latent(x, text_embedding):
            x_t_curr = alpha_t_curr.sqrt() * x + (1 - alpha_t_curr).sqrt() * noise_t
            x_t_prev = alpha_t_prev.sqrt() * x + (1 - alpha_t_prev).sqrt() * noise_t_prev

            with torch.no_grad():
                noise_pred = self.get_noise_preds(x_t_curr, t, text_embedding, guidance_scale)
            pred_x0 = (x_t_curr - (1 - alpha_t_curr).sqrt() * noise_pred) / alpha_t_curr.sqrt()
            mean = (1 - alpha_t_prev).sqrt() * pred_x0 + (1 - alpha_t_prev - var ** 2) * noise_pred

            z = (x_t_prev - mean) / var

            return z

        z_src = compute_latent(src_latents, src_text_embedding)
        z_tgt = compute_latent(tgt_latents, tgt_text_embedding)

        loss = z_tgt - z_src
        loss = (loss - tgt_latents).detach() + tgt_latents
        loss = (loss ** 2).mean()

        return loss
    
    
    @torch.no_grad()
    def decode_latents(self, latents):

        latents = 1 / self.vae.config.scaling_factor * latents

        imgs = self.vae.decode(latents).sample
        imgs = (imgs / 2 + 0.5).clamp(0, 1)

        return imgs

    @torch.no_grad()
    def encode_imgs(self, imgs):
        # imgs: [B, 3, H, W]

        imgs = 2 * imgs - 1

        posterior = self.vae.encode(imgs).latent_dist
        latents = posterior.sample() * self.vae.config.scaling_factor

        return latents
