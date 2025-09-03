
import torch
import torch.nn as nn
import numpy as np
from torch.nn.parallel import DistributedDataParallel as DDP
from time import time
from batchgenerators.utilities.file_and_folder_operations import join

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.loss.dice import get_tp_fp_fn_tn
from nnunetv2.training.loss.compound_losses import DC_and_CE_loss
from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels
from nnunetv2.training.dataloading.nnunet_dataset import infer_dataset_class
from nnunetv2.utilities.collate_outputs import collate_outputs

# A simple MLP for the prior and posterior networks
class AxisAlignedConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, two_d=False):
        super(AxisAlignedConv, self).__init__()
        self.two_d = two_d
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size) if self.two_d else nn.Conv3d(in_channels, out_channels, kernel_size)

    def forward(self, x):
        return self.conv(x)

class ProbabilisticUNet(nn.Module):
    def __init__(self, unet, num_annotators, latent_dim=6, num_channels=1, patch_size=None, final_conv_channels=None, device=None):
        super(ProbabilisticUNet, self).__init__()
        self.unet = unet
        self.num_annotators = num_annotators
        self.latent_dim = latent_dim
        self.patch_size = patch_size
        self.two_d = len(self.patch_size) == 2
        self.final_conv_channels = final_conv_channels
        self.num_channels = num_channels

        # Prior network
        self.prior_net = AxisAlignedConv(self.final_conv_channels + num_annotators, 2 * latent_dim, two_d=self.two_d).to(device)

        # Posterior network
        self.posterior_net = AxisAlignedConv(self.final_conv_channels + num_channels, 2 * latent_dim, two_d=self.two_d).to(device)

        # Latent vector to feature map for the bottleneck
        self.z_to_features = AxisAlignedConv(latent_dim, self.final_conv_channels, two_d=self.two_d).to(device)

        # =================================================================================================
        # [MODIFICATION START] 주석: z를 디코더의 Skip Connection에 주입하기 위한 Conv 레이어들을 추가합니다.
        # =================================================================================================
        
        self.z_to_skip_features = nn.ModuleList()
        
        # [CORRECTION 3] 주석: ConvDropoutNormReLU 모듈의 구조에 맞게 feature size를 가져옵니다.
        # 각 stage의 마지막 블록(-1) -> convs 리스트의 마지막 요소(-1) -> 실제 conv 레이어(.conv)의 출력 채널(.out_channels)을 참조합니다.
        encoder_feature_sizes = [stage[-1].convs[-1].conv.out_channels for stage in self.unet.encoder.stages]
        decoder_stages = self.unet.decoder.stages
        
        num_injection_stages = len(decoder_stages) - 1
        
        for i in range(num_injection_stages):
            skip_channels = encoder_feature_sizes[-(i + 2)]
            z_conv = AxisAlignedConv(latent_dim, skip_channels, two_d=self.two_d).to(device)
            self.z_to_skip_features.append(z_conv)
        
        # =================================================================================================
        # [MODIFICATION END]
        # =================================================================================================

    def forward(self, x, annotator_id=None, target_seg=None):
        # Pass input through the UNet encoder
        encoder_features = self.unet.encoder(x)
        bottleneck = encoder_features[-1]

        # Prior (used during inference and training)
        if annotator_id is not None:
            annotator_id_map = annotator_id.view(annotator_id.size(0), self.num_annotators, 1, 1)
            if not self.two_d:
                annotator_id_map = annotator_id_map.unsqueeze(-1)
            annotator_id_map = annotator_id_map.expand(-1, -1, *bottleneck.shape[2:]).to(bottleneck.device)
            prior_params = self.prior_net(torch.cat([bottleneck, annotator_id_map], dim=1))
            prior_mu, prior_log_var = torch.chunk(prior_params, 2, dim=1)
        else:
            prior_mu, prior_log_var = None, None

        # Posterior (used during training only)
        if target_seg is not None:
            target_seg_resized = torch.nn.functional.interpolate(
                target_seg.float(), size=bottleneck.shape[2:], mode='nearest'
            ).to(target_seg.dtype)
            if self.two_d:
                target_seg_onehot = torch.nn.functional.one_hot(target_seg_resized.squeeze(1).long(), num_classes=self.num_channels).permute(0, 3, 1, 2).float()
            else:
                target_seg_onehot = torch.nn.functional.one_hot(target_seg_resized.squeeze(1).long(), num_classes=self.num_channels).permute(0, 4, 1, 2, 3).float()
            posterior_input = torch.cat([bottleneck, target_seg_onehot], dim=1)
            posterior_params = self.posterior_net(posterior_input)
            posterior_mu, posterior_log_var = torch.chunk(posterior_params, 2, dim=1)
        else:
            posterior_mu, posterior_log_var = None, None

        # Sample from posterior during training, from prior during inference
        if self.training:
            mu, log_var = posterior_mu, posterior_log_var
        else:
            mu, log_var = prior_mu, prior_log_var

        # Reparameterization trick
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mu + eps * std

        # =================================================================================================
        # [MODIFICATION START] 주석: z를 병목과 여러 Skip Connection에 주입하는 로직입니다.
        # =================================================================================================
        
        modified_encoder_features = list(encoder_features)

        z_bottleneck_features = self.z_to_features(z)
        modified_encoder_features[-1] = bottleneck + z_bottleneck_features

        for i, z_skip_conv in enumerate(self.z_to_skip_features):
            skip_connection_index = -(i + 2)
            skip_feature_map = modified_encoder_features[skip_connection_index]

            z_resized = nn.functional.interpolate(z, size=skip_feature_map.shape[2:], mode='nearest')

            z_skip_features = z_skip_conv(z_resized)
            modified_encoder_features[skip_connection_index] = skip_feature_map + z_skip_features
            
        # =================================================================================================
        # [MODIFICATION END]
        # =================================================================================================

        seg_output = self.unet.decoder(modified_encoder_features)

        return {
            "seg_output": seg_output,
            "prior_mu": prior_mu, "prior_log_var": prior_log_var,
            "posterior_mu": posterior_mu, "posterior_log_var": posterior_log_var
        }

class KLDivergenceLoss(nn.Module):
    def __init__(self, epsilon=1e-8):
        super(KLDivergenceLoss, self).__init__()
        self.epsilon = epsilon

    def forward(self, prior_mu, prior_log_var, posterior_mu, posterior_log_var):
        # Add epsilon for numerical stability
        prior_var = torch.exp(prior_log_var) + self.epsilon
        posterior_var = torch.exp(posterior_log_var) + self.epsilon

        kl_div = 0.5 * torch.sum(
            torch.log(prior_var) - torch.log(posterior_var) + 
            (posterior_var + (posterior_mu - prior_mu).pow(2)) / prior_var - 1,
            dim=1
        )
        return kl_div.mean()

class ProbabilisticUNetTrainer(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        # [MODIFICATION] 학습 안정화를 위해 초기 학습률을 1e-3으로 낮춥니다.
        self.initial_lr = 1e-2
        # [MODIFICATION] 학습률 변경에 맞춰 KL 가중치를 0.01로 재조정합니다.
        self.kl_weight = 0.01 # Initial KL weight

    def initialize(self):
        if not self.was_initialized:
            self.num_input_channels = determine_num_input_channels(self.plans_manager, self.configuration_manager,
                                                                   self.dataset_json)

            # DDP batch size and oversampling can differ between workers and needs adaptation
            # we need to change the batch size in DDP because we don't use any of those distributed samplers
            self._set_batch_size_and_oversample()

            # dataloaders must be instantiated here (instead of __init__) because they need access to the training data
            # which may not be present  when doing inference
            self.dataloader_train, self.dataloader_val, num_annotators = self.get_dataloaders()
            self.num_annotators = num_annotators
            # Add num_annotators to my_init_kwargs so it gets saved in the checkpoint
            self.my_init_kwargs['num_annotators'] = num_annotators

            # Build the base UNet first
            base_unet = self.build_network_architecture(
                self.configuration_manager.network_arch_class_name,
                self.configuration_manager.network_arch_init_kwargs,
                self.configuration_manager.network_arch_init_kwargs_req_import,
                self.num_input_channels,
                self.label_manager.num_segmentation_heads,
                self.enable_deep_supervision
            ).to(self.device)

            # Now wrap the base UNet with ProbabilisticUNet
            # num_annotators = self.dataloader_train.data_loader.num_annotators if hasattr(self.dataloader_train.data_loader, 'num_annotators') else 0
            if num_annotators == 0:
                raise ValueError("ProbabilisticUNetTrainer requires a dataloader with annotator information.")
            self.network = ProbabilisticUNet(base_unet, num_annotators=num_annotators, latent_dim=6, num_channels=self.label_manager.num_segmentation_heads, patch_size=self.configuration_manager.patch_size, final_conv_channels=self.configuration_manager.network_arch_init_kwargs['features_per_stage'][-1], device=self.device)

            # compile network for free speedup
            if self._do_i_compile():
                self.print_to_log_file('Using torch.compile...')
                self.network = torch.compile(self.network)

            self.optimizer, self.lr_scheduler = self.configure_optimizers()
            # if ddp, wrap in DDP wrapper
            if self.is_ddp:
                self.network = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.network)
                self.network = DDP(self.network, device_ids=[self.local_rank])

            self.loss = self._build_loss()

            self.dataset_class = infer_dataset_class(self.preprocessed_dataset_folder)

            self.was_initialized = True
        else:
            raise RuntimeError("You have called self.initialize even though the trainer was already initialized. "
                               "That should not happen.")

    def _get_loss(self):
        # Reconstruction loss is the same as in nnUNetTrainer
        recon_loss = DC_and_CE_loss({'batch_dice': self.configuration_manager.batch_dice, 'smooth': 1e-5, 'do_bg': False, 'ddp': self.is_ddp},
                                    {}, weight_ce=2, weight_dice=1,
                                    ignore_label=self.label_manager.ignore_label if self.label_manager.has_ignore_label else -100)
        
        # KL Divergence Loss
        kl_loss = KLDivergenceLoss()

        # Wrap recon_loss in deep supervision if required
        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            # Fix: Convert deep_supervision_scales to a flat list of floats
            if deep_supervision_scales is not None:
                deep_supervision_scales_flat = [item for sublist in deep_supervision_scales for item in sublist]
            else:
                deep_supervision_scales_flat = None

            recon_loss = DeepSupervisionWrapper(recon_loss, deep_supervision_scales_flat)
            # kl_loss는 DeepSupervisionWrapper로 감싸지 않습니다.

        return recon_loss, kl_loss

    def on_epoch_start(self):
        super().on_epoch_start()
        # Anneal KL weight
        self.kl_weight = min(self.kl_weight + 0.00001, 1.0) # Anneal KL weight (slower)

    def train_step(self, batch: dict) -> dict:
        data = batch['data']
        target = batch['target']
        annotator_id = batch.get('annotator_id') # .get() is safer

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True).long() for i in target]
        else:
            target = target.to(self.device, non_blocking=True).long()

        self.optimizer.zero_grad(set_to_none=True)
        
        # Autocast can be annoying
        # If the device_type is 'cpu' then it's slow as heck and needs to be disabled.
        # If the device_type is 'mps' then it will complain that mps is not implemented, even if enabled=False is set. Whyyyyyyy. (this is why we don't make use of enabled=False)
        # So autocast will only be active if we have a cuda device.
        from nnunetv2.utilities.helpers import dummy_context
        with torch.autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            # Forward pass
            outputs = self.network(data, annotator_id=annotator_id, target_seg=target[0] if self.enable_deep_supervision else target)
            
            # Calculate losses
            recon_loss = self.recon_loss(outputs["seg_output"], target)
            kl_loss = self.kl_loss(outputs["prior_mu"], outputs["prior_log_var"], outputs["posterior_mu"], outputs["posterior_log_var"])
            
            total_loss = recon_loss + self.kl_weight * kl_loss

        if self.grad_scaler is not None:
            self.grad_scaler.scale(total_loss).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()

        return {'loss': total_loss.detach().cpu().numpy(), 'recon_loss': recon_loss.detach().cpu().numpy(), 'kl_loss': kl_loss.detach().cpu().numpy()}

    def validation_step(self, batch: dict) -> dict:
        data = batch['data']
        target = batch['target']
        annotator_id = batch.get('annotator_id') # Get the actual annotator_id from the batch

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True).long() for i in target]
        else:
            target = target.to(self.device, non_blocking=True).long()

        with torch.no_grad():
            # Use the actual annotator_id from the batch for validation
            outputs = self.network(data, annotator_id=annotator_id) 
            del data
            l = self.recon_loss(outputs["seg_output"], target)
        
        # we only need the output of the validation set, and not deep supervision
        output = outputs["seg_output"]
        if self.enable_deep_supervision:
            output = output[0]
            target = target[0]

        # the following is needed for online evaluation. Fake dice (green line)
        axes = [0] + list(range(2, output.ndim))

        if self.label_manager.has_regions:
            predicted_segmentation_onehot = (torch.sigmoid(output) > 0.5).long()
        else:
            # no need for softmax
            output_seg = output.argmax(1)[:, None]
            predicted_segmentation_onehot = torch.zeros(output.shape, device=output.device, dtype=torch.float32)
            predicted_segmentation_onehot.scatter_(1, output_seg, 1)
            del output_seg

        if self.label_manager.has_ignore_label:
            if not self.label_manager.has_regions:
                mask = (target != self.label_manager.ignore_label).float()
                # CAREFUL that you don't rely on target after this line!
                target[target == self.label_manager.ignore_label] = 0
            else:
                mask = 1 - target[:, -1:]
                # CAREFUL that you don't rely on target after this line!
                target = target[:, :-1]
        else:
            mask = None

        tp, fp, fn, _ = get_tp_fp_fn_tn(predicted_segmentation_onehot, target, axes=axes, mask=mask)

        # sum over batch
        tp_hard = np.atleast_1d(tp.sum(0).detach().cpu().numpy())
        fp_hard = np.atleast_1d(fp.sum(0).detach().cpu().numpy())
        fn_hard = np.atleast_1d(fn.sum(0).detach().cpu().numpy())

        if not self.label_manager.has_regions:
            # if we train with regions all segmentation heads predict some kind of foreground. In conventional
            # (softmax training) there needs tobe one output for the background. We are not interested in the
            # background Dice
            # [1:] in order to remove background
            tp_hard = tp_hard[1:]
            fp_hard = fp_hard[1:]
            fn_hard = fn_hard[1:]

        return {'loss': l.detach().cpu().numpy(), 'tp_hard': tp_hard, 'fp_hard': fp_hard, 'fn_hard': fn_hard}

    def on_train_start(self):
        super().on_train_start()
        self.recon_loss, self.kl_loss = self._get_loss()

    def on_validation_epoch_end(self, val_outputs: list):
        outputs_collated = collate_outputs(val_outputs)
        tp = np.sum(outputs_collated['tp_hard'], 0)
        fp = np.sum(outputs_collated['fp_hard'], 0)
        fn = np.sum(outputs_collated['fn_hard'], 0)

        if self.is_ddp:
            world_size = dist.get_world_size()

            tps = [None for _ in range(world_size)]
            dist.all_gather_object(tps, tp)
            tp = np.vstack([i[None] for i in tps]).sum(0)

            fps = [None for _ in range(world_size)]
            dist.all_gather_object(fps, fp)
            fp = np.vstack([i[None] for i in fps]).sum(0)

            fns = [None for _ in range(world_size)]
            dist.all_gather_object(fns, fn)
            fn = np.vstack([i[None] for i in fns]).sum(0)

            losses_val = [None for _ in range(world_size)]
            dist.all_gather_object(losses_val, outputs_collated['loss'])
            loss_here = np.vstack(losses_val).mean()
        else:
            loss_here = np.mean(outputs_collated['loss'])

        global_dc_per_class = [2 * i / (2 * i + j + k) for i, j, k in zip(tp, fp, fn) if (2 * i + j + k) > 0]
        
        # Handle case where global_dc_per_class might be empty
        if not global_dc_per_class:
            mean_fg_dice = 0.0
        else:
            mean_fg_dice = np.nanmean(global_dc_per_class)
        self.logger.log('mean_fg_dice', mean_fg_dice, self.current_epoch)
        self.logger.log('dice_per_class_or_region', global_dc_per_class, self.current_epoch)
        self.logger.log('val_losses', loss_here, self.current_epoch)

    def on_epoch_end(self):
        # nnUNetTrainer의 on_epoch_end를 그대로 사용하지 않고, 로그 출력 부분을 커스터마이징합니다.
        self.logger.log('epoch_end_timestamps', time(), self.current_epoch)

        # train/val loss 및 클래스별 dice score 출력
        self.print_to_log_file('train_loss', np.round(self.logger.my_fantastic_logging['train_losses'][-1], decimals=4))
        self.print_to_log_file('val_loss', np.round(self.logger.my_fantastic_logging['val_losses'][-1], decimals=4))
        
        # [MODIFICATION] 클래스별 Dice 점수를 출력하도록 수정
        dice_scores = self.logger.my_fantastic_logging['dice_per_class_or_region'][-1]
        self.print_to_log_file('Pseudo dice', [np.round(i, decimals=4) for i in dice_scores])

        self.print_to_log_file(
            f"Epoch time: {np.round(self.logger.my_fantastic_logging['epoch_end_timestamps'][-1] - self.logger.my_fantastic_logging['epoch_start_timestamps'][-1], decimals=2)} s")

        # handling periodic checkpointing
        current_epoch = self.current_epoch
        if (current_epoch + 1) % self.save_every == 0 and current_epoch != (self.num_epochs - 1):
            self.save_checkpoint(join(self.output_folder, 'checkpoint_latest.pth'))

        # handle 'best' checkpointing
        if self._best_ema is None or self.logger.my_fantastic_logging['ema_fg_dice'][-1] > self._best_ema:
            self._best_ema = self.logger.my_fantastic_logging['ema_fg_dice'][-1]
            self.print_to_log_file(f"Yayy! New best EMA pseudo Dice: {np.round(self._best_ema, decimals=4)}")
            self.save_checkpoint(join(self.output_folder, 'checkpoint_best.pth'))

        if self.local_rank == 0:
            self.logger.plot_progress_png(self.output_folder)

        self.current_epoch += 1
