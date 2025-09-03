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

class AxisAlignedConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, two_d=False):
        super(AxisAlignedConv, self).__init__()
        self.two_d = two_d
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size) if self.two_d else nn.Conv3d(in_channels, out_channels, kernel_size)

    def forward(self, x):
        return self.conv(x)

class ProbabilisticUNet(nn.Module):
    def __init__(self, unet, num_annotators, latent_dim=6, num_channels=1, patch_size=None, final_conv_channels=None, variance_scaling_factor=1.0, device=None):
        super(ProbabilisticUNet, self).__init__()
        self.unet = unet
        self.num_annotators = num_annotators
        self.latent_dim = latent_dim
        self.patch_size = patch_size
        self.two_d = len(self.patch_size) == 2
        self.final_conv_channels = final_conv_channels
        self.num_channels = num_channels
        self.variance_scaling_factor = variance_scaling_factor

        self.prior_net = AxisAlignedConv(self.final_conv_channels + num_annotators, 2 * latent_dim, two_d=self.two_d).to(device)
        self.posterior_net = AxisAlignedConv(self.final_conv_channels + num_channels, 2 * latent_dim, two_d=self.two_d).to(device)
        self.z_to_features = AxisAlignedConv(latent_dim, self.final_conv_channels, two_d=self.two_d).to(device)

        self.z_to_skip_features = nn.ModuleList()
        encoder_feature_sizes = [stage[-1].convs[-1].conv.out_channels for stage in self.unet.encoder.stages]
        decoder_stages = self.unet.decoder.stages
        num_injection_stages = len(decoder_stages) - 1
        
        for i in range(num_injection_stages):
            skip_channels = encoder_feature_sizes[-(i + 2)]
            z_conv = AxisAlignedConv(latent_dim, skip_channels, two_d=self.two_d).to(device)
            self.z_to_skip_features.append(z_conv)

    def forward(self, x, annotator_id=None, target_seg=None):
        # [MODIFICATION] Read sampling_scale from environment variable for easy inference control
        import os
        try:
            sampling_scale = float(os.environ.get('SAMPLING_SCALE', 1.0))
        except (ValueError, TypeError):
            sampling_scale = 1.0

        encoder_features = self.unet.encoder(x)
        bottleneck = encoder_features[-1]

        if annotator_id is not None:
            annotator_id_map = annotator_id.view(annotator_id.size(0), self.num_annotators, 1, 1)
            if not self.two_d:
                annotator_id_map = annotator_id_map.unsqueeze(-1)
            annotator_id_map = annotator_id_map.expand(-1, -1, *bottleneck.shape[2:]).to(bottleneck.device)
            prior_params = self.prior_net(torch.cat([bottleneck, annotator_id_map], dim=1))
            prior_mu, prior_log_var = torch.chunk(prior_params, 2, dim=1)

            if self.variance_scaling_factor != 1.0:
                prior_log_var = prior_log_var + torch.log(torch.tensor(self.variance_scaling_factor).to(prior_log_var.device))
        else:
            prior_mu, prior_log_var = None, None

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

        if self.training:
            mu, log_var = posterior_mu, posterior_log_var
        else:
            mu, log_var = prior_mu, prior_log_var

        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        
        # [MODIFICATION] Apply sampling_scale during z sampling
        z = mu + eps * (std * sampling_scale)
        
        modified_encoder_features = list(encoder_features)
        z_bottleneck_features = self.z_to_features(z)
        modified_encoder_features[-1] = bottleneck + z_bottleneck_features

        for i, z_skip_conv in enumerate(self.z_to_skip_features):
            skip_connection_index = -(i + 2)
            skip_feature_map = modified_encoder_features[skip_connection_index]
            z_resized = nn.functional.interpolate(z, size=skip_feature_map.shape[2:], mode='nearest')
            z_skip_features = z_skip_conv(z_resized)
            modified_encoder_features[skip_connection_index] = skip_feature_map + z_skip_features
            
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
        prior_var = torch.exp(prior_log_var) + self.epsilon
        posterior_var = torch.exp(posterior_log_var) + self.epsilon

        kl_div = 0.5 * torch.sum(
            torch.log(prior_var) - torch.log(posterior_var) + 
            (posterior_var + (posterior_mu - prior_mu).pow(2)) / prior_var - 1,
            dim=1
        )
        return kl_div.mean()

class ProbabilisticUNetTrainer(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, variance_scaling_factor: float = 1.0, device: torch.device = torch.device('cuda')):
        # =================================================================================================
        # [MODIFICATION] nnUNetTrainer.__init__ code is copied here to handle the new 'variance_scaling_factor' argument.
        # This is necessary because the original __init__ uses inspect() in a way that is not compatible with subclassing.
        # =================================================================================================
        import inspect
        from torch.cuda import device_count
        from nnunetv2.paths import nnUNet_preprocessed, nnUNet_results
        from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
        from torch.cuda.amp import GradScaler
        from nnunetv2.training.logging.nnunet_logger import nnUNetLogger
        from datetime import datetime
        from batchgenerators.utilities.file_and_folder_operations import join, maybe_mkdir_p
        import torch.distributed as dist

        self.is_ddp = dist.is_available() and dist.is_initialized()
        self.local_rank = 0 if not self.is_ddp else dist.get_rank()
        self.device = device

        if self.is_ddp:
            print(f"I am local rank {self.local_rank}. {device_count()} GPUs are available. The world size is {dist.get_world_size()}.")
            print(f"Setting device to {self.device}")
            self.device = torch.device(type='cuda', index=self.local_rank)
        else:
            if self.device.type == 'cuda':
                self.device = torch.device(type='cuda', index=0)
            print(f"Using device: {self.device}")

        self.my_init_kwargs = {}
        for k in inspect.signature(self.__init__).parameters.keys():
            self.my_init_kwargs[k] = locals()[k]

        self.plans_manager = PlansManager(plans)
        self.configuration_manager = self.plans_manager.get_configuration(configuration)
        self.configuration_name = configuration
        self.dataset_json = dataset_json
        self.fold = fold

        self.preprocessed_dataset_folder_base = join(nnUNet_preprocessed, self.plans_manager.dataset_name) if nnUNet_preprocessed is not None else None
        self.output_folder_base = join(nnUNet_results, self.plans_manager.dataset_name, self.__class__.__name__ + '__' + self.plans_manager.plans_name + "__" + configuration) if nnUNet_results is not None else None
        self.output_folder = join(self.output_folder_base, f'fold_{fold}')
        self.preprocessed_dataset_folder = join(self.preprocessed_dataset_folder_base, self.configuration_manager.data_identifier)
        self.is_cascaded = self.configuration_manager.previous_stage_name is not None
        self.folder_with_segs_from_previous_stage = join(nnUNet_results, self.plans_manager.dataset_name, self.__class__.__name__ + '__' + self.plans_manager.plans_name + "__" + self.configuration_manager.previous_stage_name, 'predicted_next_stage', self.configuration_name) if self.is_cascaded else None

        # ProbabilisticUNetTrainer-specific modifications
        self.initial_lr = 1e-3
        self.kl_weight = 0.01
        self.variance_scaling_factor = variance_scaling_factor
        
        # Default nnUNetTrainer hyperparameters (can be overridden by child classes)
        self.weight_decay = 3e-5
        self.oversample_foreground_percent = 0.33
        self.probabilistic_oversampling = False
        self.num_iterations_per_epoch = 250
        self.num_val_iterations_per_epoch = 50
        self.num_epochs = 1000
        self.current_epoch = 0
        self.enable_deep_supervision = True

        self.label_manager = self.plans_manager.get_label_manager(dataset_json)
        self.num_input_channels = None
        self.network = None
        self.optimizer = self.lr_scheduler = None
        # [MODIFICATION] Explicitly set init_scale to a float value for GradScaler to avoid TypeError
        self.grad_scaler = GradScaler(init_scale=65536.0) if self.device.type == 'cuda' else None
        self.loss = None
        
        self.dataset_class = None
        self.dataloader_train = self.dataloader_val = None
        self._best_ema = None
        self.inference_allowed_mirroring_axes = None
        self.save_every = 50
        self.disable_checkpointing = False
        self.was_initialized = False

        timestamp = datetime.now()
        maybe_mkdir_p(self.output_folder)
        self.log_file = join(self.output_folder, "training_log_%d_%d_%d_%02.0d_%02.0d_%02.0d.txt" % (timestamp.year, timestamp.month, timestamp.day, timestamp.hour, timestamp.minute, timestamp.second))
        self.logger = nnUNetLogger()
        
        self.print_to_log_file("\n#######################################################################\n"
                               "Please cite the following paper when using nnU-Net:\n"
                               "Isensee, F., Jaeger, P. F., Kohl, S. A., Petersen, J., & Maier-Hein, K. H. (2021). "
                               "nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation. "
                               "Nature methods, 18(2), 203-211.\n"
                               "#######################################################################\n",
                               also_print_to_console=True, add_timestamp=False)

    def initialize(self):
        if not self.was_initialized:
            self.num_input_channels = determine_num_input_channels(self.plans_manager, self.configuration_manager,
                                                                   self.dataset_json)
            self._set_batch_size_and_oversample()
            self.dataloader_train, self.dataloader_val, num_annotators = self.get_dataloaders()
            self.num_annotators = num_annotators
            self.my_init_kwargs['num_annotators'] = num_annotators

            base_unet = self.build_network_architecture(
                self.configuration_manager.network_arch_class_name,
                self.configuration_manager.network_arch_init_kwargs,
                self.configuration_manager.network_arch_init_kwargs_req_import,
                self.num_input_channels,
                self.label_manager.num_segmentation_heads,
                self.enable_deep_supervision
            ).to(self.device)

            self.network = ProbabilisticUNet(base_unet, num_annotators=num_annotators, latent_dim=6, num_channels=self.label_manager.num_segmentation_heads, patch_size=self.configuration_manager.patch_size, final_conv_channels=self.configuration_manager.network_arch_init_kwargs['features_per_stage'][-1], variance_scaling_factor=self.variance_scaling_factor, device=self.device)

            if self._do_i_compile():
                self.print_to_log_file('Using torch.compile...')
                self.network = torch.compile(self.network)

            self.optimizer, self.lr_scheduler = self.configure_optimizers()
            if self.is_ddp:
                self.network = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.network)
                self.network = DDP(self.network, device_ids=[self.local_rank])

            self.loss = self._build_loss()
            self.dataset_class = infer_dataset_class(self.preprocessed_dataset_folder)
            self.was_initialized = True
        else:
            raise RuntimeError("You have called self.initialize even though the trainer was already initialized. That should not happen.")

    def _get_loss(self):
        recon_loss = DC_and_CE_loss({'batch_dice': self.configuration_manager.batch_dice, 'smooth': 1e-5, 'do_bg': False, 'ddp': self.is_ddp},
                                    {}, weight_ce=2, weight_dice=1,
                                    ignore_label=self.label_manager.ignore_label if self.label_manager.has_ignore_label else -100)
        
        kl_loss = KLDivergenceLoss()

        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            if deep_supervision_scales is not None:
                deep_supervision_scales_flat = [item for sublist in deep_supervision_scales for item in sublist]
            else:
                deep_supervision_scales_flat = None
            recon_loss = DeepSupervisionWrapper(recon_loss, deep_supervision_scales_flat)
        return recon_loss, kl_loss

    def on_epoch_start(self):
        super().on_epoch_start()
        self.kl_weight = min(self.kl_weight + 0.00001, 1.0)

    def train_step(self, batch: dict) -> dict:
        data = batch['data']
        target = batch['target']
        annotator_id = batch.get('annotator_id')

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True).long() for i in target]
        else:
            target = target.to(self.device, non_blocking=True).long()

        self.optimizer.zero_grad(set_to_none=True)
        
        from nnunetv2.utilities.helpers import dummy_context
        with torch.autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            outputs = self.network(data, annotator_id=annotator_id, target_seg=target[0] if self.enable_deep_supervision else target)
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
        annotator_id = batch.get('annotator_id')

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        with torch.no_grad():
            outputs = self.network(data, annotator_id=annotator_id)
            del data
            l = self.recon_loss(outputs["seg_output"], target)

        output = outputs["seg_output"]
        if self.enable_deep_supervision:
            output = output[0]
            target = target[0]

        axes = [0] + list(range(2, output.ndim))

        if self.label_manager.has_regions:
            predicted_segmentation_onehot = (torch.sigmoid(output) > 0.5).long()
        else:
            output_seg = output.argmax(1)[:, None]
            predicted_segmentation_onehot = torch.zeros(output.shape, device=output.device, dtype=torch.float32)
            predicted_segmentation_onehot.scatter_(1, output_seg, 1)
            del output_seg

        if self.label_manager.has_ignore_label:
            if not self.label_manager.has_regions:
                mask = (target != self.label_manager.ignore_label).float()
                target[target == self.label_manager.ignore_label] = 0
            else:
                mask = 1 - target[:, -1:]
                target = target[:, :-1]
        else:
            mask = None

        tp, fp, fn, _ = get_tp_fp_fn_tn(predicted_segmentation_onehot, target, axes=axes, mask=mask)

        tp_hard = tp.detach().cpu().numpy()
        fp_hard = fp.detach().cpu().numpy()
        fn_hard = fn.detach().cpu().numpy()

        if not self.label_manager.has_regions:
            tp_hard = tp_hard[1:]
            fp_hard = fp_hard[1:]
            fn_hard = fn_hard[1:]

        return {'loss': l.detach().cpu().numpy(), 'tp_hard': tp_hard, 'fp_hard': fp_hard, 'fn_hard': fn_hard}

    def on_train_start(self):
        super().on_train_start()
        self.recon_loss, self.kl_loss = self._get_loss()

    def on_train_end(self):
        self.save_checkpoint(join(self.output_folder, "checkpoint_final.pth"))