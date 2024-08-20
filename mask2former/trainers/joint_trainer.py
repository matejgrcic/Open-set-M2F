import time
# from mask2former.trainers.trainer import Trainer
import torch
from detectron2.utils.events import get_event_storage
from detectron2.engine.train_loop import AMPTrainer
import math
import numpy as np
import random 
from detectron2.structures import Boxes, ImageList, Instances, BitMasks


class JointTrainer(AMPTrainer):

    def __init__(self, model, data_loader, optimizer, gather_metric_period=1, zero_grad_before_forward=False, grad_scaler=None, precision: torch.dtype = torch.float16, log_grad_scaler: bool = False, async_write_metrics=False):
        super().__init__(model, data_loader, optimizer, gather_metric_period, zero_grad_before_forward, grad_scaler, precision, log_grad_scaler, async_write_metrics)
        
        self.flow = model.module.flow
        self.device = model.device

        self.flow_optimizer = torch.optim.Adamax(self.flow.parameters(), lr=0.00001, betas=(0.9, 0.999), eps=1e-7)

        self.pixel_mean = model.module.pixel_mean
        self.pixel_std = model.module.pixel_std
        self.size_divisibility = model.module.size_divisibility

        self.ignore_label = 255


    def sample_shape(self, bs):
        sizes = [i for i in range(16, 217, 8)]
        w = np.random.choice(sizes) // 8
        h = np.random.choice(sizes) // 8

        return (bs, None, h, w)
    
    def _paste_square_patch(self, x, label, ood_patch, ood_id):
        N, _, p_h, p_w = ood_patch.shape
        _, _, h, w = x.shape
        id_patch = torch.zeros_like(ood_patch)
        for i in range(x.size(0)):
            pos_i = random.randint(0, h - p_h)
            pos_j = random.randint(0, w - p_w)
            id_patch[i] = x[i, :, pos_i: pos_i + p_h, pos_j: pos_j + p_w].detach()
            x[i, :, pos_i: pos_i + p_h, pos_j: pos_j + p_w] = ood_patch[i]
            label[i, pos_i: pos_i + p_h, pos_j: pos_j + p_w] = ood_id
        return x, id_patch, label



    def run_step(self):
        """
        Implement the AMP training logic.
        """
        assert self.model.training, "[AMPTrainer] model was changed to eval mode!"
        assert torch.cuda.is_available(), "[AMPTrainer] CUDA is required for AMP training!"
        from torch.cuda.amp import autocast

        start = time.perf_counter()
        data = next(self._data_loader_iter)
        data_time = time.perf_counter() - start

        if self.zero_grad_before_forward:
            self.optimizer.zero_grad()

        self.flow_optimizer.zero_grad()

        # paste negative ---------------------------------------------------
        # get images and gt in batch
        images = [x["image"].to(self.device) for x in data]
        images = ImageList.from_tensors(images, self.size_divisibility)
        image_sizes = images.image_sizes
        images = images.tensor

        sem_seg_gt = [x["sem_seg"] for x in data]

        sem_seg_gt = ImageList.from_tensors(sem_seg_gt, self.size_divisibility).tensor

        bs = self.sample_shape(images.shape[0])
        ood_patch = self.flow.sample(bs)

        images, id_patch, sem_seg_gt = self._paste_square_patch(images, sem_seg_gt, ood_patch, 19)
        
        images = (images - self.pixel_mean) / self.pixel_std
                
        instances_mix = []
        sem_seg_gt = sem_seg_gt.numpy()
        for i in range(sem_seg_gt.shape[0]):
            # get gt instances
            sem_seg_gt_ = sem_seg_gt[i]
            image_shape = (images.shape[-2], images.shape[-1])
            instances = Instances(image_shape)
            classes = np.unique(sem_seg_gt_)
            # remove ignored region
            classes = classes[classes != self.ignore_label]
            instances.gt_classes = torch.tensor(classes, dtype=torch.int64)

            masks = []
            for class_id in classes:
                masks.append(sem_seg_gt_ == class_id)

            if len(masks) == 0:
                # Some image does not have annotation (all ignored)
                instances.gt_masks = torch.zeros((0, sem_seg_gt_.shape[-2], sem_seg_gt_.shape[-1]))
            else:
                masks = BitMasks(
                    torch.stack([torch.from_numpy(np.ascontiguousarray(x.copy())) for x in masks])
                )
                instances.gt_masks = masks.tensor

            instances_mix.append(instances.to(self.device))

        ## -----------------------------------------------------------------

        with autocast(dtype=self.precision):
            loss_dict = self.model(data, images, instances_mix, image_sizes)
            if isinstance(loss_dict, torch.Tensor):
                losses = loss_dict
                loss_dict = {"total_loss": loss_dict}
            else:
                losses = sum(loss_dict.values())

        if not self.zero_grad_before_forward:
            self.optimizer.zero_grad()

        self.grad_scaler.scale(losses).backward()

        if self.log_grad_scaler:
            storage = get_event_storage()
            storage.put_scalar("[metric]grad_scaler", self.grad_scaler.get_scale())

        self.after_backward()

        if self.async_write_metrics:
            # write metrics asynchronically
            self.concurrent_executor.submit(
                self._write_metrics, loss_dict, data_time, iter=self.iter
            )
        else:
            self._write_metrics(loss_dict, data_time)

        ### step towards id 
        id_patch = (id_patch * 255.).long()
        loss_mle = - self.flow.log_prob(id_patch).sum() / (math.log(2) * id_patch.shape.numel())
        loss_mle.backward()
        
        self.flow_optimizer.step()

        self.grad_scaler.step(self.optimizer)
        self.grad_scaler.update()
