import torch

class CustomScheduler:
    def __init__(self, optimizer, warmup_steps, total_steps, min_lr, max_lr, final_lr, T_0, T_mult):
        """
        Custom Learning Rate Scheduler.
        
        - Warmup (Linear): Increases from min_lr to max_lr over `warmup_steps`
        - Cosine Annealing: Decays from max_lr to mid-range (5e-4) until 8000 steps
        - Final Linear Decay: Reduces from 5e-4 to final_lr over last 2000 steps

        Args:
            optimizer: PyTorch optimizer
            warmup_steps: Number of warmup steps
            total_steps: Total training steps
            min_lr: Starting learning rate
            max_lr: Peak learning rate
            final_lr: Final learning rate at the end of training
        """
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.final_lr = final_lr
        self.current_step = 0
        self.T_0 = T_0
        self.T_mult = T_mult
        self.cos_anneal_stage = self.warmup_steps+self.T_0
        self.fin_mid_lr = (self.max_lr + self.min_lr)/2

        # Cosine Annealing Phase (Mid-Phase: 5e-4 as transition point)
        #self.cosine_scheduler = CosineAnnealingLR(optimizer, T_max=(8000 - warmup_steps), eta_min=5e-4)
        self.cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=self.T_0, T_mult=self.T_mult, eta_min=self.min_lr)

    def step(self, step=None):
        if step is not None:
            self.current_step = step
        else:
            self.current_step += 1

        if self.current_step < self.warmup_steps:
            # Linear warm-up
            progress = self.current_step / self.warmup_steps
            new_lr = self.min_lr + (self.max_lr - self.min_lr) * progress
        elif self.current_step < self.cos_anneal_stage:
            # Cosine annealing decay
            self.cosine_scheduler.step()
            new_lr = self.cosine_scheduler.get_last_lr()[0]
        else:
            # Final decay to stabilize learning
            progress = (self.current_step - self.cos_anneal_stage) / (self.total_steps - self.cos_anneal_stage)
            new_lr = self.fin_mid_lr + (self.final_lr - self.fin_mid_lr) * progress

        # Apply new learning rate
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr

    def state_dict(self):
        return {
            'current_step': self.current_step
        }

    def load_state_dict(self, state_dict):
        self.current_step = state_dict.get('current_step', 0)            

    def get_last_lr(self):
        return [param_group['lr'] for param_group in self.optimizer.param_groups]