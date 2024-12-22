import torch
import torch.optim as optim
from torch.nn import functional as F
from torch.utils.data import DataLoader

from models.timesnet.lora_timesnet import LoRATimesNet


class LoRAFineTuner:
    def __init__(
        self,
        base_model,
        rank=4,
        alpha=1.0,
        learning_rate=1e-3,
        batch_size=32,
        num_epochs=10,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lora_model = LoRATimesNet(base_model, rank, alpha).to(self.device)
        self.optimizer = optim.AdamW(
            self.lora_model.get_lora_parameters(), lr=learning_rate
        )
        self.batch_size = batch_size
        self.num_epochs = num_epochs

    def train_on_synthetic(self, synthetic_generator, num_samples=1000):
        """Fine-tune on synthetic data"""
        # Generate synthetic data
        synthetic_data = synthetic_generator.generate()

        # Create dataloader
        train_loader = DataLoader(
            synthetic_data, batch_size=self.batch_size, shuffle=True
        )

        # Training loop
        self.lora_model.train()
        for epoch in range(self.num_epochs):
            total_loss = 0
            for batch_data in train_loader:
                self.optimizer.zero_grad()

                # Forward pass
                outputs = self.lora_model(batch_data)
                loss = self._compute_loss(outputs, batch_data)

                # Backward pass
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)
            print(f"Epoch {epoch+1}/{self.num_epochs}, Loss: {avg_loss:.4f}")

    def _compute_loss(self, outputs, targets):
        """Compute appropriate loss based on task"""
        task_name = self.lora_model.base_model.task_name

        if task_name in ["long_term_forecast", "short_term_forecast"]:
            return F.mse_loss(outputs, targets)
        elif task_name == "classification":
            return F.cross_entropy(outputs, targets)
        else:
            return F.mse_loss(outputs, targets)

    def save_lora_weights(self, path):
        """Save LoRA weights"""
        lora_state_dict = {
            name: param
            for name, param in self.lora_model.named_parameters()
            if "lora_" in name
        }
        torch.save(lora_state_dict, path)

    def load_lora_weights(self, path):
        """Load LoRA weights"""
        lora_state_dict = torch.load(path)
        self.lora_model.load_state_dict(lora_state_dict, strict=False)
