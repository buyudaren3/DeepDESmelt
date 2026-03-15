import numpy as np
import torch
from torch import nn
from torch.nn import init
import logging
from torch.amp import autocast


class GaussianFourierProjection(nn.Module):
    def __init__(self, embed_dim, scale):
        super(GaussianFourierProjection, self).__init__()
        self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)

    def forward(self, x):
        x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
        x_proj = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
        return x_proj

class FiLMratio(nn.Module):
    def __init__(self, film_hidden_dim, activation, norm, dropout):
        super(FiLMratio, self).__init__()
        self.film_hidden_dim = film_hidden_dim

        # FiLM processing
        film_generator = []
        film_generator.append(nn.Linear(1536, film_hidden_dim))
        self.initialize_weights(film_generator[-1], activation)
        self.add_norm(norm, film_generator, film_hidden_dim)
        self.add_activation(activation, film_generator)
        film_generator.append(nn.Dropout(p=dropout))

        if activation == 'GLU':
            film_hidden_dim = int(film_hidden_dim/2)

        film_generator.append(nn.Linear(film_hidden_dim, film_hidden_dim))
        self.initialize_weights(film_generator[-1], activation)
        self.add_norm(norm, film_generator, film_hidden_dim)
        self.add_activation(activation, film_generator)
        film_generator.append(nn.Dropout(p=dropout))

        if activation == 'GLU':
            film_hidden_dim = int(film_hidden_dim/2)

        film_generator.append(nn.Linear(film_hidden_dim, 4 * self.film_hidden_dim))
        self.initialize_weights(film_generator[-1], None)

        self.film_ratio = nn.Sequential(*film_generator)

    def forward(self, x):
        return self.film_ratio(x)

    def add_activation(self, activation, layers):
        if activation == 'ReLU':
            layers.append(nn.ReLU())
        elif activation == 'LeakyReLU':
            layers.append(nn.LeakyReLU())
        elif activation == 'Sigmoid':
            layers.append(nn.Sigmoid())
        elif activation == 'GELU':
            layers.append(nn.GELU())
        elif activation == 'ELU':
            layers.append(nn.ELU())
        elif activation == 'PReLU':
            layers.append(nn.PReLU())
        elif activation == 'GLU':
            layers.append(nn.GLU())
        else:
            raise ValueError('activation not supported')

    def initialize_weights(self, module, activation):
        if isinstance(module, nn.Linear):
            if activation == 'ReLU' or activation is None:  # Output layer or after ReLU, do not use Kaiming initialization
                init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                logging.info(f'Initialize {module} with ReLU')
            elif activation == 'LeakyReLU':
                init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='leaky_relu')
                logging.info(f'Initialize {module} with LeakyReLU')
            else:
                init.xavier_normal_(module.weight)
                logging.info(f'Initialize {module} with {activation}')
            if module.bias is not None:
                init.constant_(module.bias, 0)
                logging.info(f'Initialize {module} bias with 0')

    def add_norm(self, norm, layers, current_dim):
        if norm == 'BatchNorm1d':
            layers.append(nn.BatchNorm1d(current_dim))
        elif norm == 'LayerNorm':
            layers.append(nn.LayerNorm(current_dim))
        elif norm == 'none':
            pass
        else:
            raise ValueError(f'Invalid {norm} type Choose from "batch_norm" or "layer_norm"')

class GatingNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, num_experts, dropout, num_hidden_layers, norm, activation):
        super(GatingNetwork, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        layers = []
        # Build hidden layers
        layers.append(nn.Linear(self.input_size, hidden_size))
        self.initialize_weights(layers[-1], activation)
        self.add_norm(norm, layers, hidden_size)
        self.add_activation(activation, layers)

        layers.append(nn.Dropout(p=dropout))

        for _ in range(num_hidden_layers - 1):
            if activation == 'GLU':
                hidden_size = int(hidden_size/2)
                self.hidden_size = hidden_size
            layers.append(nn.Linear(hidden_size, hidden_size))
            self.initialize_weights(layers[-1], activation)
            self.add_norm(norm, layers, hidden_size)
            self.add_activation(activation, layers)
            layers.append(nn.Dropout(p=dropout))

        if activation == 'GLU':
            hidden_size = int(hidden_size/2)

        # Output layer
        layers.append(nn.Linear(hidden_size, num_experts))
        self.initialize_weights(layers[-1], None)

        self.net = nn.Sequential(*layers)
        self.softmax = nn.Softmax(dim=1)
        
        # Manually get the last layer
        output_layer = self.net[-1]
        
        # Force initialize weights and biases to zero
        # This overrides the Kaiming initialization in initialize_weights function
        torch.nn.init.zeros_(output_layer.weight)
        if output_layer.bias is not None:
            torch.nn.init.zeros_(output_layer.bias)

    def forward(self, x):
        logits = self.net(x)  # Get raw logits
        with autocast('cuda', enabled=False):
            logits_float32 = logits.float()
            output = self.softmax(logits_float32)
        return output
    
    def add_activation(self, activation, layers):
        if activation == 'ReLU':
            layers.append(nn.ReLU())
        elif activation == 'LeakyReLU':
            layers.append(nn.LeakyReLU())
        elif activation == 'Sigmoid':
            layers.append(nn.Sigmoid())
        elif activation == 'GELU':
            layers.append(nn.GELU())
        elif activation == 'ELU':
            layers.append(nn.ELU())
        elif activation == 'PReLU':
            layers.append(nn.PReLU())
        elif activation == 'GLU':
            layers.append(nn.GLU())
        else:
            raise ValueError('activation not supported')

    def initialize_weights(self, module, activation):
        if isinstance(module, nn.Linear):
            if activation == 'ReLU' or activation is None:  # Output layer or after ReLU, do not use Kaiming initialization
                init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                logging.info(f'Initialize {module} with ReLU')
            elif activation == 'LeakyReLU':
                init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='leaky_relu')
                logging.info(f'Initialize {module} with LeakyReLU')
            else:
                init.xavier_normal_(module.weight)
                logging.info(f'Initialize {module} with {activation}')
            if module.bias is not None:
                init.constant_(module.bias, 0)
                logging.info(f'Initialize {module} bias with 0')

    def add_norm(self, norm, layers, current_dim):
        if norm == 'BatchNorm1d':
            layers.append(nn.BatchNorm1d(current_dim))
        elif norm == 'LayerNorm':
            layers.append(nn.LayerNorm(current_dim))
        elif norm == 'none':
            pass
        else:
            raise ValueError(f'Invalid {norm} type Choose from "batch_norm" or "layer_norm"')

class Expert(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout, num_hidden_layers, norm, activation):
        super(Expert, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        layers = []
        # Build hidden layers
        layers.append(nn.Linear(self.input_size, hidden_size))
        self.initialize_weights(layers[-1], activation)
        self.add_norm(norm, layers, hidden_size)
        self.add_activation(activation, layers)

        layers.append(nn.Dropout(p=dropout))

        for _ in range(num_hidden_layers - 1):
            if activation == 'GLU':
                hidden_size = int(hidden_size/2)
                self.hidden_size = hidden_size
            layers.append(nn.Linear(hidden_size, hidden_size))
            self.initialize_weights(layers[-1], activation)
            self.add_norm(norm, layers, hidden_size)
            self.add_activation(activation, layers)
            layers.append(nn.Dropout(p=dropout))

        if activation == 'GLU':
            hidden_size = int(hidden_size/2)

        layers.append(nn.Linear(hidden_size, output_size))
        self.initialize_weights(layers[-1], None)

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

    def add_activation(self, activation, layers):
        if activation == 'ReLU':
            layers.append(nn.ReLU())
        elif activation == 'LeakyReLU':
            layers.append(nn.LeakyReLU())
        elif activation == 'Sigmoid':
            layers.append(nn.Sigmoid())
        elif activation == 'GELU':
            layers.append(nn.GELU())
        elif activation == 'ELU':
            layers.append(nn.ELU())
        elif activation == 'PReLU':
            layers.append(nn.PReLU())
        elif activation == 'GLU':
            layers.append(nn.GLU())
        else:
            raise ValueError('activation not supported')

    def initialize_weights(self, module, activation):
        if isinstance(module, nn.Linear):
            if activation == 'ReLU' or activation is None:  # Output layer or after ReLU, do not use Kaiming initialization
                init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                logging.info(f'Initialize {module} with ReLU')
            elif activation == 'LeakyReLU':
                init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='leaky_relu')
                logging.info(f'Initialize {module} with LeakyReLU')
            else:
                init.xavier_normal_(module.weight)
                logging.info(f'Initialize {module} with {activation}')
            if module.bias is not None:
                init.constant_(module.bias, 0)
                logging.info(f'Initialize {module} bias with 0')

    def add_norm(self, norm, layers, current_dim):
        if norm == 'BatchNorm1d':
            layers.append(nn.BatchNorm1d(current_dim))
        elif norm == 'LayerNorm':
            layers.append(nn.LayerNorm(current_dim))
        elif norm == 'none':
            pass
        else:
            raise ValueError(f'Invalid {norm} type Choose from "batch_norm" or "layer_norm"')

class MOE(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_experts, dropout, num_hidden_layers, norm, activation, load_balance_alpha=0.01, adaptive_weight=False, target_ratio=0.05):
        super(MOE, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.dropout = dropout
        self.num_hidden_layers = num_hidden_layers
        self.norm = norm
        self.activation = activation
        self.load_balance_alpha = load_balance_alpha
        self.adaptive_weight = adaptive_weight
        self.target_ratio = target_ratio
        
        # Moving average for adaptive weight calculation
        self.register_buffer('main_loss_ema', torch.tensor(1.0))
        self.register_buffer('importance_loss_ema', torch.tensor(0.01))
        self.register_buffer('ema_initialized', torch.tensor(False))  # Flag for initialization status
        self.register_buffer('ema_step_count', torch.tensor(0))  # EMA update count
        self.ema_decay = 0.95
        self.ema_warmup_steps = 100  # Use smaller decay for first 100 steps for faster EMA convergence

        self.ratio_set = GaussianFourierProjection(embed_dim=input_size, scale=10)
        self.FiLM = FiLMratio(film_hidden_dim=input_size, activation=activation, norm=norm, dropout=dropout)
        self.gate = GatingNetwork(input_size=input_size, hidden_size=hidden_size, num_experts=num_experts, dropout=dropout, num_hidden_layers=num_hidden_layers, norm=norm, activation=activation)
        self.experts = nn.ModuleList([Expert(input_size=input_size, hidden_size=hidden_size, output_size=output_size, dropout=dropout, num_hidden_layers=num_hidden_layers, norm=norm, activation=activation) for _ in range(num_experts)])

    def compute_adaptive_weight(self, main_loss, raw_importance_loss):
        if not self.adaptive_weight:
            return self.load_balance_alpha
        
        # Adaptive initialization: use actual values to initialize EMA on first call
        with torch.no_grad():
            if not self.ema_initialized:
                # On first call, directly initialize EMA with current values
                self.main_loss_ema = main_loss.detach().clone()
                self.importance_loss_ema = raw_importance_loss.detach().clone()
                self.ema_initialized = torch.tensor(True)
                print(f"EMA Adaptive Initialization - Main Loss EMA: {self.main_loss_ema.item():.6f}, Importance Loss EMA: {self.importance_loss_ema.item():.6f}")
            else:
                # Subsequent updates use dynamic EMA formula, with smaller decay during warmup
                self.ema_step_count += 1
                
                # Use smaller decay (0.8) for first warmup_steps for faster EMA convergence
                if self.ema_step_count <= self.ema_warmup_steps:
                    current_decay = 0.8  # Smaller decay for faster convergence
                else:
                    current_decay = self.ema_decay  # Standard decay
                    
                self.main_loss_ema = current_decay * self.main_loss_ema + (1 - current_decay) * main_loss.detach()
                self.importance_loss_ema = current_decay * self.importance_loss_ema + (1 - current_decay) * raw_importance_loss.detach()
        
        # Calculate adaptive weight such that alpha * raw_importance_loss / main_loss = target_ratio
        # i.e., alpha = target_ratio * main_loss / raw_importance_loss  
        # Use EMA to avoid single-batch fluctuations
        eps = 1e-8  # Avoid division by zero
        adaptive_alpha = self.target_ratio * self.main_loss_ema / (self.importance_loss_ema + eps)
        
        return adaptive_alpha

    def compute_importance_loss(self, gate_probs, main_loss=None):
        # P_i: Average gating probability of expert i over the entire batch
        # P_i = sum(softmax(logits)_i) / batch_size
        P_i = gate_probs.mean(dim=0)  # Shape: (num_experts,)
        
        # Calculate raw importance loss (unweighted)
        # L_importance = sum_over_experts((P_i)^2)
        # This loss encourages diversity in expert weights, avoiding over-concentration on certain experts
        raw_importance_loss = torch.sum(P_i ** 2)
        
        # Calculate adaptive weight
        if self.adaptive_weight and main_loss is not None:
            adaptive_alpha = self.compute_adaptive_weight(main_loss, raw_importance_loss)
        else:
            adaptive_alpha = self.load_balance_alpha  # Keep parameter name consistent
        
        # Apply weight
        importance_loss = adaptive_alpha * raw_importance_loss
        
        return importance_loss, adaptive_alpha

    def forward(self, x, compute_load_balance_loss=True):
        HBA_emb = x[:, :1536]
        HBD_emb = x[:, 1536:3072]
        ratio = x[:, 3072:]

        ratio_emb = self.ratio_set(ratio).squeeze(1)
        film_params = self.FiLM(ratio_emb)
        gamma_HBA, beta_HBA, gamma_HBD, beta_HBD = torch.chunk(film_params, 4, dim=-1)

        HBA_emb_film = gamma_HBA * HBA_emb + beta_HBA
        HBD_emb_film = gamma_HBD * HBD_emb + beta_HBD

        features = torch.stack([HBA_emb_film, HBD_emb_film, ratio_emb], dim=1)

        aggregate_features = features.mean(dim=1)

        gate = self.gate(aggregate_features)

        experts_outputs = [expert(aggregate_features) for expert in self.experts]
        experts_outputs = torch.stack(experts_outputs, dim=1)

        output = torch.sum(experts_outputs * gate.unsqueeze(-1), dim=1)
        
        # Calculate importance loss (for dense MoE)
        importance_loss = None
        adaptive_alpha = None
        if compute_load_balance_loss and self.training:  # Keep parameter name consistent
            # Pass None as main_loss here because we don't know the main task loss in forward yet
            # The actual main_loss will be passed in the training loop
            importance_loss, adaptive_alpha = self.compute_importance_loss(gate, main_loss=None)
        
        # Return values explanation:
        # output: Final prediction (batch_size, output_size)
        # aggregate_features: Aggregated features (batch_size, input_size)  
        # experts_outputs: Raw predictions from each expert (batch_size, num_experts, output_size)
        # gate: Expert weights/routing probabilities (batch_size, num_experts)
        return output, aggregate_features, experts_outputs, gate, importance_loss, adaptive_alpha

    def recompute_importance_loss_with_main_loss(self, gate_probs, main_loss):
        return self.compute_importance_loss(gate_probs, main_loss)
