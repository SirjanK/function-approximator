import os
import shutil
import torch
from typing import Callable, Any, List, Dict
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from abc import ABC, abstractmethod
from overrides import overrides
from approximator.fitter import Fitter
from approximator.input_sampler import InputSampler


class ModelBasedFitter(Fitter, ABC):
    """
    The ModelBasedFitter is a Fitter that employs a pytorch model to approximate a function.
    """

    def __init__(self, fn: Callable, input_sampler: InputSampler, name: str, train_batch_size: int = 32, eval_batch_size: int = 1000, \
                 num_steps: int = 10000, eval_steps: int = 100):
        """
        Initializes the fitter

        :param fn: the function to approximate
        :param input_sampler: the input sampler
        :param name: the name of the fitter (used for logging)
        :param train_batch_size: the batch size for training
        :param eval_batch_size: the batch size for evaluation
        :param num_steps: the number of training steps
        :param eval_steps: the number of steps between evaluations
        """

        self._name = name
        self._train_batch_size = train_batch_size
        self._eval_batch_size = eval_batch_size
        self._num_steps = num_steps
        self._eval_steps = eval_steps

        self._loss_fn = self.get_loss_fn()
        self._model = self.construct_model()
        self._optimizer = self.get_optimizer()

        super().__init__(fn, input_sampler)
        self.init_other_state_variables()
    
    @abstractmethod
    def get_loss_fn(self) -> Callable:
        """
        Get the loss function for the model
        :return: the loss function
        """

        raise NotImplementedError()
    
    @abstractmethod
    def construct_model(self) -> nn.Module:
        """
        Construct the model
        :return: the constructed model (with weights initialized)
        """

        raise NotImplementedError()
    
    @abstractmethod
    def get_optimizer(self) -> torch.optim.Optimizer:
        """
        Get the optimizer for the model
        :return: the optimizer
        """

        raise NotImplementedError()
    
    @abstractmethod
    def convert_input_to_tensor(self, *args) -> torch.Tensor:
        """
        Convert the input to a tensor
        :param args: the input arguments
        :return: the tensor
        """

        raise NotImplementedError()
    
    @abstractmethod
    def convert_output_to_tensor(self, output: Any) -> torch.Tensor:
        """
        Convert the output to a tensor
        :param output: the output target
        :return: the tensor
        """

        raise NotImplementedError()
    
    @abstractmethod
    def convert_model_output_to_function_output(self, model_output: torch.Tensor) -> Any:
        """
        Convert the model output to a function output. This acts as the inverse of convert_output_to_tensor
        :param model_output: the model output
        :return: the function output
        """

        raise NotImplementedError()
    
    def compute_metrics(self, model_output: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
        """
        Compute metrics for the model output and target
        :param model_output: the model output
        :param target: the target
        :return: metrics dictionary mapping from metric name to value
        """

        # subclasses can override this for custom metrics
        return {}
    
    def init_other_state_variables(self) -> None:
        """
        Helper for subclasses to initialize any other state variables that can be used during computation
        """

        pass

    @overrides
    def fit(self) -> Callable:
        # initialize a writer for metrics
        # remove the runs/ directory if it exists
        if os.path.exists(f"runs/{self._name}"):
            shutil.rmtree(f"runs/{self._name}")
        self._writer = SummaryWriter(f"runs/{self._name}/logs")

        for step_idx in range(self._num_steps):
            self._step()

            if step_idx % self._eval_steps == 0:
                self._evaluate_and_log(step_idx)

        def fitted_fn(*args) -> Any:
            model_out = self._model(self.convert_input_to_tensor(args))
            return self.convert_model_output_to_function_output(model_out)
        
        return fitted_fn
    
    def _forward(self, batch: List[List[float]]) -> torch.Tensor:
        """
        Take a forward pass through the model based on an input batch

        :param batch: input batch data
        :return: the output tensor for the forward pass
        """

        inputs = torch.stack([self.convert_input_to_tensor(*inp) for inp in batch])
        return self._model(inputs)
    
    def _step(self) -> None:
        """
        Perform one optimization step
        """

        # zero the gradients
        self._optimizer.zero_grad()

        # sample a batch of inputs and outputs
        inputs, targets = self._generate_batch(self._train_batch_size)

        # forward pass
        model_outputs = self._forward(inputs)

        # compute loss tensor
        targets = self.convert_output_to_tensor(targets)
        loss = self._loss_fn(model_outputs, targets)

        # backward pass
        loss.backward()

        # optimization step
        self._optimizer.step()

    def _evaluate_and_log(self, step_idx: int) -> None:
        """
        Evaluate and log metrics for the given step index for the model so far

        :param step_idx: the step index
        """

        # sample a batch of inputs and outputs
        inputs, targets = self._generate_batch(self._eval_batch_size)

        # forward pass
        model_outputs = self._forward(inputs)

        # compute loss tensor
        targets = self.convert_output_to_tensor(targets)
        loss = self._loss_fn(model_outputs, targets)

        # compute other metrics
        metrics = self.compute_metrics(model_outputs, targets)

        # log the loss and metrics
        self._writer.add_scalar("loss", loss, step_idx)
        for metric_name, metric_value in metrics.items():
            self._writer.add_scalar(metric_name, metric_value, step_idx)
