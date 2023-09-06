import re
import warnings

from keras_core import backend
from keras_core import initializers
from keras_core import ops
from keras_core.optimizers.schedules import learning_rate_schedule
from keras_core.saving import serialization_lib
from keras_core.utils import tracking
from keras_core.utils.naming import auto_name


class BaseOptimizer:
    def __init__(
        self,
        learning_rate,
        weight_decay=None,
        clipnorm=None,
        clipvalue=None,
        global_clipnorm=None,
        use_ema=False,
        ema_momentum=0.99,
        ema_overwrite_frequency=None,
        use_loss_scale=False,
        loss_scale_dynamic=True,
        loss_scale_factor=2**15,
        loss_scale_growth_steps=2000,
        name=None,
        **kwargs,
    ):
        self._lock = False

        if kwargs.pop("decay", None) is not None:
            warnings.warn(
                "Argument `decay` is no longer supported and will be ignored."
            )
        if kwargs:
            raise ValueError(f"Argument(s) not recognized: {kwargs}")

        if name is None:
            name = auto_name(self.__class__.__name__)
        self.name = name
        self.weight_decay = weight_decay
        self.clipnorm = clipnorm
        self.global_clipnorm = global_clipnorm
        self.clipvalue = clipvalue
        self.use_ema = use_ema
        self.use_loss_scale = use_loss_scale
        self.loss_scale_dynamic = loss_scale_dynamic
        self.loss_scale_growth_steps = loss_scale_growth_steps
        self.loss_scale_multiplier = 2.0  # TODO: should this be configurable?

        if use_ema:
            # Verify the arguments related to EMA.
            if ema_momentum > 1 or ema_momentum < 0:
                raise ValueError(
                    "`ema_momentum` must be in the range [0, 1]. "
                    f"Received: ema_momentum={ema_momentum}"
                )
            if ema_overwrite_frequency and (
                not isinstance(ema_overwrite_frequency, int)
                or ema_overwrite_frequency < 1
            ):
                raise ValueError(
                    "`ema_overwrite_frequency` must be an integer >= 1 or "
                    "None. Received: ema_overwrite_frequency="
                    f"{ema_overwrite_frequency}"
                )
        self.ema_momentum = ema_momentum
        self.ema_overwrite_frequency = ema_overwrite_frequency

        if self.clipnorm is not None and self.global_clipnorm is not None:
            raise ValueError(
                "Only one of `clipnorm` and `global_clipnorm` can "
                f"be set. Received: clipnorm={self.clipnorm}, "
                f"global_clipnorm={self.global_clipnorm}"
            )

        if self.use_loss_scale:
            # Verify the arguments related to loss scaling.
            if not isinstance(loss_scale_factor, (int, float)):
                raise ValueError(
                    "Argument `loss_scale_factor` should be float. "
                    f"Received: loss_scale_factor={loss_scale_factor}."
                )
        self.built = False

        # Set up variable tracking.
        self._variables = []
        self._trainable_variables = []
        self._tracker = tracking.Tracker(
            {
                "variables": (
                    lambda x: isinstance(x, backend.Variable),
                    self._variables,
                ),
            }
        )
        self._trainable_variables_indices = {}

        # Create iteration variable
        # Note: dtype="int" will resolve to int32 in JAX
        # (since int64 is disallowed in JAX) and to int64 in TF.
        with backend.name_scope(self.name, caller=self):
            iterations = backend.Variable(
                0, name="iteration", dtype="int", trainable=False
            )
        self._track_variable(iterations)
        self.iterations = iterations

        # Create learning rate (schedule or variable)
        if isinstance(
            learning_rate, learning_rate_schedule.LearningRateSchedule
        ):
            self._learning_rate = learning_rate
        elif callable(learning_rate):
            self._learning_rate = learning_rate
        else:
            if not isinstance(learning_rate, float):
                raise ValueError(
                    "Argument `learning_rate` should be float, or an instance "
                    "of LearningRateSchedule, or a callable "
                    "(that takes in the current iteration value "
                    "and returns the corresponding learning rate value). "
                    f"Received instead: learning_rate={learning_rate}"
                )
            with backend.name_scope(self.name, caller=self):
                learning_rate = backend.Variable(
                    learning_rate,
                    name="learning_rate",
                    dtype=backend.floatx(),
                    trainable=False,
                )
            self._track_variable(learning_rate)
            self._learning_rate = learning_rate

        # Create loss_scale_factor for mixed precision training
        with backend.name_scope(self.name, caller=self):
            loss_scale_factor = backend.Variable(
                loss_scale_factor,
                name="loss_scale_factor",
                dtype=backend.floatx(),
                trainable=False,
            )
        self._track_variable(loss_scale_factor)
        self._loss_scale_factor = loss_scale_factor

    def _track_variable(self, variable):
        self._tracker.add_to_store("variables", variable)

    @tracking.no_automatic_dependency_tracking
    def build(self, variables):
        if self.use_ema:
            self._model_variables_moving_average = []
            self._ema_vars_initialized = False
        for i, variable in enumerate(variables):
            self._trainable_variables_indices[self._var_key(variable)] = i
            if self.use_ema:
                self._model_variables_moving_average.append(
                    self.add_variable_from_reference(
                        variable,
                        "average",
                    )
                )
        if self.use_loss_scale and self.loss_scale_dynamic:
            with backend.name_scope(self.name, caller=self):
                loss_scale_good_steps = backend.Variable(
                    0,
                    name="loss_scale_good_steps",
                    dtype="int",
                    trainable=False,
                )
            self._track_variable(loss_scale_good_steps)
            self._loss_scale_good_steps = loss_scale_good_steps

        self._trainable_variables = variables[:]
        self.built = True

    def _var_key(self, variable):
        # Helper function to get a stable ID and the variable instance mapping.
        return id(variable)

    @property
    def variables(self):
        return self._variables[:]

    def _get_variable_index(self, variable):
        return self._trainable_variables_indices[self._var_key(variable)]

    def add_variable(
        self,
        shape,
        initializer,
        dtype=None,
        name=None,
    ):
        self._check_super_called()
        initializer = initializers.get(initializer)
        with backend.name_scope(self.name, caller=self):
            variable = backend.Variable(
                initializer=initializer,
                shape=shape,
                dtype=dtype,
                trainable=False,
                name=name,
            )
        self._track_variable(variable)
        return variable

    def add_variable_from_reference(self, reference_variable, name=None):
        """Add an all-zeros variable with the shape and dtype of a reference
        variable.
        """
        initializer = initializers.Zeros()
        name = name or "var"
        if hasattr(reference_variable, "path"):
            name = reference_variable.path.replace("/", "_") + "_" + name
        else:
            name = reference_variable.name + "_" + name
        return self.add_variable(
            shape=reference_variable.shape,
            initializer=initializer,
            dtype=reference_variable.dtype,
            name=name,
        )

    def _check_variables_are_known(self, variables):
        for v in variables:
            if self._var_key(v) not in self._trainable_variables_indices:
                raise ValueError(
                    f"Unknown variable: {v}. This optimizer can only "
                    "be called for the variables it was originally built with. "
                    "When working with a new set of variables, you should "
                    "recreate a new optimizer instance."
                )

    def update_step(self, gradient, variable, learning_rate):
        raise NotImplementedError

    def apply_gradients(self, grads_and_vars):
        grads, trainable_variables = zip(*grads_and_vars)
        if self.use_loss_scale:
            grads = self.unscale_gradients(grads)
            should_apply = self.update_loss_scale_factor(grads)
            return ops.cond(
                should_apply,
                lambda: self.apply(grads, trainable_variables),
                lambda: self.iterations,
            )
        else:
            return self.apply(grads, trainable_variables)

    def apply(self, grads, trainable_variables=None):
        """
        `grads` should be a list of gradient tensors
        with 1:1 mapping to the list of variables the optimizer was built with.

        `variables` can be provided on the first call to build the optimizer.
        """
        if len(grads) == 0:
            # It is possible that the grad is empty. In this case,
            # `apply_gradients` is a no-op.
            return

        if trainable_variables is None:
            if not self.built:
                raise ValueError(
                    "When passing `grads` without `variables`, the optimizer "
                    "must already be built on a list of variables. "
                    "Call `optimizer.build(trainable_variables)` first. "
                )
            if len(grads) != len(self._trainable_variables_indices):
                raise ValueError(
                    "When passing `grads` as a list of gradient tensors, the "
                    f"gradients must match `optimizer.variables` one-to-on. "
                    f"Received a list of {len(grads)} gradients, but the "
                    f"optimizer is tracking {len(self._trainable_variables)} "
                    "trainable variables."
                )
            trainable_variables = self._trainable_variables
        else:
            trainable_variables = list(trainable_variables)
            # Optionally build optimizer.
            if not self.built:
                with backend.name_scope(self.name, caller=self):
                    self.build(trainable_variables)
                self.built = True
            self._check_variables_are_known(trainable_variables)

        with backend.name_scope(self.name, caller=self):
            # Filter empty gradients.
            grads, trainable_variables = self._filter_empty_gradients(
                grads, trainable_variables
            )
            if len(list(grads)) == 0:
                return

            # Apply clipping and weight decay.
            grads = self._clip_gradients(grads)
            self._apply_weight_decay(trainable_variables)

            # Apply gradient updates.
            self._internal_apply_gradients(
                list(zip(grads, trainable_variables))
            )

            # Apply variable constraints after applying gradients.
            for variable in trainable_variables:
                if getattr(variable, "constraint", None) is not None:
                    variable.assign(variable.constraint(variable))
            return self.iterations

    def _internal_apply_gradients(self, grads_and_vars):
        learning_rate = self._get_current_learning_rate()
        for grad, var in grads_and_vars:
            self.update_step(grad, var, learning_rate)
        self.iterations.assign(self.iterations + 1)

    def stateless_apply(self, optimizer_variables, grads, trainable_variables):
        self._check_super_called()

        if not self.built:
            raise ValueError(
                f"To call `stateless_apply`, {self.__class__.__name__} "
                "must be built (i.e. its variables must have been created). "
                "You can build it via `optimizer.build(trainable_variables)`."
            )
        if len(optimizer_variables) != len(self.variables):
            raise ValueError(
                "Argument `optimizer_variables` must be a list of tensors "
                f"corresponding 1:1 to {self.__class__.__name__}().variables. "
                f"Received list with length {len(optimizer_variables)}, but "
                f"expected {len(self.variables)} variables."
            )
        if len(trainable_variables) != len(self._trainable_variables):
            raise ValueError(
                "Argument `optimizer_variables` must be a list of tensors "
                "corresponding 1:1 to the trainable variables list that "
                "the optimizer was built with. Received "
                f"len(trainable_variables) == {len(trainable_variables)} "
                "whereas the optimizer was built with "
                f"{len(self._trainable_variables)} variables."
            )

        # Gather variable mapping
        mapping = list(
            zip(self._trainable_variables, trainable_variables)
        ) + list(zip(self.variables, optimizer_variables))

        # Call in stateless scope
        with backend.StatelessScope(state_mapping=mapping) as scope:
            self.apply(grads)

        # Gather updated variables
        trainable_variables = []
        for v in self._trainable_variables:
            new_v = scope.get_current_value(v)
            if new_v is not None:
                trainable_variables.append(new_v)
            else:
                trainable_variables.append(v)
        optimizer_variables = []
        for v in self.variables:
            new_v = scope.get_current_value(v)
            if new_v is not None:
                optimizer_variables.append(new_v)
            else:
                optimizer_variables.append(v)
        return trainable_variables, optimizer_variables

    def scale_loss(self, loss):
        if self.use_loss_scale:
            return loss * ops.cast(self._loss_scale_factor, loss.dtype)
        else:
            return loss

    def unscale_gradients(self, grads):
        return [
            g * ops.cast(1.0 / self._loss_scale_factor, g.dtype)
            if g is not None
            else None
            for g in grads
        ]

    def update_loss_scale_factor(self, grads):
        is_finite_per_grad = [
            ops.all(ops.isfinite(g)) for g in grads if g is not None
        ]
        is_all_finite = ops.all(is_finite_per_grad)
        if not self.loss_scale_dynamic:
            return is_all_finite

        def update_if_all_finite_grads():
            def exceed_good_step():
                self._loss_scale_good_steps.assign(0)
                return self._loss_scale_factor * self.loss_scale_multiplier

            def not_exceed_good_step():
                self._loss_scale_good_steps.assign_add(1)
                return self._loss_scale_factor

            return ops.cond(
                self._loss_scale_good_steps > self.loss_scale_growth_steps,
                exceed_good_step,
                not_exceed_good_step,
            )

        def update_if_any_infinite_grads():
            self._loss_scale_good_steps.assign(0)
            return ops.maximum(
                self._loss_scale_factor / self.loss_scale_multiplier, 1.0
            )

        new_loss_scale_factor = ops.cond(
            is_all_finite,
            update_if_all_finite_grads,
            update_if_any_infinite_grads,
        )
        self._loss_scale_factor.assign(new_loss_scale_factor)
        return is_all_finite

    @property
    def learning_rate(self):
        return self._get_current_learning_rate()

    @learning_rate.setter
    def learning_rate(self, learning_rate):
        if isinstance(
            learning_rate, learning_rate_schedule.LearningRateSchedule
        ):
            self._learning_rate = learning_rate
        elif callable(learning_rate):
            self._learning_rate = learning_rate
        else:
            if isinstance(
                self._learning_rate, learning_rate_schedule.LearningRateSchedule
            ):
                raise TypeError(
                    "This optimizer was created with a `LearningRateSchedule`"
                    " object as its `learning_rate` constructor argument, "
                    "hence its learning rate is not settable. If you need the"
                    " learning rate to be settable, you should instantiate "
                    "the optimizer with a float `learning_rate` argument."
                )
            self._learning_rate.assign(learning_rate)

    @property
    def loss_scale_factor(self):
        return self._loss_scale_factor

    @loss_scale_factor.setter
    def loss_scale_factor(self, loss_scale_factor):
        self._loss_scale_factor.assign(loss_scale_factor)

    def set_weights(self, weights):
        """Set the weights of the optimizer."""
        if not self.built:
            raise ValueError(
                "You are calling `set_weights()` on an optimizer that has not "
                "yet been built. Please call "
                "`optimizer.build(trainable_variables)` to create the "
                "optimizer weights before calling `set_weights()`."
            )
        for variable, weight in zip(self._variables, weights):
            if variable.shape != weight.shape:
                raise ValueError(
                    f"Optimizer variable {self._var_key(variable)} has shape "
                    f"{str(variable.shape)} not compatible with provided "
                    f"weight shape {str(weight.shape)}."
                )
            variable.assign(weight)

    def save_own_variables(self, store):
        """Get the state of this optimizer object."""
        for i, variable in enumerate(self.variables):
            store[str(i)] = variable.numpy()

    def load_own_variables(self, store):
        """Set the state of this optimizer object."""
        if len(store.keys()) != len(self.variables):
            msg = (
                f"Skipping variable loading for optimizer '{self.name}', "
                f"because it has {len(self.variables)} variables whereas "
                f"the saved optimizer has {len(store.keys())} variables. "
            )
            if len(self.variables) == 0:
                msg += (
                    "This is likely because the optimizer has not been "
                    "called/built yet."
                )
            warnings.warn(msg, stacklevel=2)
            return
        for i, variable in enumerate(self.variables):
            variable.assign(store[str(i)])

    def _get_current_learning_rate(self):
        if isinstance(
            self._learning_rate, learning_rate_schedule.LearningRateSchedule
        ):
            return self._learning_rate(self.iterations)
        elif callable(self._learning_rate):
            return self._learning_rate(self.iterations)
        return self._learning_rate

    def _filter_empty_gradients(self, grads, vars):
        for grad in grads:
            if grad is None:
                # Filtering is required.
                filtered = [
                    (g, v) for g, v in zip(grads, vars) if g is not None
                ]
                if not filtered:
                    raise ValueError("No gradients provided for any variable.")
                if len(filtered) < len(grads):
                    missing_grad_vars = [
                        v for g, v in zip(grads, vars) if g is None
                    ]
                    warnings.warn(
                        "Gradients do not exist for variables "
                        f"{[v.name for v in missing_grad_vars]} when "
                        "minimizing the loss. If using `model.compile()`, "
                        "did you forget to provide a `loss` argument?"
                    )
                return zip(*filtered)
        return grads, vars

    def _clip_gradients(self, grads):
        if self.clipnorm and self.clipnorm > 0:
            clipped_grads = []
            for g in grads:
                if g is None:
                    clipped_grads.append(g)
                else:
                    clipped_grads.append(clip_by_norm(g, self.clipnorm))
            return clipped_grads

        if self.global_clipnorm and self.global_clipnorm > 0:
            return clip_by_global_norm(grads, self.global_clipnorm)

        if self.clipvalue and self.clipvalue > 0:
            clipped_grads = []
            for g in grads:
                if g is None:
                    clipped_grads.append(g)
                else:
                    clipped_grads.append(
                        ops.clip(g, -self.clipvalue, self.clipvalue)
                    )
            return clipped_grads
        return grads

    def exclude_from_weight_decay(self, var_list=None, var_names=None):
        """Exclude variables from weight decay.

        This method must be called before the optimizer's `build` method is
        called. You can set specific variables to exclude out, or set a list of
        strings as the anchor words, if any of which appear in a variable's
        name, then the variable is excluded.

        Args:
            var_list: A list of `tf.Variable`s to exclude from weight decay.
            var_names: A list of strings. If any string in `var_names` appear
                in the model variable's name, then this model variable is
                excluded from weight decay. For example, `var_names=['bias']`
                excludes all bias variables from weight decay.
        """
        if hasattr(self, "_built") and self._built:
            raise ValueError(
                "`exclude_from_weight_decay()` can only be configued before "
                "the optimizer is built."
            )

        if var_list:
            self._exclude_from_weight_decay = [
                self._var_key(variable) for variable in var_list
            ]
        else:
            self._exclude_from_weight_decay = []
        self._exclude_from_weight_decay_names = var_names or []

    def _use_weight_decay(self, variable):
        exclude_from_weight_decay = getattr(
            self, "_exclude_from_weight_decay", []
        )
        exclude_from_weight_decay_names = getattr(
            self, "_exclude_from_weight_decay_names", []
        )
        variable_id = self._var_key(variable)
        for exclude_id in exclude_from_weight_decay:
            if variable_id == exclude_id:
                return False
        for name in exclude_from_weight_decay_names:
            if re.search(name, variable.name) is not None:
                return False
        return True

    def _apply_weight_decay(self, variables):
        if self.weight_decay is None:
            return
        for variable in variables:
            if self._use_weight_decay(variable):
                lr = ops.cast(self._get_current_learning_rate(), variable.dtype)
                wd = ops.cast(self.weight_decay, variable.dtype)
                variable.assign(variable - variable * wd * lr)

    def _check_super_called(self):
        if not hasattr(self, "_lock"):
            raise RuntimeError(
                f"In optimizer '{self.__class__.__name__}', you forgot to call "
                "`super().__init__()` as the first statement "
                "in the `__init__()` method. "
                "Go add it!"
            )

    def _update_model_variables_moving_average(self, var_list):
        """Update the stored moving average using the latest value."""
        if self.use_ema:
            for var, average in zip(
                var_list, self._model_variables_moving_average
            ):
                if self._ema_vars_initialized:
                    average.assign(
                        self.ema_momentum * average
                        + (1 - self.ema_momentum) * var
                    )
                else:
                    average.assign(var)
            self._ema_vars_initialized = True

    def _overwrite_model_variables_with_average_value(self, var_list):
        """Overwrite model variables with its moving average."""
        if len(var_list) != len(self._model_variables_moving_average):
            raise ValueError(
                f"The length of model variables ({len(var_list)}) to "
                "override does not match the length of model variables "
                "stored in the optimizer "
                f"({len(self._model_variables_moving_average)}). Please "
                "check if the optimizer was called on your model."
            )
        self._overwrite_model_variables_with_average_value_helper(var_list)

    def _overwrite_model_variables_with_average_value_helper(self, var_list):
        """Helper function that overwrites model variables."""
        for var, average_var in zip(
            var_list, self._model_variables_moving_average
        ):
            var.assign(average_var)

    def finalize_variable_values(self, var_list):
        """Set the final value of model's trainable variables.

        Sometimes there are some extra steps before ending the variable updates,
        such as overriding the model variables with its average value.

        Args:
          var_list: list of model variables.
        """
        if self.use_ema:
            # If the optimizer uses EMA, then when finalizing, we replace the
            # model variable value with its moving average stored inside
            # optimizer.
            self._overwrite_model_variables_with_average_value(var_list)

    def get_config(self):
        """Returns the config of the optimizer.

        An optimizer config is a Python dictionary (serializable)
        containing the configuration of an optimizer.
        The same optimizer can be reinstantiated later
        (without any saved state) from this configuration.

        Subclass optimizer should override this method to include other
        hyperparameters.

        Returns:
            Python dictionary.
        """

        if isinstance(
            self._learning_rate, learning_rate_schedule.LearningRateSchedule
        ):
            learning_rate = learning_rate_schedule.serialize(
                self._learning_rate
            )
        elif isinstance(self._learning_rate, backend.Variable):
            learning_rate = float(self._learning_rate.numpy())
        elif ops.is_tensor(self._learning_rate):
            learning_rate = float(self._learning_rate)
        elif callable(self._learning_rate):
            learning_rate = serialization_lib.serialize_keras_object(
                self._learning_rate
            )

        if isinstance(self._loss_scale_factor, backend.Variable):
            loss_scale_factor = float(self._loss_scale_factor.numpy())
        elif ops.is_tensor(self._loss_scale_factor):
            loss_scale_factor = float(self._loss_scale_factor)

        config = {
            "name": self.name,
            "learning_rate": learning_rate,
            "weight_decay": self.weight_decay,
            "clipnorm": self.clipnorm,
            "global_clipnorm": self.global_clipnorm,
            "clipvalue": self.clipvalue,
            "use_ema": self.use_ema,
            "ema_momentum": self.ema_momentum,
            "ema_overwrite_frequency": self.ema_overwrite_frequency,
            "use_loss_scale": self.use_loss_scale,
            "loss_scale_dynamic": self.loss_scale_dynamic,
            "loss_scale_growth_steps": self.loss_scale_growth_steps,
            "loss_scale_factor": loss_scale_factor,
        }
        return config

    @classmethod
    def from_config(cls, config, custom_objects=None):
        """Creates an optimizer from its config.

        This method is the reverse of `get_config`, capable of instantiating the
        same optimizer from the config dictionary.

        Args:
            config: A Python dictionary, typically the output of get_config.
            custom_objects: A Python dictionary mapping names to additional
              user-defined Python objects needed to recreate this optimizer.

        Returns:
            An optimizer instance.
        """
        if "learning_rate" in config:
            if isinstance(config["learning_rate"], dict):
                config[
                    "learning_rate"
                ] = serialization_lib.deserialize_keras_object(
                    config["learning_rate"], custom_objects=custom_objects
                )
        return cls(**config)

    def __setattr__(self, name, value):
        # Prevent users from attaching state to the
        # layer before `super()` is called -- since that
        # state would silently not be tracked.
        if name != "_lock":
            self._check_super_called()
        # Track Variables.
        if hasattr(self, "_tracker"):
            value = self._tracker.track(value)
        return super().__setattr__(name, value)


base_optimizer_keyword_args = """name: String. The name to use
          for momentum accumulator weights created by
          the optimizer.
      weight_decay: Float, defaults to None. If set, weight decay is applied.
      clipnorm: Float. If set, the gradient of each weight is individually
          clipped so that its norm is no higher than this value.
      clipvalue: Float. If set, the gradient of each weight is clipped to be no
          higher than this value.
      global_clipnorm: Float. If set, the gradient of all weights is clipped so
          that their global norm is no higher than this value.
      use_ema: Boolean, defaults to False. If True, exponential moving average
          (EMA) is applied. EMA consists of computing an exponential moving
          average of the weights of the model (as the weight values change after
          each training batch), and periodically overwriting the weights with
          their moving average.
      ema_momentum: Float, defaults to 0.99. Only used if `use_ema=True`.
          This is the momentum to use when computing
          the EMA of the model's weights:
          `new_average = ema_momentum * old_average + (1 - ema_momentum) *
          current_variable_value`.
      ema_overwrite_frequency: Int or None, defaults to None. Only used if
          `use_ema=True`. Every `ema_overwrite_frequency` steps of iterations,
          we overwrite the model variable by its moving average.
          If None, the optimizer
          does not overwrite model variables in the middle of training, and you
          need to explicitly overwrite the variables at the end of training
          by calling `optimizer.finalize_variable_values()`
          (which updates the model
          variables in-place). When using the built-in `fit()` training loop,
          this happens automatically after the last epoch,
          and you don't need to do anything."""


def clip_by_norm(values, clip_norm, axes=None):
    # Calculate L2-norm, clip elements by ratio of clip_norm to L2-norm
    l2sum = ops.sum(values * values, axes, keepdims=True)
    pred = l2sum > 0
    # Two-tap tf.where trick to bypass NaN gradients
    l2sum_safe = ops.where(pred, l2sum, ops.ones_like(l2sum))
    l2norm = ops.where(pred, ops.sqrt(l2sum_safe), l2sum)
    intermediate = values * clip_norm
    values_clip = ops.convert_to_tensor(intermediate) / ops.maximum(
        l2norm, clip_norm
    )
    return values_clip


def global_norm(value_list):
    """Computes the global norm of multiple tensors."""
    squared_norms = []
    for v in value_list:
        if v is not None:
            squared_norms.append(ops.sum(ops.square(v)))
    squared_norm = ops.sum(ops.stack(squared_norms))
    return ops.sqrt(squared_norm)


def clip_by_global_norm(value_list, clip_norm):
    use_norm = global_norm(value_list)
    # Calculate L2-norm, clip elements by ratio of clip_norm to L2-norm
    scale_for_finite = clip_norm * ops.minimum(1.0 / use_norm, 1.0 / clip_norm)
    # If use_norm is any finite number, this is a no-op. For inf/-inf/NaN,
    # this will make scale NaN.
    scale = scale_for_finite + (use_norm - use_norm)
    values_clipped = []
    for v in value_list:
        if v is None:
            values_clipped.append(None)
        else:
            values_clipped.append(v * scale)
    return values_clipped
