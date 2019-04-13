from __future__ import absolute_import, division, print_function

import six
import logging
import os
import json
import numpy as np
import torch
from torch import optim

from madminer.utils.ml.models.maf import ConditionalMaskedAutoregressiveFlow
from madminer.utils.ml.models.maf_mog import ConditionalMixtureMaskedAutoregressiveFlow
from madminer.utils.ml.models.ratio import ParameterizedRatioEstimator, DoublyParameterizedRatioEstimator
from madminer.utils.ml.models.score import LocalScoreEstimator
from madminer.utils.ml.eval import evaluate_flow_model, evaluate_ratio_model, evaluate_local_score_model
from madminer.utils.ml.utils import check_required_data
from madminer.utils.various import create_missing_folders, load_and_check, shuffle, restrict_samplesize
from madminer.utils.ml.methods import get_method_type, get_trainer, get_loss, package_training_data

try:
    FileNotFoundError
except NameError:
    FileNotFoundError = IOError

logger = logging.getLogger(__name__)


class MLForge:
    """
    Estimating likelihood ratios and scores with machine learning.

    Each instance of this class represents one neural estimator. The most important functions are:

    * `MLForge.train()` to train an estimator. The keyword `method` determines the inference technique
      and whether a class instance represents a single-parameterized likelihood ratio estimator, a doubly-parameterized
      likelihood ratio estimator, or a local score estimator.
    * `MLForge.evaluate()` to evaluate the estimator.
    * `MLForge.save()` to save the trained model to files.
    * `MLForge.load()` to load the trained model from files.

    Please see the tutorial for a detailed walk-through.
    """

    def __init__(self):
        self.method_type = None
        self.model = None
        self.method = None
        self.nde_type = None
        self.n_observables = None
        self.n_parameters = None
        self.n_hidden = None
        self.activation = None
        self.maf_n_mades = None
        self.maf_batch_norm = None
        self.maf_batch_norm_alpha = None
        self.maf_mog_n_components = None
        self.features = None
        self.x_scaling_means = None
        self.x_scaling_stds = None

    def train(
        self,
        method,
        x_filename,
        y_filename=None,
        theta0_filename=None,
        theta1_filename=None,
        r_xz_filename=None,
        t_xz0_filename=None,
        t_xz1_filename=None,
        features=None,
        nde_type="mafmog",
        n_hidden=(100, 100),
        activation="tanh",
        maf_n_mades=3,
        maf_batch_norm=False,
        maf_batch_norm_alpha=0.1,
        maf_mog_n_components=10,
        alpha=1.0,
        optimizer="amsgrad",
        n_epochs=50,
        batch_size=200,
        initial_lr=0.001,
        final_lr=0.0001,
        nesterov_momentum=None,
        validation_split=0.25,
        early_stopping=True,
        scale_inputs=True,
        shuffle_labels=False,
        grad_x_regularization=None,
        limit_samplesize=None,
        verbose="some",
    ):

        """
        Trains a neural network to estimate either the likelihood, the likelihood ratio, or the
        score.

        The keyword method determines the structure of the estimator that an instance of this class represents:

        * For 'alice', 'alices', 'carl', 'nde', 'rascal', 'rolr', and 'scandal', the neural network models
          the likelihood ratio as a function of the observables `x` and the numerator hypothesis `theta0`, while
          the denominator hypothesis is kept at a fixed reference value ("single-parameterized likelihood ratio
          estimator"). In addition to the likelihood ratio, the estimator allows to estimate the score at `theta0`.
        * For 'alice2', 'alices2', 'carl2', 'rascal2', and 'rolr2', the neural network models
          the likelihood ratio as a function of the observables `x`, the numerator hypothesis `theta0`, and the
          denominator hypothesis `theta1` ("doubly parameterized likelihood ratio estimator"). The score at `theta0`
          and `theta1` can also be evaluated.
        * For 'sally' and 'sallino', the neural networks models the score evaluated at some reference hypothesis
          ("local score regression"). The likelihood ratio cannot be estimated directly from the neural network, but
          can be estimated in a second step through density estimation in the estimated score space.

        Parameters
        ----------
        method : str
            The inference method used. Allows values are 'alice', 'alices', 'carl', 'nde', 'rascal', 'rolr', and
            'scandal' for a single-parameterized likelihood ratio estimator; 'alice2', 'alices2', 'carl2', 'rascal2',
            and 'rolr2' for a doubly-parameterized likelihood ratio estimator; and 'sally' and 'sallino' for local
            score regression.
            
        x_filename : str
            Path to an unweighted sample of observations, as saved by the `madminer.sampling.SampleAugmenter` functions.
            Required for all inference methods.
            
        y_filename : str or None, optional
            Path to an unweighted sample of class labels, as saved by the `madminer.sampling.SampleAugmenter` functions.
            Required for the 'alice', 'alice2', 'alices', 'alices2', 'carl', 'carl2', 'rascal', 'rascal2', 'rolr',
            and 'rolr2' methods. Default value: None.

        theta0_filename : str or None, optional
            Path to an unweighted sample of numerator parameters, as saved by the `madminer.sampling.SampleAugmenter`
            functions. Required for the 'alice', 'alice2', 'alices', 'alices2', 'carl', 'carl2', 'nde', 'rascal',
            'rascal2', 'rolr', 'rolr2', and 'scandal' methods. Default value: None.

        theta1_filename : str or None, optional
            Path to an unweighted sample of denominator parameters, as saved by the `madminer.sampling.SampleAugmenter`
            functions. Required for the 'alice2', 'alices2', 'carl2', 'rascal2', and 'rolr2' methods. Default value:
            None.

        r_xz_filename : str or None, optional
            Path to an unweighted sample of joint likelihood ratios, as saved by the `madminer.sampling.SampleAugmenter`
            functions. Required for the 'alice', 'alice2', 'alices', 'alices2', 'rascal', 'rascal2', 'rolr', and 'rolr2'
            methods. Default value: None.

        t_xz0_filename : str or None, optional
            Path to an unweighted sample of joint scores at theta0, as saved by the `madminer.sampling.SampleAugmenter`
            functions. Required for the 'alices', 'alices2', 'rascal', 'rascal2', 'sallino', 'sally', and 'scandal'
            methods. Default value: None.

        t_xz1_filename : str or None, optional
            Path to an unweighted sample of joint scores at theta1, as saved by the `madminer.sampling.SampleAugmenter`
            functions. Required for the 'rascal2' and 'alices2' methods. Default value: None.

        features : list of int or None, optional
            Indices of observables (features) that are used as input to the neural networks. If None, all observables
            are used. Default value: None.

        nde_type : {'maf', 'mafmog'}, optional
            If the method is 'nde' or 'scandal', nde_type determines the architecture used in the neural density
            estimator. Currently supported are 'maf' for a Masked Autoregressive Flow with a Gaussian base density, or
            'mafmog' for a Masked Autoregressive Flow with a mixture of Gaussian base densities. Default value:
            'mafmog'.

        n_hidden : tuple of int, optional
            Units in each hidden layer in the neural networks. If method is 'nde' or 'scandal', this refers to the
            setup of each individual MADE layer. Default value: (100, 100).
            
        activation : {'tanh', 'sigmoid', 'relu'}, optional
            Activation function. Default value: 'tanh'.

        maf_n_mades : int, optional
            If method is 'nde' or 'scandal', this sets the number of MADE layers. Default value: 3.

        maf_batch_norm : bool, optional
            If method is 'nde' or 'scandal', switches batch normalization layers after each MADE layer on or off.
            Default: False.

        maf_batch_norm_alpha : float, optional
            If method is 'nde' or 'scandal' and maf_batch_norm is True, this sets the alpha parameter in the calculation
            of the running average of the mean and variance. Default value: 0.1.

        maf_mog_n_components : int, optional
            If method is 'nde' or 'scandal' and nde_type is 'mafmog', this sets the number of Gaussian base components.
            Default value: 10.

        alpha : float, optional
            Hyperparameter weighting the score error in the loss function of the 'alices', 'alices2', 'rascal',
            'rascal2', and 'scandal' methods. Default value: 1.

        optimizer : {"adam", "amsgrad", "sgd"}, optional
            Optimization algorithm. Default value: "amsgrad".

        n_epochs : int, optional
            Number of epochs. Default value: 50.

        batch_size : int, optional
            Batch size. Default value: 200.

        initial_lr : float, optional
            Learning rate during the first epoch, after which it exponentially decays to final_lr. Default value:
            0.001.

        final_lr : float, optional
            Learning rate during the last epoch. Default value: 0.0001.

        nesterov_momentum : float or None, optional
            If trainer is "sgd", sets the Nesterov momentum. Default value: None.

        validation_split : float or None, optional
            Fraction of samples used  for validation and early stopping (if early_stopping is True). If None, the entire
            sample is used for training and early stopping is deactivated. Default value: 0.25.

        early_stopping : bool, optional
            Activates early stopping based on the validation loss (only if validation_split is not None). Default value:
            True.

        scale_inputs : bool, optional
            Scale the observables to zero mean and unit variance. Default value: True.

        shuffle_labels : bool, optional
            If True, the labels (`y`, `r_xz`, `t_xz`) are shuffled, while the observations (`x`) remain in their
            normal order. This serves as a closure test, in particular as cross-check against overfitting: an estimator
            trained with shuffle_labels=True should predict to likelihood ratios around 1 and scores around 0.

        grad_x_regularization : None
            Currently not supported.

        limit_samplesize : int or None, optional
            If not None, only this number of samples (events) is used to train the estimator. Default value: None.

        verbose : {"all", "many", "some", "few", "none}, optional
            Determines verbosity of training. Default value: "some".

        Returns
        -------
        results: ndarray
            Results from SingleParameterizedRatioTrainer.train or DoubleParameterizedRatioTrainer.train for example

        """

        logger.info("Starting training")
        logger.info("  Method:                 %s", method)
        logger.info("  Training data:          x at %s", x_filename)
        if theta0_filename is not None:
            logger.info("                          theta0 at %s", theta0_filename)
        if theta1_filename is not None:
            logger.info("                          theta1 at %s", theta1_filename)
        if y_filename is not None:
            logger.info("                          y at %s", y_filename)
        if r_xz_filename is not None:
            logger.info("                          r_xz at %s", r_xz_filename)
        if t_xz0_filename is not None:
            logger.info("                          t_xz (theta0) at %s", t_xz0_filename)
        if t_xz1_filename is not None:
            logger.info("                          t_xz (theta1) at %s", t_xz1_filename)
        if features is None:
            logger.info("  Features:               all")
        else:
            logger.info("  Features:               %s", features)
        logger.info("  Method:                 %s", method)
        if method in ["nde", "scandal"]:
            logger.info("  Neural density est.:    %s", nde_type)
        if method not in ["nde", "scandal"]:
            logger.info("  Hidden layers:          %s", n_hidden)
        if method in ["nde", "scandal"]:
            logger.info("  MAF, number MADEs:      %s", maf_n_mades)
            logger.info("  MAF, batch norm:        %s", maf_batch_norm)
            logger.info("  MAF, BN alpha:          %s", maf_batch_norm_alpha)
            logger.info("  MAF MoG, components:    %s", maf_mog_n_components)
        logger.info("  Activation function:    %s", activation)
        if method in ["cascal", "cascal2", "rascal", "rascal2", "scandal", "alices"]:
            logger.info("  alpha:                  %s", alpha)
        logger.info("  Batch size:             %s", batch_size)
        logger.info("  Optimizer:              %s", optimizer)
        logger.info("  Epochs:                 %s", n_epochs)
        logger.info("  Learning rate:          %s initially, decaying to %s", initial_lr, final_lr)
        if optimizer == "sgd":
            logger.info("  Nesterov momentum:      %s", nesterov_momentum)
        logger.info("  Validation split:       %s", validation_split)
        logger.info("  Early stopping:         %s", early_stopping)
        logger.info("  Scale inputs:           %s", scale_inputs)
        logger.info("  Shuffle labels          %s", shuffle_labels)
        if limit_samplesize is None:
            logger.info("  Samples:                all")
        else:
            logger.info("  Samples:                %s", limit_samplesize)

        # Check
        if grad_x_regularization is not None:
            logger.warning("grad_x_regularization is not supported in this version of MadMiner")

        # Load training data
        logger.info("Loading training data")
        theta0 = load_and_check(theta0_filename)
        theta1 = load_and_check(theta1_filename)
        x = load_and_check(x_filename)
        y = load_and_check(y_filename)
        r_xz = load_and_check(r_xz_filename)
        t_xz0 = load_and_check(t_xz0_filename)
        t_xz1 = load_and_check(t_xz1_filename)
        if y is not None:
            y = y.reshape((-1, 1))

        # Check necessary information is there
        if not check_required_data(method, r_xz, t_xz0, t_xz1, theta0, theta1, x, y):
            raise ValueError("Not all required data for method {} provided!".format(method))

        # Infer dimensions of problem
        n_samples = x.shape[0]
        n_observables = x.shape[1]
        if theta0 is not None:
            n_parameters = theta0.shape[1]
        else:
            n_parameters = t_xz0.shape[1]
        logger.info("Found %s samples with %s parameters and %s observables", n_samples, n_parameters, n_observables)

        # Limit sample size
        if limit_samplesize is not None and limit_samplesize < n_samples:
            logger.info("Only using %s of %s training samples", limit_samplesize, n_samples)
            x, theta0, theta1, y, r_xz, t_xz0, t_xz1 = restrict_samplesize(
                limit_samplesize, x, theta0, theta1, y, r_xz, t_xz0, t_xz1
            )

        # Scale features
        if scale_inputs:
            logger.info("Rescaling inputs")
            self._initialize_input_transform(x)
            x = self._transform_inputs(x)
        else:
            self._initialize_input_transform(x, False)

        logger.debug("Observable ranges:")
        for i in range(n_observables):
            logger.debug(
                "  x_%s: mean %s, std %s, range %s ... %s",
                i + 1,
                np.mean(x[:, i]),
                np.std(x[:, i]),
                np.min(x[:, i]),
                np.max(x[:, i]),
            )

        # Shuffle labels
        if shuffle_labels:
            logger.info("Shuffling labels")
            y, r_xz, t_xz0, t_xz1 = shuffle(y, r_xz, t_xz0, t_xz1)

        # Features
        self.features = features
        if features is not None:
            x = x[:, features]
            logger.info("Only using %s of %s observables", x.shape[1], n_observables)
            n_observables = x.shape[1]

        # Data
        data = package_training_data(method, x, theta0, theta1, y, r_xz, t_xz0, t_xz1)

        # Create model and save settings
        logger.info("Creating model for method %s", method)
        self._create_model(
            method,
            n_observables,
            n_parameters,
            n_hidden,
            activation,
            nde_type,
            maf_n_mades,
            maf_batch_norm,
            maf_batch_norm_alpha,
            maf_mog_n_components,
        )

        # Losses
        loss_functions, loss_labels, loss_weights = get_loss(method, alpha)

        # Optimizer
        opt_kwargs = None
        if optimizer == "adam":
            opt = optim.Adam
        elif optimizer == "amsgrad":
            opt = optim.Adam
            opt_kwargs = {"amsgrad": True}
        elif optimizer == "sgd":
            opt = optim.SGD
            if nesterov_momentum is not None:
                opt_kwargs = {"momentum": nesterov_momentum}
        else:
            raise ValueError("Unknown optimizer {}".format(optimizer))

        # Train model
        logger.info("Training model")
        trainer = get_trainer(method)(self.model)
        result = trainer.train(
            data=data,
            loss_functions=loss_functions,
            loss_weights=loss_weights,
            loss_labels=loss_labels,
            epochs=n_epochs,
            batch_size=batch_size,
            optimizer=opt,
            optimizer_kwargs=opt_kwargs,
            initial_lr=initial_lr,
            final_lr=final_lr,
            validation_split=validation_split,
            early_stopping=early_stopping,
            verbose=verbose,
        )
        return result

    def evaluate(self, x, theta0_filename=None, theta1_filename=None, test_all_combinations=True, evaluate_score=False):

        """
        Evaluates a trained estimator of the log likelihood ratio, the log likelihood, or the score, depending on the
        method.

        Parameters
        ----------
        x : str or ndarray
            Sample of observations, or path to numpy file with observations, as saved by the
            `madminer.sampling.SampleAugmenter` functions.

        theta0_filename : str or None, optional
            Path to an unweighted sample of numerator parameters, as saved by the `madminer.sampling.SampleAugmenter`
            functions. Required if the estimator was trained with the 'alice', 'alice2', 'alices', 'alices2', 'carl',
            'carl2', 'nde', 'rascal', 'rascal2', 'rolr', 'rolr2', or 'scandal' method. Default value: None.

        theta1_filename : str or None, optional
            Path to an unweighted sample of denominator parameters, as saved by the `madminer.sampling.SampleAugmenter`
            functions. Required if the estimator was trained with the 'alice2', 'alices2', 'carl2', 'rascal2', or
            'rolr2' method. Default value: None.

        test_all_combinations : bool, optional
            If method is not 'sally' and not 'sallino': If False, the number of samples in the observable and theta
            files has to match, and the likelihood ratio is evaluated only for the combinations
            `r(x_i | theta0_i, theta1_i)`. If True, `r(x_i | theta0_j, theta1_j)` for all pairwise combinations `i, j`
            are evaluated. Default value: True.

        evaluate_score : bool, optional
            If method is not 'sally' and not 'sallino', this sets whether in addition to the likelihood ratio the score
            is evaluated. Default value: False.

        return_grad_x : bool, optional
            If True, `grad_x log r(x)` or `grad_x t(x)` (for 'sally' or 'sallino' estimators) are returned in addition
            to the other outputs. Default value: False.

        Returns
        -------
        sally_estimated_score : ndarray
            Only returned if the network was trained with `method='sally'` or `method='sallino'`. In this case, an
            array of the estimator for `t(x_i | theta_ref)` is returned for all events `i`.

        log_likelihood_ratio : ndarray
            Only returned if the network was trained with neither `method='sally'` nor `method='sallino'`. The estimated
            log likelihood ratio. If test_all_combinations is True, the result has shape `(n_thetas, n_x)`. Otherwise,
            it has shape `(n_samples,)`.

        score_theta0 : ndarray or None
            Only returned if the network was trained with neither `method='sally'` nor `method='sallino'`. None if
            evaluate_score is False. Otherwise the derived estimated score at `theta0`. If test_all_combinations is
            True, the result has shape `(n_thetas, n_x, n_parameters)`. Otherwise, it has shape
            `(n_samples, n_parameters)`.

        score_theta1 : ndarray or None
            Only returned if the network was trained with neither `method='sally'` nor `method='sallino'`. None if
            evaluate_score is False, or the network was trained with any method other than 'alice2', 'alices2', 'carl2',
            'rascal2', or 'rolr2'. Otherwise the derived estimated score at `theta1`. If test_all_combinations is
            True, the result has shape `(n_thetas, n_x, n_parameters)`. Otherwise, it has shape
            `(n_samples, n_parameters)`.

        grad_x : ndarray
            Only returned if return_grad_x is True.

        """
        if self.method_type in ["parameterized", "doubly_parameterized"]:
            return self.evaluate_log_likelihood_ratio(
                x, theta0_filename, theta1_filename, test_all_combinations, evaluate_score
            )
        elif self.method_type == "nde":
            return self.evaluate_log_likelihood(x, theta0_filename, test_all_combinations, evaluate_score)
        elif self.method_type == "local_score":
            return self.evaluate_score(x)
        else:
            raise RuntimeError("Unknown method type %s", self.method_type)

    def evaluate_log_likelihood_ratio(
        self, x, theta0_filename=None, theta1_filename=None, test_all_combinations=True, evaluate_score=False
    ):

        """
        Evaluates a trained estimator of the log likelihood ratio, the log likelihood, or the score, depending on the
        method.

        Parameters
        ----------
        x : str or ndarray
            Sample of observations, or path to numpy file with observations, as saved by the
            `madminer.sampling.SampleAugmenter` functions.

        theta0_filename : str or None, optional
            Path to an unweighted sample of numerator parameters, as saved by the `madminer.sampling.SampleAugmenter`
            functions. Required if the estimator was trained with the 'alice', 'alice2', 'alices', 'alices2', 'carl',
            'carl2', 'nde', 'rascal', 'rascal2', 'rolr', 'rolr2', or 'scandal' method. Default value: None.

        theta1_filename : str or None, optional
            Path to an unweighted sample of denominator parameters, as saved by the `madminer.sampling.SampleAugmenter`
            functions. Required if the estimator was trained with the 'alice2', 'alices2', 'carl2', 'rascal2', or
            'rolr2' method. Default value: None.

        test_all_combinations : bool, optional
            If method is not 'sally' and not 'sallino': If False, the number of samples in the observable and theta
            files has to match, and the likelihood ratio is evaluated only for the combinations
            `r(x_i | theta0_i, theta1_i)`. If True, `r(x_i | theta0_j, theta1_j)` for all pairwise combinations `i, j`
            are evaluated. Default value: True.

        evaluate_score : bool, optional
            If method is not 'sally' and not 'sallino', this sets whether in addition to the likelihood ratio the score
            is evaluated. Default value: False.

        Returns
        -------
        log_likelihood_ratio : ndarray
            Only returned if the network was trained with neither `method='sally'` nor `method='sallino'`. The estimated
            log likelihood ratio. If test_all_combinations is True, the result has shape `(n_thetas, n_x)`. Otherwise,
            it has shape `(n_samples,)`.

        score_theta0 : ndarray or None
            Only returned if the network was trained with neither `method='sally'` nor `method='sallino'`. None if
            evaluate_score is False. Otherwise the derived estimated score at `theta0`. If test_all_combinations is
            True, the result has shape `(n_thetas, n_x, n_parameters)`. Otherwise, it has shape
            `(n_samples, n_parameters)`.

        score_theta1 : ndarray or None
            Only returned if the network was trained with neither `method='sally'` nor `method='sallino'`. None if
            evaluate_score is False, or the network was trained with any method other than 'alice2', 'alices2', 'carl2',
            'rascal2', or 'rolr2'. Otherwise the derived estimated score at `theta1`. If test_all_combinations is
            True, the result has shape `(n_thetas, n_x, n_parameters)`. Otherwise, it has shape
            `(n_samples, n_parameters)`.

        """

        if self.model is None:
            raise ValueError("No model -- train or load model before evaluating it!")

        # Load training data
        logger.debug("Loading evaluation data")
        theta0s = load_and_check(theta0_filename)
        theta1s = load_and_check(theta1_filename)
        if isinstance(x, six.string_types):
            x = load_and_check(x)

        # Scale observables
        x = self._transform_inputs(x)

        # Restrict features
        if self.features is not None:
            x = x[:, self.features]

        # Balance thetas
        if theta1s is None and theta0s is not None:
            theta1s = [None for _ in theta0s]
        elif theta1s is not None and theta0s is not None:
            if len(theta1s) > len(theta0s):
                theta0s = [theta0s[i % len(theta0s)] for i in range(len(theta1s))]
            elif len(theta1s) < len(theta0s):
                theta1s = [theta1s[i % len(theta1s)] for i in range(len(theta0s))]

        # Evaluation for all other methods
        all_log_r_hat = []
        all_t_hat0 = []
        all_t_hat1 = []

        if test_all_combinations:
            logger.debug("Starting ratio evaluation for all combinations")

            for i, (theta0, theta1) in enumerate(zip(theta0s, theta1s)):
                logger.debug(
                    "Starting ratio evaluation for thetas %s / %s: %s vs %s", i + 1, len(theta0s), theta0, theta1
                )
                _, log_r_hat, t_hat0, t_hat1 = evaluate_ratio_model(
                    model=self.model,
                    method_type=self.method_type,
                    theta0s=[theta0],
                    theta1s=[theta1] if theta1 is not None else None,
                    xs=x,
                    evaluate_score=evaluate_score,
                )

                all_log_r_hat.append(log_r_hat)
                all_t_hat0.append(t_hat0)
                all_t_hat1.append(t_hat1)

            all_log_r_hat = np.array(all_log_r_hat)
            all_t_hat0 = np.array(all_t_hat0)
            all_t_hat1 = np.array(all_t_hat1)

        else:
            logger.debug("Starting ratio evaluation")
            _, all_log_r_hat, all_t_hat0, all_t_hat1 = evaluate_ratio_model(
                model=self.model,
                method_type=self.method_type,
                theta0s=theta0s,
                theta1s=None if None in theta1s else theta1s,
                xs=x,
                evaluate_score=evaluate_score,
            )

        logger.debug("Evaluation done")
        return all_log_r_hat, all_t_hat0, all_t_hat1

    def evaluate_score(self, x, return_grad_x=False):

        """
        Evaluates a trained estimator of the the score.

        Parameters
        ----------
        x : str or ndarray
            Sample of observations, or path to numpy file with observations, as saved by the
            `madminer.sampling.SampleAugmenter` functions.

        return_grad_x : bool, optional
            If True, `grad_x log r(x)` or `grad_x t(x)` (for 'sally' or 'sallino' estimators) are returned in addition
            to the other outputs. Default value: False.

        Returns
        -------
        sally_estimated_score : ndarray
            Only returned if the network was trained with `method='sally'` or `method='sallino'`. In this case, an
            array of the estimator for `t(x_i | theta_ref)` is returned for all events `i`.

        grad_x : ndarray
            Only returned if return_grad_x is True.

        """

        if self.model is None:
            raise ValueError("No model -- train or load model before evaluating it!")

        # Load training data
        logger.debug("Loading evaluation data")
        if isinstance(x, six.string_types):
            x = load_and_check(x)

        # Scale observables
        x = self._transform_inputs(x)

        # Restrict featuers
        if self.features is not None:
            x = x[:, self.features]

        # SALLY evaluation
        if self.method not in ["sally", "sallino"]:
            raise NotImplementedError("Score evaluation only implemented for methods SALLY and SALLINO.")

        logger.debug("Starting score evaluation")

        all_t_hat = evaluate_local_score_model(model=self.model, xs=x)
        return all_t_hat

    def evaluate_log_likelihood(self, x, theta0_filename=None, test_all_combinations=True, evaluate_score=False):

        """
        Evaluates a trained estimator of the log likelihood.

        Parameters
        ----------
        x : str or ndarray
            Sample of observations, or path to numpy file with observations, as saved by the
            `madminer.sampling.SampleAugmenter` functions.

        theta0_filename : str or None, optional
            Path to an unweighted sample of numerator parameters, as saved by the `madminer.sampling.SampleAugmenter`
            functions. Required if the estimator was trained with the 'alice', 'alice2', 'alices', 'alices2', 'carl',
            'carl2', 'nde', 'rascal', 'rascal2', 'rolr', 'rolr2', or 'scandal' method. Default value: None.

        test_all_combinations : bool, optional
            If method is not 'sally' and not 'sallino': If False, the number of samples in the observable and theta
            files has to match, and the likelihood ratio is evaluated only for the combinations
            `r(x_i | theta0_i, theta1_i)`. If True, `r(x_i | theta0_j, theta1_j)` for all pairwise combinations `i, j`
            are evaluated. Default value: True.

        evaluate_score : bool, optional
            If method is not 'sally' and not 'sallino', this sets whether in addition to the likelihood ratio the score
            is evaluated. Default value: False.

        Returns
        -------

        log_likelihood : ndarray
            The estimated log likelihood. If test_all_combinations is True, the result has shape `(n_thetas, n_x)`.
            Otherwise, it has shape `(n_samples,)`.

        score_theta0 : ndarray or None
            None if
            evaluate_score is False. Otherwise the derived estimated score at `theta0`. If test_all_combinations is
            True, the result has shape `(n_thetas, n_x, n_parameters)`. Otherwise, it has shape
            `(n_samples, n_parameters)`.

        """

        if self.model is None:
            raise ValueError("No model -- train or load model before evaluating it!")

        # Load training data
        logger.debug("Loading evaluation data")
        thetas = load_and_check(theta0_filename)
        if isinstance(x, six.string_types):
            x = load_and_check(x)

        # Scale observables
        x = self._transform_inputs(x)

        # Restrict featuers
        if self.features is not None:
            x = x[:, self.features]

        if self.method_type != "nde":
            raise RuntimeError("Likelihood estimation only possible for methods NDE and SCANDAL")

        # Evaluation for all other methods
        all_log_p_hat = []
        all_t_hat = []

        if test_all_combinations:
            logger.debug("Starting ratio evaluation for all combinations")

            for i, theta in enumerate(thetas):
                logger.debug("Starting log likelihood evaluation for theta %s / %s: %s", i + 1, len(thetas), theta)

                log_p_hat, t_hat = evaluate_flow_model(
                    model=self.model, thetas=[theta], xs=x, evaluate_score=evaluate_score
                )

                all_log_p_hat.append(log_p_hat)
                all_t_hat.append(t_hat)

            all_log_p_hat = np.array(all_log_p_hat)
            all_t_hat = np.array(all_t_hat)

        else:
            logger.debug("Starting log likelihood evaluation")

            all_log_p_hat, all_t_hat = evaluate_flow_model(
                model=self.model, thetas=thetas, xs=x, evaluate_score=evaluate_score
            )

        logger.debug("Evaluation done")
        return all_log_p_hat, all_t_hat

    def calculate_fisher_information(self, x, weights=None, n_events=1, sum_events=True):

        """
        Calculates the expected Fisher information matrix based on the kinematic information in a given number of
        events. Currently only supported for estimators trained with `method='sally'` or `method='sallino'`.

        Parameters
        ----------
        x : str or ndarray
            Sample of observations, or path to numpy file with observations, as saved by the
            `madminer.sampling.SampleAugmenter` functions. Note that this sample has to be sampled from the reference
            parameter where the score is estimated with the SALLY / SALLINO estimator!

        weights : None or ndarray, optional
            Weights for the observations. If None, all events are taken to have equal weight. Default value: None.
            
        n_events : float, optional
            Expected number of events for which the kinematic Fisher information should be calculated. Default value: 1.

        sum_events : bool, optional
            If True, the expected Fisher information summed over the events x is calculated. If False, the per-event
            Fisher information for each event is returned. Default value: True.

        Returns
        -------
        fisher_information : ndarray
            Expected kinematic Fisher information matrix with shape `(n_events, n_parameters, n_parameters)` if
            sum_events is False or `(n_parameters, n_parameters)` if sum_events is True.

        """

        if self.model is None:
            raise ValueError("No model -- train or load model before evaluating it!")

        # Load training data
        logger.debug("Loading evaluation data")
        if isinstance(x, six.string_types):
            x = load_and_check(x)
        n_samples = x.shape[0]

        # Scale observables
        x = self._transform_inputs(x)

        # Restrict featuers
        if self.features is not None:
            x = x[:, self.features]

        # Estimate scores
        if self.method in ["sally", "sallino"]:
            logger.debug("Starting score evaluation")

            t_hats = evaluate_local_score_model(model=self.model, xs=x)
        else:
            raise NotImplementedError("Fisher information calculation only implemented for SALLY estimators")

        # Weights
        if weights is None:
            weights = np.ones(n_samples)
        weights /= np.sum(weights)

        # Calculate Fisher information
        if sum_events:
            fisher_information = float(n_events) * np.einsum("n,ni,nj->ij", weights, t_hats, t_hats)
        else:
            fisher_information = float(n_events) * np.einsum("n,ni,nj->nij", weights, t_hats, t_hats)

        # Calculate expected score
        expected_score = np.mean(t_hats, axis=0)
        logger.debug("Expected per-event score (should be close to zero): %s", expected_score)

        return fisher_information

    def save(self, filename, save_model=False):

        """
        Saves the trained model to four files: a JSON file with the settings, a pickled pyTorch state dict
        file, and numpy files for the mean and variance of the inputs (used for input scaling).

        Parameters
        ----------
        filename : str
            Path to the files. '_settings.json' and '_state_dict.pl' will be added.

        save_model : bool, optional
            If True, the whole model is saved in addition to the state dict. This is not necessary for loading it
            again with MLForge.load(), but can be useful for debugging, for instance to plot the computational graph.

        Returns
        -------
            None

        """

        if self.model is None:
            raise ValueError("No model -- train or load model before saving!")

        # Check paths
        create_missing_folders([os.path.dirname(filename)])

        # Save settings
        logger.debug("Saving settings to %s_settings.json", filename)

        settings = {
            "method": self.method,
            "method_type": self.method_type,
            "n_observables": self.n_observables,
            "n_parameters": self.n_parameters,
            "n_hidden": list(self.n_hidden),
            "activation": self.activation,
            "features": self.features,
            "nde_type": self.nde_type,
            "maf_n_mades": self.maf_n_mades,
            "maf_batch_norm": self.maf_batch_norm,
            "maf_batch_norm_alpha": self.maf_batch_norm_alpha,
            "maf_mog_n_components": self.maf_mog_n_components,
        }

        with open(filename + "_settings.json", "w") as f:
            json.dump(settings, f)

        # Save scaling
        if self.x_scaling_stds is not None and self.x_scaling_means is not None:
            logger.debug("Saving input scaling information to %s_x_means.npy and %s_x_stds.npy", filename, filename)
            np.save(filename + "_x_means.npy", self.x_scaling_means)
            np.save(filename + "_x_stds.npy", self.x_scaling_stds)

        # Save state dict
        logger.debug("Saving state dictionary to %s_state_dict.pt", filename)
        torch.save(self.model.state_dict(), filename + "_state_dict.pt")

        # Save model
        if save_model:
            logger.debug("Saving model to %s_model.pt", filename)
            torch.save(self.model, filename + "_model.pt")

    def load(self, filename):

        """
        Loads a trained model from files.

        Parameters
        ----------
        filename : str
            Path to the files. '_settings.json' and '_state_dict.pl' will be added.

        Returns
        -------
            None

        """

        # Load settings
        logger.debug("Loading settings from %s_settings.json", filename)

        with open(filename + "_settings.json", "r") as f:
            settings = json.load(f)

        method = settings["method"]
        n_observables = int(settings["n_observables"])
        n_parameters = int(settings["n_parameters"])
        n_hidden = tuple([int(item) for item in settings["n_hidden"]])
        activation = str(settings["activation"])
        features = settings["features"]
        nde_type = settings["nde_type"]
        maf_n_mades = int(settings["maf_n_mades"])
        maf_batch_norm = bool(settings["maf_batch_norm"])
        maf_batch_norm_alpha = float(settings["maf_batch_norm_alpha"])
        maf_mog_n_components = int(settings["maf_mog_n_components"])

        logger.debug(
            "  Found method %s, %s observables, %s parameters, %s hidden layers, %s activation function, "
            "features %s",
            method,
            n_observables,
            n_parameters,
            n_hidden,
            activation,
            features,
        )

        # Features
        if features == "None":
            self.features = None
        if features is not None:
            self.features = list([int(item) for item in features])

        # Load scaling
        try:
            self.x_scaling_means = np.load(filename + "_x_means.npy")
            self.x_scaling_stds = np.load(filename + "_x_stds.npy")
            logger.debug(
                "  Found input scaling information: means %s, stds %s", self.x_scaling_means, self.x_scaling_stds
            )
        except FileNotFoundError:
            logger.warning("Scaling information not found in %s", filename)
            self.x_scaling_means = None
            self.x_scaling_stds = None

        # Create model and save in self
        self._create_model(
            method,
            n_observables,
            n_parameters,
            n_hidden,
            activation,
            nde_type,
            maf_n_mades,
            maf_batch_norm,
            maf_batch_norm_alpha,
            maf_mog_n_components,
        )

        # Load state dict
        logger.debug("Loading state dictionary from %s_state_dict.pt", filename)
        self.model.load_state_dict(torch.load(filename + "_state_dict.pt"))

    def _initialize_input_transform(self, x, transform=True):
        if transform:
            self.x_scaling_means = np.mean(x, axis=0)
            self.x_scaling_stds = np.maximum(np.std(x, axis=0), 1.0e-6)
        else:
            n_parameters = x.shape[0]

            self.x_scaling_means = np.zeros(n_parameters)
            self.x_scaling_stds = np.ones(n_parameters)

    def _transform_inputs(self, x):
        if self.x_scaling_means is not None and self.x_scaling_stds is not None:
            x_scaled = x - self.x_scaling_means
            x_scaled /= self.x_scaling_stds
        else:
            x_scaled = x
        return x_scaled

    def _create_model(
        self,
        method,
        n_observables,
        n_parameters,
        n_hidden,
        activation,
        nde_type=None,
        maf_n_mades=None,
        maf_batch_norm=None,
        maf_batch_norm_alpha=None,
        maf_mog_n_components=None,
    ):

        self.method = method
        self.n_observables = n_observables
        self.n_parameters = n_parameters
        self.n_hidden = n_hidden
        self.activation = activation
        self.maf_n_mades = maf_n_mades
        self.maf_batch_norm = maf_batch_norm
        self.maf_batch_norm_alpha = maf_batch_norm_alpha
        self.maf_mog_n_components = maf_mog_n_components

        self.method_type = get_method_type(method)
        if self.method_type == "parameterized":
            self.model = ParameterizedRatioEstimator(
                n_observables=n_observables, n_parameters=n_parameters, n_hidden=n_hidden, activation=activation
            )
        elif self.method_type == "doubly_parameterized":
            self.model = DoublyParameterizedRatioEstimator(
                n_observables=n_observables, n_parameters=n_parameters, n_hidden=n_hidden, activation=activation
            )
        elif self.method_type == "local_score":
            self.model = LocalScoreEstimator(
                n_observables=n_observables, n_parameters=n_parameters, n_hidden=n_hidden, activation=activation
            )
        elif self.method_type == "nde":
            self.nde_type = nde_type
            if nde_type == "maf":
                self.model = ConditionalMaskedAutoregressiveFlow(
                    n_conditionals=n_parameters,
                    n_inputs=n_observables,
                    n_hiddens=n_hidden,
                    n_mades=maf_n_mades,
                    activation=activation,
                    batch_norm=maf_batch_norm,
                    alpha=maf_batch_norm_alpha,
                )
            elif nde_type == "mafmog":
                self.model = ConditionalMixtureMaskedAutoregressiveFlow(
                    n_conditionals=n_parameters,
                    n_inputs=n_observables,
                    n_components=maf_mog_n_components,
                    n_hiddens=n_hidden,
                    n_mades=maf_n_mades,
                    activation=activation,
                    batch_norm=maf_batch_norm,
                    alpha=maf_batch_norm_alpha,
                )
            else:
                raise RuntimeError("Unknown NDE type {}".format(nde_type))
        else:
            raise RuntimeError("Unknown method {}".format(method))


class EnsembleForge:
    """
    Ensemble methods for likelihood ratio and score information.

    Generally, EnsembleForge instances can be used very similarly to MLForge instances:

    * The initialization of EnsembleForge takes a list of (trained or untrained) MLForge instances.
    * The methods `EnsembleForge.train_one()` and `EnsembleForge.train_all()` train the estimators (this can also be
      done outside of EnsembleForge).
    * `EnsembleForge.calculate_expectation()` can be used to calculate the expectation of the estimation likelihood
      ratio or the expected estimated score over a validation sample. Ideally (and assuming the correct sampling),
      these expectation values should be close to zero. Deviations from zero therefore point out that the estimator
      is probably inaccurate.
    * `EnsembleForge.evaluate()` and `EnsembleForge.calculate_fisher_information()` can then be used to calculate
      ensemble predictions. The user has the option to treat all estimators equally ('committee method') or to give those
      with expected score / ratio close to zero a higher weight.
    * `EnsembleForge.save()` and `EnsembleForge.load()` can store all estimators in one folder.

    The individual estimators in the ensemble can be trained with different methods, but they have to be of the same
    type: either all estimators are single-parameterized likelihood ratio estimators, or all estimators are
    doubly-parameterized likelihood estimators, or all estimators are local score regressors.

    Parameters
    ----------
    estimators : None or int or list of (MLForge or str), optional
        If int, sets the number of estimators that will be created as new MLForge instances. If list, sets
        the estimators directly, either from MLForge instances or filenames (that are then loaded with
        `MLForge.load()`). If None, the ensemble is initialized without estimators. Note that the estimators have
        to be consistent: either all of them are trained with a local score method ('sally' or 'sallino'); or all of
        them are trained with a single-parameterized method ('carl', 'rolr', 'rascal', 'scandal', 'alice', or 'alices');
        or all of them are trained with a doubly parameterized method ('carl2', 'rolr2', 'rascal2', 'alice2', or
        'alices2'). Mixing estimators of different types within one of these three categories is supported, but mixing
        estimators from different categories is not and will raise a RuntimeException. Default value: None.

    Attributes
    ----------
    estimators : list of MLForge
        The estimators in the form of MLForge instances.
    """

    def __init__(self, estimators=None):
        self.n_parameters = None
        self.n_observables = None

        # Initialize estimators
        if estimators is None:
            self.estimators = []
        elif isinstance(estimators, int):
            self.estimators = [MLForge() for _ in range(estimators)]
        else:
            self.estimators = []
            for estimator in estimators:
                if isinstance(estimator, six.string_types):
                    estimator_object = MLForge()
                    estimator_object.load(estimator)
                elif isinstance(estimator, MLForge):
                    estimator_object = estimator
                else:
                    raise ValueError("Entry {} in estimators is neither str nor MLForge instance")

                self.estimators.append(estimator_object)

        self.n_estimators = len(self.estimators)
        self.expectations = None

        # Consistency checks
        for estimator in self.estimators:
            assert isinstance(estimator, MLForge), "Estimator is no MLForge instance!"

        self._check_consistency()

    def add_estimator(self, estimator):
        """
        Adds an estimator to the ensemble.

        Parameters
        ----------
        estimator : MLForge or str
            The estimator, either as MLForge instance or filename (which is then loaded with `MLForge.load()`).

        Returns
        -------
            None

        """
        if isinstance(estimator, six.string_types):
            estimator_object = MLForge()
            estimator_object.load(estimator)
        elif isinstance(estimator, MLForge):
            estimator_object = estimator
        else:
            raise ValueError("Entry {} in estimators is neither str nor MLForge instance")

        self.estimators.append(estimator_object)
        self.n_estimators = len(self.estimators)

    def train_one(self, i, **kwargs):
        """
        Trains an individual estimator. See `MLForge.train()`.

        Parameters
        ----------
        i : int
            The index `0 <= i < n_estimators` of the estimator to be trained.

        kwargs : dict
            Parameters for `MLForge.train()`.

        Returns
        -------
            None

        """

        self._check_consistency(kwargs)

        self.estimators[i].train(**kwargs)

    def train_all(self, **kwargs):
        """
        Trains all estimators. See `MLForge.train()`.

        Parameters
        ----------
        kwargs : dict
            Parameters for `MLForge.train()`. If a value in this dict is a list, it has to have length `n_estimators`
            and contain one value of this parameter for each of the estimators. Otherwise the value is used as parameter
            for the training of all the estimators.

        Returns
        -------
            None

        """
        logger.info("Training %s estimators in ensemble", self.n_estimators)

        for key, value in six.iteritems(kwargs):
            if not isinstance(value, list):
                kwargs[key] = [value for _ in range(self.n_estimators)]

            assert len(kwargs[key]) == self.n_estimators, "Keyword {} has wrong length {}".format(key, len(value))

        self._check_consistency(kwargs)

        for i, estimator in enumerate(self.estimators):
            kwargs_this_estimator = {}
            for key, value in six.iteritems(kwargs):
                kwargs_this_estimator[key] = value[i]

            logger.info("Training estimator %s / %s in ensemble", i + 1, self.n_estimators)
            estimator.train(**kwargs_this_estimator)

    def calculate_expectation(self, x_filename, theta0_filename=None, theta1_filename=None):
        """
        Calculates the expectation of the estimation likelihood ratio or the expected estimated score over a validation
        sample. Ideally (and assuming the correct sampling), these expectation values should be close to zero.
        Deviations from zero therefore point out that the estimator is probably inaccurate.

        Parameters
        ----------
        x_filename : str
            Path to an unweighted sample of observations, as saved by the `madminer.sampling.SampleAugmenter` functions.

        theta0_filename : str or None, optional
            Path to an unweighted sample of numerator parameters, as saved by the `madminer.sampling.SampleAugmenter`
            functions. Required if the estimators were trained with the 'alice', 'alice2', 'alices', 'alices2', 'carl',
            'carl2', 'nde', 'rascal', 'rascal2', 'rolr', 'rolr2', or 'scandal' method. Default value: None.

        theta1_filename : str or None, optional
            Path to an unweighted sample of denominator parameters, as saved by the `madminer.sampling.SampleAugmenter`
            functions. Required if the estimators were trained with the 'alice2', 'alices2', 'carl2', 'rascal2', or
            'rolr2' method. Default value: None.

        Returns
        -------
        expectations : ndarray
            Expected score (if the estimators were trained with the 'sally' or 'sallino' methods) or likelihood ratio
            (otherwise).

        """

        logger.info("Calculating expectation for %s estimators in ensemble", self.n_estimators)

        self.expectations = []
        method_type = self._check_consistency()

        for i, estimator in enumerate(self.estimators):
            logger.info("Starting evaluation for estimator %s / %s in ensemble", i + 1, self.n_estimators)

            # Calculate expected score / ratio
            if method_type == "local_score":
                prediction = estimator.evaluate(x_filename, theta0_filename, theta1_filename)
            else:
                raise NotImplementedError("Expectation calculation currently only implemented for SALLY and SALLINO!")

            self.expectations.append(np.mean(prediction, axis=0))

        self.expectations = np.array(self.expectations)

        return self.expectations

    def evaluate(
        self,
        x,
        theta0_filename=None,
        theta1_filename=None,
        test_all_combinations=True,
        vote_expectation_weight=None,
        calculate_covariance=False,
        return_individual_predictions=False,
    ):

        """
        Evaluates the estimators of the likelihood ratio (or, if method is 'sally' or 'sallino', the score), and
        calculates the ensemble mean or variance.

        The user has the option to treat all estimators equally ('committee method') or to give those with expected
        score / ratio close to zero (as calculated by `calculate_expectation()`) a higher weight. In the latter case,
        the ensemble mean `f(x)` is calculated as `f(x)  =  sum_i w_i f_i(x)` with weights
        `w_i  =  exp(-vote_expectation_weight |E[f_i]|) / sum_j exp(-vote_expectation_weight |E[f_j]|)`. Here `f_i(x)`
        are the individual estimators and `E[f_i]` is the expectation value calculated by `calculate_expectation()`.

        Parameters
        ----------
        x : str or ndarray
            Sample of observations, or path to numpy file with observations, as saved by the
            `madminer.sampling.SampleAugmenter` functions. Note that this sample has to be sampled from the reference
            parameter where the score is estimated with the SALLY / SALLINO estimator!

        theta0_filename : str or None, optional
            Path to an unweighted sample of numerator parameters, as saved by the `madminer.sampling.SampleAugmenter`
            functions. Required if the estimator was trained with the 'alice', 'alice2', 'alices', 'alices2', 'carl',
            'carl2', 'nde', 'rascal', 'rascal2', 'rolr', 'rolr2', or 'scandal' method. Default value: None.

        theta1_filename : str or None, optional
            Path to an unweighted sample of denominator parameters, as saved by the `madminer.sampling.SampleAugmenter`
            functions. Required if the estimator was trained with the 'alice2', 'alices2', 'carl2', 'rascal2', or
            'rolr2' method. Default value: None.

        test_all_combinations : bool, optional
            If method is not 'sally' and not 'sallino': If False, the number of samples in the observable and theta
            files has to match, and the likelihood ratio is evaluated only for the combinations
            `r(x_i | theta0_i, theta1_i)`. If True, `r(x_i | theta0_j, theta1_j)` for all pairwise combinations `i, j`
            are evaluated. Default value: True.

        vote_expectation_weight : float or list of float or None, optional
            Factor that determines how much more weight is given to those estimators with small expectation value (as
            calculated by `calculate_expectation()`). If a list is given, results are returned for each element in the
            list. If None, or if `calculate_expectation()` has not been called, all estimators are treated equal.
            Default value: None.

        calculate_covariance : bool, optional
            Whether the covariance matrix is calculated. Default value: False.

        return_individual_predictions : bool, optional
            Whether the individual estimator predictions are returned. Default value: False.

        Returns
        -------
        mean_prediction : ndarray or list of ndarray
            The (weighted) ensemble mean of the estimators. If the estimators were trained with `method='sally'` or
            `method='sallino'`, this is an array of the estimator for `t(x_i | theta_ref)` for all events `i`.
            Otherwise, the estimated likelihood ratio (if test_all_combinations is True, the result has shape
            `(n_thetas, n_x)`, otherwise, it has shape `(n_samples,)`). If more then one value vote_expectation_weight
            is given, this is a list with results for all entries in vote_expectation_weight.

        covariance : None or ndarray or list of ndarray
            The covariance matrix of the (flattened) predictions, defined as the ensemble covariance. If more then one
            value vote_expectation_weight is given, this is a list with results
            for all entries in vote_expectation_weight. If calculate_covariance is False, None is returned.

        weights : ndarray or list of ndarray
            Only returned if return_individual_predictions is True. The estimator weights `w_i`. If more then one value
            vote_expectation_weight is given, this is a list with results for all entries in vote_expectation_weight.

        individual_predictions : ndarray
            Only returned if return_individual_predictions is True. The individual estimator predictions.

        """
        logger.info("Evaluating %s estimators in ensemble", self.n_estimators)

        # Calculate weights of each estimator in vote
        if self.expectations is None or vote_expectation_weight is None:
            weights = [np.ones(self.n_estimators)]
        else:
            if len(self.expectations.shape) == 1:
                expectations_norm = self.expectations
            elif len(self.expectations.shape) == 2:
                expectations_norm = np.linalg.norm(self.expectations, axis=1)
            else:
                expectations_norm = [np.linalg.norm(expectation) for expectation in self.expectations]

            if not isinstance(vote_expectation_weight, list):
                vote_expectation_weight = [vote_expectation_weight]

            weights = []
            for vote_weight in vote_expectation_weight:
                if vote_weight is None:
                    these_weights = np.ones(self.n_estimators)
                else:
                    these_weights = np.exp(-vote_weight * expectations_norm)
                these_weights /= np.sum(these_weights)
                weights.append(these_weights)

        logger.debug("Estimator weights: %s", weights)

        # Calculate estimator predictions
        predictions = []
        for i, estimator in enumerate(self.estimators):
            logger.info("Starting evaluation for estimator %s / %s in ensemble", i + 1, self.n_estimators)

            predictions.append(
                estimator.evaluate(x, theta0_filename, theta1_filename, test_all_combinations, evaluate_score=False)
            )

            logger.debug("Estimator %s predicts %s for first event", i + 1, predictions[-1][0, :])
        predictions = np.array(predictions)

        # Calculate weighted means and covariance matrices
        means = []
        covariances = []

        for these_weights in weights:
            mean = np.average(predictions, axis=0, weights=these_weights)
            means.append(mean)

            if calculate_covariance:
                predictions_flat = predictions.reshape((predictions.shape[0], -1))

                covariance = np.cov(predictions_flat.T, aweights=these_weights)
            else:
                covariance = None

            covariances.append(covariance)

        # Returns
        if len(weights) == 1:
            if return_individual_predictions:
                return means[0], covariances[0], weights[0], predictions
            return means[0], covariances[0]

        if return_individual_predictions:
            return means, covariances, weights, predictions
        return means, covariances

    def calculate_fisher_information(
        self,
        x,
        obs_weights=None,
        n_events=1,
        mode="score",
        uncertainty="ensemble",
        vote_expectation_weight=None,
        return_individual_predictions=False,
        sum_events=True,
    ):
        """
        Calculates expected Fisher information matrices for an ensemble of SALLY estimators.

        There are two ways of calculating the ensemble average. In the default "score" mode, the ensemble average for
        the score is calculated for each event, and the Fisher information is calculated based on these mean scores. In
        the "information" mode, the Fisher information is calculated for each estimator separately and the ensemble
        mean is calculated only for the final Fisher information matrix. The "score" mode is generally assumed to be
        more precise and is the default.

        In the "score" mode, the covariance matrix of the final result is calculated in the following way:
        - For each event `x` and each estimator `a`, the "shifted" predicted score is calculated as
          `t_a'(x) = t(x) + 1/sqrt(n) * (t_a(x) - t(x))`. Here `t(x)` is the mean score (averaged over the ensemble)
          for this event, `t_a(x)` is the prediction of estimator `a` for this event, and `n` is the number of
          estimators. The ensemble variance of these shifted score predictions is equal to the uncertainty on the mean
          of the ensemble of original predictions.
        - For each estimator `a`, the shifted Fisher information matrix `I_a'` is calculated  from the shifted predicted
          scores.
        - The ensemble covariance between all Fisher information matrices `I_a'` is calculated and taken as the
          measure of uncertainty on the Fisher information calculated from the mean scores.

        In the "information" mode, the user has the option to treat all estimators equally ('committee method') or to
        give those with expected score close to zero (as calculated by `calculate_expectation()`) a higher weight. In
        this case, the ensemble mean `I` is calculated as `I  =  sum_i w_i I_i` with weights
        `w_i  =  exp(-vote_expectation_weight |E[t_i]|) / sum_j exp(-vote_expectation_weight |E[t_k]|)`. Here `I_i`
        are the individual estimators and `E[t_i]` is the expectation value calculated by `calculate_expectation()`.

        Parameters
        ----------
        x : str or ndarray
            Sample of observations, or path to numpy file with observations, as saved by the
            `madminer.sampling.SampleAugmenter` functions. Note that this sample has to be sampled from the reference
            parameter where the score is estimated with the SALLY / SALLINO estimator!

        obs_weights : None or ndarray, optional
            Weights for the observations. If None, all events are taken to have equal weight. Default value: None.

        n_events : float, optional
            Expected number of events for which the kinematic Fisher information should be calculated. Default value: 1.

        mode : {"score", "information"}, optional
            If mode is "information", the Fisher information for each estimator is calculated individually and only then
            are the sample mean and covariance calculated. If mode is "score", the sample mean is
            calculated for the score for each event. Default value: "score".

        uncertainty : {"ensemble", "expectation", "sum", "none"}, optional
            How the covariance matrix of the Fisher information estimate is calculate. With "ensemble", the ensemble
            covariance is used (only supported if mode is "information"). With "expectation", the expectation of the
            score is used as a measure of the uncertainty of the score estimator, and this uncertainty is propagated
            through to the covariance matrix. With "sum", both terms are summed (only supported if mode is
            "information"). With "none", no uncertainties are calculated. Default value: "ensemble".

        vote_expectation_weight : float or list of float or None, optional
            If mode is "information", this factor determines how much more weight is given to those estimators with
            small expectation value (as calculated by `calculate_expectation()`). If a list is given, results are
            returned for each element in the list. If None, or if `calculate_expectation()` has not been called, all
            estimators are treated equal. Default value: None.

        return_individual_predictions : bool, optional
            If mode is "information", sets whether the individual estimator predictions are returned. Default value:
            False.

        sum_events : bool, optional
            If True or mode is "information", the expected Fisher information summed over the events x is calculated.
            If False and mode is "score", the per-event Fisher information for each event is returned. Default value:
            True.

        Returns
        -------
        mean_prediction : ndarray or list of ndarray
            Expected kinematic Fisher information matrix with shape `(n_events, n_parameters, n_parameters)` if
            sum_events is False and mode is "score", or `(n_parameters, n_parameters)` in any other case.

        covariance : ndarray or list of ndarray
            The covariance matrix of the Fisher information estimate. Its definition depends on the value of
            uncertainty; by default, the covariance is defined as the ensemble covariance (only supported if mode is
            "information"). This object has four indices, `cov_(ij)(i'j')`, ordered as i j i' j'. It has shape
            `(n_parameters, n_parameters, n_parameters, n_parameters)`. If more then one value vote_expectation_weight
            is given, this is a list with results for all entries in vote_expectation_weight.

        weights : ndarray or list of ndarray
            Only returned if return_individual_predictions is True. The estimator weights `w_i`. If more then one value
            vote_expectation_weight is given, this is a list with results for all entries in vote_expectation_weight.

        individual_predictions : ndarray
            Only returned if return_individual_predictions is True. The individual estimator predictions.

        """
        logger.debug("Evaluating Fisher information for %s estimators in ensemble", self.n_estimators)

        # Check input
        if mode not in ["score", "information"]:
            raise ValueError("Unknown mode {}, has to be 'score' or 'information'!".format(mode))

        if mode == "score":
            vote_expectation_weight = None

        if uncertainty == "expectation" or uncertainty == "sum":
            if self.expectations is None:
                raise RuntimeError(
                    "Expectations have not been calculated, cannot use uncertainty mode 'expectation' " "or 'sum'!"
                )

        # Calculate estimator_weights of each estimator in vote
        if self.expectations is None or vote_expectation_weight is None:
            estimator_weights = [np.ones(self.n_estimators)]
        else:
            if len(self.expectations.shape) == 1:
                expectations_norm = self.expectations
            elif len(self.expectations.shape) == 2:
                expectations_norm = np.linalg.norm(self.expectations, axis=1)
            else:
                expectations_norm = [np.linalg.norm(expectation) for expectation in self.expectations]

            if not isinstance(vote_expectation_weight, list):
                vote_expectation_weight = [vote_expectation_weight]

            estimator_weights = []
            for vote_weight in vote_expectation_weight:
                if vote_weight is None:
                    these_weights = np.ones(self.n_estimators)
                else:
                    these_weights = np.exp(-vote_weight * expectations_norm)
                these_weights /= np.sum(these_weights)
                estimator_weights.append(these_weights)

        logger.debug("  Estimator estimator_weights: %s", estimator_weights)

        predictions = []

        # "information" mode
        if mode == "information":
            # Calculate estimator predictions
            for i, estimator in enumerate(self.estimators):
                logger.debug("Starting evaluation for estimator %s / %s in ensemble", i + 1, self.n_estimators)

                predictions.append(estimator.calculate_fisher_information(x=x, weights=obs_weights, n_events=n_events))
            predictions = np.array(predictions)

            # Calculate weighted means and covariance matrices
            means = []
            ensemble_covariances = []

            for these_weights in estimator_weights:
                mean = np.average(predictions, axis=0, weights=these_weights)
                means.append(mean)

                predictions_flat = predictions.reshape((predictions.shape[0], -1))
                covariance = np.cov(predictions_flat.T, aweights=these_weights)
                covariance_shape = (
                    predictions.shape[1],
                    predictions.shape[2],
                    predictions.shape[1],
                    predictions.shape[2],
                )
                covariance = covariance.reshape(covariance_shape)

                ensemble_covariances.append(covariance)

        # "score" mode:
        else:
            # Load training data
            if isinstance(x, six.string_types):
                x = load_and_check(x)
            n_samples = x.shape[0]

            # Calculate score predictions
            score_predictions = []
            for i, estimator in enumerate(self.estimators):
                logger.debug("Starting evaluation for estimator %s / %s in ensemble", i + 1, self.n_estimators)

                score_predictions.append(estimator.evaluate(x=x))
                logger.debug("Estimator %s predicts t(x) = %s for first event", i + 1, score_predictions[-1][0, :])
            score_predictions = np.array(score_predictions)  # (n_estimators, n_events, n_parameters)

            # Get ensemble mean and ensemble covariance
            score_mean = np.mean(score_predictions, axis=0)  # (n_events, n_parameters)

            # For uncertainty calculation: calculate points betweeen mean and original predictions with same mean and
            # variance / n compared to the original predictions
            score_shifted_predictions = (score_predictions - score_mean[np.newaxis, :, :]) / self.n_estimators ** 0.5
            score_shifted_predictions = score_mean[np.newaxis, :, :] + score_shifted_predictions

            # Event weights
            if obs_weights is None:
                obs_weights = np.ones(n_samples)
            obs_weights /= np.sum(obs_weights)

            # Fisher information prediction (based on mean scores)
            if sum_events:
                information_mean = float(n_events) * np.sum(
                    obs_weights[:, np.newaxis, np.newaxis]
                    * score_mean[:, :, np.newaxis]
                    * score_mean[:, np.newaxis, :],
                    axis=0,
                )
            else:
                information_mean = (
                    float(n_events)
                    * obs_weights[:, np.newaxis, np.newaxis]
                    * score_mean[:, :, np.newaxis]
                    * score_mean[:, np.newaxis, :]
                )
            means = [information_mean]

            # Fisher information predictions based on shifted scores
            informations_shifted = float(n_events) * np.sum(
                obs_weights[np.newaxis, :, np.newaxis, np.newaxis]
                * score_shifted_predictions[:, :, :, np.newaxis]
                * score_shifted_predictions[:, :, np.newaxis, :],
                axis=1,
            )  # (n_estimators, n_parameters, n_parameters)

            n_params = score_mean.shape[1]
            informations_shifted = informations_shifted.reshape(-1, n_params ** 2)
            information_cov = np.cov(informations_shifted.T)
            information_cov = information_cov.reshape(n_params, n_params, n_params, n_params)
            ensemble_covariances = [information_cov]

            # Let's check the expected score
            expected_score = [np.einsum("n,ni->i", obs_weights, score_mean)]
            logger.debug("Expected per-event score (should be close to zero):\n%s", expected_score)

        # Calculate uncertainty through non-zero score expectation
        expectation_covariances = None
        if uncertainty == "expectation" or uncertainty == "sum":
            expectation_covariances = []
            for these_weights, expectation in zip(estimator_weights, self.expectations):
                mean_expectation = np.average(expectation, weights=these_weights, axis=0)
                expectation_covariances.append(
                    n_events
                    * np.einsum("a,b,c,d->abcd", mean_expectation, mean_expectation, mean_expectation, mean_expectation)
                )

        # Final covariances
        if uncertainty == "ensemble":
            covariances = ensemble_covariances
        elif uncertainty == "expectation":
            covariances = expectation_covariances
        elif uncertainty == "sum":
            covariances = [cov1 + cov2 for cov1, cov2 in zip(ensemble_covariances, expectation_covariances)]
        elif uncertainty == "none":
            covariances = [None for cov in ensemble_covariances]
        else:
            raise ValueError("Unknown uncertainty mode {}".format(uncertainty))

        # Returns
        if len(estimator_weights) == 1:
            if return_individual_predictions and mode == "information":
                return means[0], covariances[0], estimator_weights[0], predictions
            return means[0], covariances[0]

        if return_individual_predictions and mode == "information":
            return means, covariances, estimator_weights, predictions
        return means, covariances

    def save(self, folder, save_model=False):
        """
        Saves the estimator ensemble to a folder.

        Parameters
        ----------
        folder : str
            Path to the folder.

        save_model : bool, optional
            If True, the whole model is saved in addition to the state dict. This is not necessary for loading it
            again with EnsembleForge.load(), but can be useful for debugging, for instance to plot the computational
            graph.

        Returns
        -------
            None

        """

        # Check paths
        create_missing_folders([folder])

        # Save ensemble settings
        logger.debug("Saving ensemble setup to %s/ensemble.json", folder)

        if self.expectations is None:
            expectations = "None"
        else:
            expectations = self.expectations.tolist()

        settings = {"n_estimators": self.n_estimators, "expectations": expectations}

        with open(folder + "/ensemble.json", "w") as f:
            json.dump(settings, f)

        # Save estimators
        for i, estimator in enumerate(self.estimators):
            estimator.save(folder + "/estimator_" + str(i), save_model=save_model)

    def load(self, folder):
        """
        Loads the estimator ensemble from a folder.

        Parameters
        ----------
        folder : str
            Path to the folder.

        Returns
        -------
            None

        """

        # Load ensemble settings
        logger.debug("Loading ensemble setup from %s/ensemble.json", folder)

        with open(folder + "/ensemble.json", "r") as f:
            settings = json.load(f)

        self.n_estimators = settings["n_estimators"]
        self.expectations = settings["expectations"]
        if self.expectations == "None":
            self.expectations = None
        if self.expectations is not None:
            self.expectations = np.array(self.expectations)

        logger.info("Found ensemble with %s estimators and expectations %s", self.n_estimators, self.expectations)

        # Load estimators
        self.estimators = []
        for i in range(self.n_estimators):
            estimator = MLForge()
            estimator.load(folder + "/estimator_" + str(i))
            self.estimators.append(estimator)

        # Check consistency and update n_parameters, n_observables
        self._check_consistency()

    def _check_consistency(self, keywords=None):
        """
        Internal function that checks if all estimators belong to the same category
        (local score regression, single-parameterized likelihood ratio estimator,
        doubly parameterized likelihood ratio estimator).

        Parameters
        ----------
        keywords : dict or None, optional
            kwargs passed to `train_one()` or `train_all()`.

        Returns
        -------
        method_type : {"local_score", "parameterized", "doubly_parameterized"}
            Method type of this ensemble.

        Raises
        ------
        RuntimeError
            Estimators are inconsistent.

        """
        # Accumulate methods of all estimators
        methods = [estimator.method for estimator in self.estimators]
        all_n_parameters = [estimator.n_parameters for estimator in self.estimators]
        all_n_observables = [estimator.n_observables for estimator in self.estimators]

        if keywords is not None:
            keyword_method = keywords.get("method", None)
            if isinstance(keyword_method, list):
                methods += keyword_method
            else:
                methods.append(keyword_method)

        # Check consistency of methods
        self.method_type = None

        for method in methods:
            if method in ["sally", "sallino"]:
                this_method_type = "local_score"
            elif method in ["carl", "rolr", "rascal", "alice", "alices", "nde", "scandal"]:
                this_method_type = "parameterized"
            elif method in ["carl2", "rolr2", "rascal2", "alice2", "alices2"]:
                this_method_type = "doubly_parameterized"
            elif method is None:
                continue
            else:
                raise RuntimeError("Unknown method %s", method)

            if self.method_type is None:
                self.method_type = this_method_type

            if self.method_type != this_method_type:
                raise RuntimeError(
                    "Ensemble with inconsistent estimator methods! All methods have to be either"
                    " single-parameterized ratio estimators, doubly parameterized ratio estimators,"
                    " or local score estimators. Found methods " + ", ".join(methods) + "."
                )

        # Check consistency of parameter and observable numnbers
        self.n_parameters = None
        self.n_observables = None

        for estimator_n_parameters, estimator_n_observables in zip(all_n_parameters, all_n_observables):
            if self.n_parameters is None:
                self.n_parameters = estimator_n_parameters
            if self.n_observables is None:
                self.n_observables = estimator_n_observables

            if self.n_parameters is not None and self.n_parameters != estimator_n_parameters:
                raise RuntimeError(
                    "Ensemble with inconsistent numbers of parameters for different estimators: %s", all_n_parameters
                )
            if self.n_observables is not None and self.n_observables != estimator_n_observables:
                raise RuntimeError(
                    "Ensemble with inconsistent numbers of parameters for different estimators: %s", all_n_observables
                )

        # Return method type of ensemble
        return self.method_type
