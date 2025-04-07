import parameter_estimation
import torch
from time_series_ddpm_model import TimeSeriesDDPM
import time
import os
import numpy as np
import utils


class Experiment:
    def __init__(self, name: 'str', experiment_type: 'str', directory_path, training_data, network_architecture):
        self.simulated_data_ma_coefficient = None
        self.simulated_data_ar_coefficient = None
        self.simulated_data_kurtosis = None
        self.training_data_ma_coefficient = None
        self.nn_weights = None
        self.generation_time = None
        self.ddpm_model = None
        self.training_data_ar_coefficient = None
        self.training_data_kurtosis = None
        self.training_time = None
        self.ddpm_simulated_data = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.directory_path = directory_path
        self.name = name
        self.experiment_type = experiment_type
        self.training_data = training_data
        self.network_architecture = network_architecture
        self.num_samples = training_data.shape[0]
        self.sequence_length = training_data.shape[1]

    def execute(self, epochs=100):
        # 1. Estimate parameters of the training data
        self.training_data_kurtosis = parameter_estimation.ts_kurtosis_estimation(self.training_data)
        if self.experiment_type == 'AR':
            self.training_data_ar_coefficient = parameter_estimation.AR_1_phi_parameter_estimation(
                self.training_data.numpy())
        elif self.experiment_type == 'ARMA':
            self.training_data_ar_coefficient, self.training_data_ma_coefficient = (
                parameter_estimation.ARMA_11_phi_theta_parameters_estimation(self.training_data.numpy()))

        # 2. Train neural network
        self.ddpm_model = TimeSeriesDDPM(net=self.network_architecture, T=1000)
        start_training_time = time.perf_counter()
        self.ddpm_model.train_model(self.training_data, num_epochs=epochs)
        end_training_time = time.perf_counter()
        self.training_time = end_training_time - start_training_time
        # save trained neural network weights
        self.nn_weights = self.ddpm_model.state_dict()

        # 3. Generate synthetic data
        start_generation_time = time.perf_counter()
        self.ddpm_simulated_data = self.ddpm_model.sample(seq_length=self.sequence_length,
                                                          num_samples=self.num_samples, device=self.device)
        end_generation_time = time.perf_counter()
        self.generation_time = end_generation_time - start_generation_time

        # 4. Estimate parameters of synthetic data
        self.simulated_data_kurtosis = parameter_estimation.ts_kurtosis_estimation(self.ddpm_simulated_data)
        if self.experiment_type == 'AR':
            self.simulated_data_ar_coefficient = parameter_estimation.AR_1_phi_parameter_estimation(
                self.ddpm_simulated_data)
        elif self.experiment_type == 'ARMA':
            self.simulated_data_ar_coefficient, self.simulated_data_ma_coefficient = (
                parameter_estimation.ARMA_11_phi_theta_parameters_estimation(self.ddpm_simulated_data))

    def export(self, save_training_data=False, save_simulated_data=False):
        # 1. Export neural network weights:
        torch.save(self.nn_weights, os.path.join(utils.ensure_directory(self.directory_path), self.name + '_model_weights.pth'))

        # 2. Export neural network architecture:
        with open(os.path.join(utils.ensure_directory(self.directory_path), self.name + '_nn_architecture.csv'), "w") as file:
            print(self.ddpm_model, file=file)

        # 3. Export training data:
        if save_training_data:
            np.savetxt(os.path.join(utils.ensure_directory(self.directory_path), self.name + '_training_data.csv'),
                       self.training_data.numpy(),
                       delimiter=",")

        # 4. Export simulated data:
        if save_simulated_data:
            np.savetxt(os.path.join(utils.ensure_directory(self.directory_path), self.name + '_simulated_data.csv'),
                       self.ddpm_simulated_data.numpy(),
                       delimiter=",")

        # 5. Export estimated training data parameters:
        np.savetxt(os.path.join(utils.ensure_directory(self.directory_path), self.name + '_training_kurtosis_estimations.csv'),
                   self.training_data_kurtosis,
                   delimiter=",")
        if self.experiment_type == 'AR':
            np.savetxt(os.path.join(utils.ensure_directory(self.directory_path), self.name + '_training_ar_estimations.csv'),
                       self.training_data_ar_coefficient,
                       delimiter=",")
        elif self.experiment_type == 'ARMA':
            np.savetxt(os.path.join(utils.ensure_directory(self.directory_path), self.name + '_training_ar_estimations.csv'),
                       self.training_data_ar_coefficient,
                       delimiter=",")
            np.savetxt(os.path.join(utils.ensure_directory(self.directory_path), self.name + '_training_ma_estimations.csv'),
                       self.training_data_ma_coefficient,
                       delimiter=",")

        # 6. Export estimated simulated data parameters:
        np.savetxt(os.path.join(utils.ensure_directory(self.directory_path), self.name + '_simulated_kurtosis_estimations.csv'),
                   self.simulated_data_kurtosis,
                   delimiter=",")
        if self.experiment_type == 'AR':
            np.savetxt(os.path.join(utils.ensure_directory(self.directory_path), self.name + '_simulated_ar_estimations.csv'),
                       self.simulated_data_ar_coefficient,
                       delimiter=",")
        elif self.experiment_type == 'ARMA':
            np.savetxt(os.path.join(utils.ensure_directory(self.directory_path), self.name + '_simulated_ar_estimations.csv'),
                       self.simulated_data_ar_coefficient,
                       delimiter=",")
            np.savetxt(os.path.join(utils.ensure_directory(self.directory_path), self.name + '_simulated_ma_estimations.csv'),
                       self.simulated_data_ma_coefficient,
                       delimiter=",")

        # 7. Export execution times
        with open(os.path.join(utils.ensure_directory(self.directory_path), self.name + '_execution_times.txt'), "w") as file:
            print(f'Training time: {self.training_time}, Generation time: {self.generation_time}', file=file)

        # 8. Export histograms
        utils.comparison_histograms(self.training_data_kurtosis, self.simulated_data_kurtosis, 100,
                                    'training kurtosis', 'simulated kurtosis',
                                    f'Training vs Simulated Kurtosis ({self.name})', export=True,
                                    path=os.path.join(utils.ensure_directory(self.directory_path),
                                                      self.name + '_training_vs_simulated_kurtosis.jpg'))
        if self.experiment_type == 'AR':
            utils.comparison_histograms(self.training_data_ar_coefficient, self.simulated_data_ar_coefficient, 100,
                                        'training ar coefficient', 'simulated ar coefficient',
                                        f'Training vs Simulated AR coefficient ({self.name})', export=True,
                                        path=os.path.join(utils.ensure_directory(self.directory_path),
                                                          self.name + '_training_vs_simulated_ar_coefficient.jpg'))
        elif self.experiment_type == 'ARMA':
            utils.comparison_histograms(self.training_data_ar_coefficient, self.simulated_data_ar_coefficient, 100,
                                        'training ar coefficient', 'simulated ar coefficient',
                                        f'Training vs Simulated AR coefficient ({self.name})', export=True,
                                        path=os.path.join(utils.ensure_directory(self.directory_path),
                                                          self.name + '_training_vs_simulated_ar_coefficient.jpg'))
            utils.comparison_histograms(self.training_data_ma_coefficient, self.simulated_data_ma_coefficient, 100,
                                        'training ma coefficient', 'simulated ma coefficient',
                                        f'Training vs Simulated MA coefficient ({self.name})', export=True,
                                        path=os.path.join(utils.ensure_directory(self.directory_path),
                                                          self.name + '_training_vs_simulated_ma_coefficient.jpg'))
