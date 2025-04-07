import os
import random
import sys
import numpy as np
import network_architectures
import utils
from experiment import Experiment
from time_series_generator import TimeSeriesGenerator


class Logger:
    def __init__(self, filename="log.txt"):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        # This is needed for compatibility with some environments
        self.terminal.flush()
        self.log.flush()


def run_experiment(num_series, seq_length, seed=1):
    # 0. Set seed for replication
    random.seed(seed)
    # 1. Generate training data
    ar_data = TimeSeriesGenerator.generate_ar_series(num_series, seq_length, 0.3, sigma=0.5, burnin=100)
    arma_data = TimeSeriesGenerator.generate_arma_series(num_series, seq_length, 0.2, 0.4, sigma=0.5, burnin=100)
    garch_data = TimeSeriesGenerator.generate_garch_11_series(num_series,
                                                              seq_length, 0.3, 0.5, 0.3, burnin=100)
    # 2. Save training data
    training_data_directory_path = utils.ensure_directory(utils.build_path('output', 'training_data'))
    np.savetxt(os.path.join(training_data_directory_path, 'AR_training_data.csv'), ar_data.numpy()[:, :, 0],
               delimiter=",")
    np.savetxt(os.path.join(training_data_directory_path, 'ARMA_training_data.csv'), arma_data.numpy()[:, :, 0],
               delimiter=",")
    np.savetxt(os.path.join(training_data_directory_path, 'GARCH_training_data.csv'), garch_data.numpy()[:, :, 0],
               delimiter=",")

    # 3. Set up experiments
    custom_lstm_net = network_architectures.LSTMNet(input_dim=1, hidden_dim=64, num_layers=2)
    custom_gru_net = network_architectures.GRUNet(input_dim=1, hidden_dim=64, num_layers=2)
    custom_CNN_net = network_architectures.CNNNet(input_dim=1, hidden_dims=[64, 64], kernel_size=3, padding=1)
    custom_FCNN_net = network_architectures.FCNNNet(input_dim=1, hidden_dims=[64, 128, 64])
    custom_HybridCNNLSTM_net = network_architectures.HybridCNNLSTMNet(input_dim=1, conv_channels=32, lstm_hidden_dim=64,
                                                                      num_lstm_layers=2, kernel_size=3, padding=1)
    architectures = {'lstm': custom_lstm_net,
                     'gru': custom_gru_net,
                     'CNN': custom_CNN_net,
                     'FCNN': custom_FCNN_net,
                     'hybridCNNLSTM': custom_HybridCNNLSTM_net}

    ts_models = {'AR': ar_data, 'ARMA': arma_data, 'GARCH': garch_data}

    for ts_model in ts_models:
        for architecture in architectures:
            print(f'Working on experiment for {ts_model} model and {architecture} architecture')
            exp_name = ts_model + '_' + architecture
            experiment = Experiment(name=exp_name,
                                    experiment_type=ts_model,
                                    directory_path=utils.build_path('output', exp_name),
                                    training_data=ts_models[ts_model],
                                    network_architecture=architectures[architecture])
            experiment.execute(epochs=300)
            experiment.export()


def main():
    sys.stdout = sys.stderr = Logger("log.log")
    run_experiment(500, 500, seed=1)


if __name__ == '__main__':
    main()
