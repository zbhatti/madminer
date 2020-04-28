from os import path, environ
from sys import argv
from glob import glob

from collections import namedtuple
import logging

# headless
from matplotlib import use
use("Agg")

from matplotlib import pyplot as plt

import numpy as np
from random import gauss

from madminer.core import MadMiner
from madminer.lhe import LHEReader
from madminer.utils.particle import MadMinerParticle
from madminer.plotting import plot_distributions
from madminer.sampling import combine_and_shuffle, SampleAugmenter, benchmark, benchmarks
from madminer.ml import ParameterizedRatioEstimator

Benchmark = namedtuple('Benchmark', ['mass', 'width', 'width_exp', 'name'])


# g g > h > e+ e- mu+ mu-
# p idx:    0  1  2   3
def obs_function(particles, leptons, photons, jets, met):
    positron, electron, antimuon, muon = particles


class EventRunner:
    def __init__(self, data_dir):

        theta_exp = np.linspace(np.log10(10**-5), np.log10(2), 21)  # to be exponents of base 10
        self.theta_test = 10**np.linspace(theta_exp[0], theta_exp[-1], 40)

        theta1_idx = np.argmin(np.abs(10**theta_exp - 1.0))  # closest to 1 GeV; UNUSED?
        theta1_width_exp = np.log10(1.0)
        theta1_width = 10**theta1_width_exp

        expected_mass, expected_width, expected_width_exp = 126.0, 10**np.log10(4E-3), np.log10(4E-3)

        self.expected_benchmark = Benchmark(expected_mass, expected_width, expected_width_exp, '126_{:.1E}'.format(expected_width))
        self.theta0_benchmarks = [Benchmark(expected_mass, 10**t, t, '126_{:.1E}'.format(10**t)) for t in theta_exp]
        self.theta1_benchmark = Benchmark(expected_mass, theta1_width, theta1_width_exp, '126_{:.1E}'.format(theta1_width))

        self.low_sample_benchmark_names = [cb.name for cb in self.theta0_benchmarks]
        self.high_sample_benchmark_names = [self.theta1_benchmark.name]

        self.data_dir = data_dir
        self.miner_setup_path = path.join(self.data_dir,  'data/miner_setup.h5')

    #  run solo
    def build_setup(self):
        miner = MadMiner()

        logging.info('running miner...')

        miner.add_parameter(
            lha_block=str('DECAY'),
            lha_id=25,
            parameter_name=str('HIGGS_WIDTH'),
            param_card_transform="10**theta",
        )

        # add scanning points
        for b in self.theta0_benchmarks + [self.theta1_benchmark] + [self.expected_benchmark]:
            miner.add_benchmark({'HIGGS_WIDTH': b.width_exp}, b.name)

        miner.save(self.miner_setup_path)

    # run concurrently
    def generate_events(self, few_or_many, worker_id):
        mg_dir = environ.get('MADGRAPH_DIR', '/home/zbhatti/util/MG5_aMC_v2_6_5')
        data_dir = self.data_dir
        config_dir = path.dirname(__file__)

        # shared inputs:
        proc_card_path = path.join(config_dir, 'cards/higgs_4l_proc_card.dat')
        param_template_path = path.join(config_dir, 'cards/param_card_template.dat')

        if few_or_many == 'few':
            run_card_path = path.join(config_dir, 'cards/higgs_4l_few_run_card.dat')
            sample_benchmarks = self.low_sample_benchmark_names

        elif few_or_many == 'many':
            run_card_path = path.join(config_dir, 'cards/higgs_4l_many_run_card.dat')
            sample_benchmarks = self.high_sample_benchmark_names

        else:
            run_card_path = None
            sample_benchmarks = None
            logging.error('few or many must be specified')
            return

        if not path.exists(proc_card_path) or not path.exists(param_template_path) or not path.exists(run_card_path):
            logging.error('some file is missing')
            return

        # unique outputs:
        mg_process_directory = path.join(data_dir, 'mg_processes_{}_{}/signal'.format(few_or_many, worker_id))
        log_directory = path.join(data_dir, 'logs_{}_{}/signal'.format(few_or_many, worker_id))
        event_data_path = path.join(data_dir, 'data/miner_lhe_data_{}_{}.h5'.format(few_or_many, worker_id))

        miner = MadMiner()
        miner.load(self.miner_setup_path)
        #
        # miner.run_multiple(
        #     sample_benchmarks=sample_benchmarks,
        #     mg_directory=mg_dir,
        #     mg_process_directory=mg_process_directory,
        #     proc_card_file=proc_card_path,
        #     param_card_template_file=param_template_path,
        #     run_card_files=[run_card_path],
        #     log_directory=log_directory,
        #     run_card_override={'iseed': worker_id},
        # )

        logging.info('running LHEProcessor...')
        run_smearing = False

        # name: definition
        final_state = {
            'e+': 'p[0]',
            'e-': 'p[1]',
            'mu+': 'p[2]',
            'mu-': 'p[3]',
        }

        proc = LHEReader(self.miner_setup_path)
        i = 1
        for sample_bench in sample_benchmarks:
            lhe_filename = path.join(self.data_dir, 'mg_processes_{0}_{1}/signal/Events/run_{2:0>2}/unweighted_events.lhe.gz'.format(few_or_many, worker_id, i))

            proc.add_sample(
                lhe_filename=lhe_filename,
                sampled_from_benchmark=sample_bench,
                is_background=False,
                k_factor=1.0,
            )
            i += 1

        for name, definition in final_state.items():
            proc.add_observable(
                str('{0}_E'.format(name)),
                str('{0}.e'.format(definition)),
                required=True
            )

            proc.add_observable(
                str('{0}_pt'.format(name)),
                str('{0}.pt'.format(definition)),
                required=True
            )

            # angle in plane perpendicular to collision
            proc.add_observable(
                str('{0}_eta'.format(name)),
                str('{0}.eta'.format(definition)),
                required=True
            )

            # angle from collision line to transverse plane, 0 - inf. 0 is perpendicular and inf is parallel
            proc.add_observable(
                str('{0}_phi'.format(name)),
                str('{0}.phi()'.format(definition)),
                required=True
            )

        proc.add_observable(
            'mass_e+_e-',
            '({0} + {1}).m'.format(final_state['e+'], final_state['e-']),
            required=True,
        )

        proc.add_observable(
            'mass_mu+_mu-',
            '({0} + {1}).m'.format(final_state['mu+'], final_state['mu-']),
            required=True,
        )

        proc.add_observable(
            'mass_4l',
            '({0} + {1} + {2} + {3}).m'.format(final_state['e+'], final_state['e-'], final_state['mu+'], final_state['mu-']),
            required=True,
        )

        proc.add_cut('mass_4l >= 120')
        proc.add_cut('mass_4l <= 130')

        if run_smearing:
            logging.info('running with smearing')

            # charged lepton smearing is minimal since semiconductor based detection works well
            proc.set_smearing(
                pdgids=[13, -13, 11, -11],
                energy_resolution_abs=0.,
                energy_resolution_rel=0.05,
                pt_resolution_abs=None,
                pt_resolution_rel=None,
                eta_resolution_abs=0,
                eta_resolution_rel=0,
                phi_resolution_abs=0,
                phi_resolution_rel=0,
            )

        proc.analyse_samples(parse_events_as_xml=True)
        proc.save(event_data_path)

        logging.info(proc.observables)
        logging.info(proc.observations.keys())

    # run solo
    def merge_and_train(self):

        miner_data_file_pattern = path.join(self.data_dir, 'data/miner_lhe_data_*_*.h5')
        miner_data_shuffled_path = path.join(self.data_dir, 'data/miner_lhe_data_shuffled.h5')
        n_train_events = 20000000
        n_val_events = 4000000
        n_test_events = 100000

        miner_data_file_paths = glob(miner_data_file_pattern)

        logging.info('shuffling LHE files {}'.format(miner_data_file_paths))
        combine_and_shuffle(miner_data_file_paths, miner_data_shuffled_path)

        sa = SampleAugmenter(miner_data_shuffled_path)
        set(sa.observables.keys())
        # TODO: new method
        # plot observables for shuffled elements, sample 1,000,000 events for example
        _ = plot_distributions(
            filename=miner_data_shuffled_path,
            observables=list(set(sa.observables.keys()) - {'mass_4l'}),
            uncertainties='none',
            n_bins=20,
            n_cols=5,
            normalize=True,
            parameter_points=[cb.name for cb in self.theta0_benchmarks[::5]] + [self.theta1_benchmark.name],
            linestyles='-',
            sample_only_from_closest_benchmark=True,
            n_events=1000000,
        )
        plt.tight_layout()
        plt.savefig(path.join(self.data_dir, 'observables_histograms.png'), bbox_inches='tight')

        _ = plot_distributions(
            filename=miner_data_shuffled_path,
            observables=['mass_4l'],
            uncertainties='none',
            n_bins=50,
            normalize=True,
            parameter_points=[cb.name for cb in self.theta0_benchmarks[::5]] + [self.theta1_benchmark.name],
            linestyles='-',
            sample_only_from_closest_benchmark=True,
            n_events=1000000,
            quantiles_for_range=(0.025, 0.85),
        )
        plt.tight_layout()
        plt.savefig(path.join(self.data_dir, 'mass_4l_histogram.png'), bbox_inches='tight')

        logging.info('running SampleAugmenter...')

        train_result = sa.sample_train_ratio(
            theta0=benchmarks([b.name for b in self.theta0_benchmarks]),
            theta1=benchmark(self.theta1_benchmark.name),
            n_samples=n_train_events,
            sample_only_from_closest_benchmark=True,
            partition='train',
            folder=path.join(self.data_dir, 'data/samples'),
            filename='train',
            return_individual_n_effective=True
        )
        logging.info('train n_effective:, {}'.format(train_result[-1]))
        #
        validation_result = sa.sample_train_ratio(
            theta0=benchmarks([b.name for b in self.theta0_benchmarks]),
            theta1=benchmark(self.theta1_benchmark.name),
            n_samples=n_val_events,
            sample_only_from_closest_benchmark=True,
            partition='validation',
            folder=path.join(self.data_dir, 'data/samples'),
            filename='valid',
            return_individual_n_effective=True
        )
        logging.info('validation n_effective:, {}'.format(validation_result[-1]))

        _0 = sa.sample_test(
            theta=benchmark(self.expected_benchmark.name),
            n_samples=n_test_events,
            folder=path.join(self.data_dir, 'data/samples'),
            filename='test',
        )

        thetas_benchmarks, xsecs_benchmarks, xsec_errors_benchmarks = sa.cross_sections(
            theta=benchmarks([b.name for b in self.theta0_benchmarks])
        )

        logging.info('effective_n_samples train and validation: {} and {}'.format(train_result[-1], validation_result[-1]))
        logging.info(str(xsecs_benchmarks))

        # forge.train
        forge = ParameterizedRatioEstimator(n_hidden=(100, 100))
        # forge.load(path.join(self.data_dir, 'models/alice'))
        logging.info('running forge')
        x_train_path = path.join(self.data_dir, 'data/samples/x_train.npy')
        y_train_path = path.join(self.data_dir, 'data/samples/y_train.npy')
        theta0_train_path = path.join(self.data_dir, 'data/samples/theta0_train.npy')
        r_xz_train_path = path.join(self.data_dir, 'data/samples/r_xz_train.npy')

        x_validation_path = path.join(self.data_dir, 'data/samples/x_valid.npy')
        y_validation_path = path.join(self.data_dir, 'data/samples/y_valid.npy')
        theta0_validation_path = path.join(self.data_dir, 'data/samples/theta0_valid.npy')
        r_xz_validation_path = path.join(self.data_dir, 'data/samples/r_xz_valid.npy')

        result = forge.train(method='alice',
                             x=x_train_path,
                             y=y_train_path,
                             theta=theta0_train_path,
                             r_xz=r_xz_train_path,
                             x_val=x_validation_path,
                             y_val=y_validation_path,
                             theta_val=theta0_validation_path,
                             r_xz_val=r_xz_validation_path,
                             n_epochs=30,
                             batch_size=100,
                             initial_lr=0.001,
                             final_lr=1e-6,
                             scale_inputs=True,
                             )

        forge.save(path.join(self.data_dir, 'models/alice'))

        # Test the model
        theta_ref = np.array([[c.mass, c.width] for c in self.theta0_benchmarks])
        np.save(path.join(self.data_dir, 'data/samples/theta_ref.npy'), theta_ref)  # UNUSED?

        mass_bins = np.array([self.expected_benchmark.mass, ])
        width_bins = np.array(self.theta_test)
        mass, width = np.meshgrid(mass_bins, width_bins)
        logging.info('mass: {0}, width: {1}'.format(mass, width))

        # theta parameters, no mass:
        test_grid = np.vstack((width.flatten(), )).T
        np.save(path.join(self.data_dir, 'data/samples/test_grid.npy'), test_grid)

        log_r_hat, _0 = forge.evaluate(
            theta=path.join(self.data_dir, 'data/samples/test_grid.npy'),
            x=path.join(self.data_dir, 'data/samples/x_test.npy'),
            test_all_combinations=True,
            evaluate_score=False,
        )

        np.save(path.join(self.data_dir, 'data/samples/log_r_hat.npy'), log_r_hat)

        # plot final results
        mean_log_r_hat = np.mean(log_r_hat, axis=1)
        llr = -2 * mean_log_r_hat
        best_fit_i = np.argmin(llr)
        best_fit_x_y = test_grid[best_fit_i]

        logging.info('best_fit {}'.format(best_fit_x_y))
        logging.info('llr: {}'.format(llr))

        logging.info('plotting...')

        fig = plt.figure(figsize=(6, 5))
        plt.plot(test_grid[:, 0], llr, marker='o', ls=' ', zorder=1)
        plt.scatter(best_fit_x_y[0], llr[best_fit_i], s=100., color='red', marker='*', zorder=2)
        plt.xlabel(r'$Width (GeV)$')
        plt.ylabel(r'$Likelihood Ratio -2logp(x|\theta)$')
        plt.xscale('log')
        plt.savefig(path.join(self.data_dir, 'llr.png'), bbox_inches='tight')
        logging.info('')


def setup_logging():
    # MadMiner output
    logging.basicConfig(
        format='%(asctime)-5.5s %(name)-20.20s %(levelname)-7.7s %(message)s',
        datefmt='%H:%M',
        level=logging.DEBUG
    )

    # Output of all other modules (e.g. matplotlib)
    for key in logging.Logger.manager.loggerDict:
        if "madminer" not in key:
            logging.getLogger(key).setLevel(logging.DEBUG)


def main():
    setup_logging()
    logging.info('args: [data_dir] {setup, generate {few, many} [worker_id]}, train')
    logging.info('args: {}'.format(argv))
    working_dir = argv[1]
    if argv[2] == 'setup':
        EventRunner(working_dir).build_setup()

    elif argv[2] == 'generate':
        few_or_many = argv[3]  # few or many
        worker_id = argv[4]  # 0, 1, 2, ... 99, etc
        EventRunner(working_dir).generate_events(few_or_many, worker_id)

    elif argv[2] == 'train':
        EventRunner(working_dir).merge_and_train()


if __name__ == '__main__':
    main()
