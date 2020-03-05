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

Benchmark = namedtuple('Benchmark', ['mass', 'width', 'name'])


# http://inspirehep.net/record/1204335/files/ATLAS-CONF-2012-082.pdf (page 3)
# vis - b-jet and charged leptons
# invis - missing energy
def calc_mT(vis, invis):
    mT = (vis.m**2 + 2.*(vis.e * invis.e - vis.pt * invis.pt))**.5
    if np.isnan(mT):
        raise ValueError('mT not determined\nvis: {}\n invis: {}')
    return mT


# proc card: generate p p > t t~, (t > w+ b, w+ > mu+ vm), (t~ > w- b~, w- > mu- vm~)
# particles index:                        2       0   1             5        3   4
# particle id:                            5      -13  14           -5        13 -14
def mt2(particles, leptons, photons, jets, met):
    antimuon, muon_neutrino, b, muon, antimuon_neutrino, bbar = particles
    n_picks = 1000
    visible_sum_1 = MadMinerParticle()
    visible_sum_1.setpxpypze(0.0, 0.0, 0.0, 0.0)

    visible_sum_2 = MadMinerParticle()
    visible_sum_2.setpxpypze(0.0, 0.0, 0.0, 0.0)

    visible_sum_1 = b + antimuon
    visible_sum_2 = bbar + muon

    met_1 = MadMinerParticle()
    met_1.setpxpypze(-visible_sum_1.px, -visible_sum_1.py, np.NaN, visible_sum_1.pt)

    met_2 = MadMinerParticle()
    met_2.setpxpypze(-visible_sum_2.px, -visible_sum_2.py, np.NaN, visible_sum_2.pt)

    met_sum = met_1 + met_2

    max_mT_list = []
    nu1 = MadMinerParticle()
    nu2 = MadMinerParticle()
    for i in range(n_picks):
        # met_sum.px == nu1.px + nu2.px
        nu1px = gauss(met_sum.px/2., met_sum.px)
        nu1py = gauss(met_sum.py/2., met_sum.py)
        nu1e = (nu1px**2 + nu1py**2)**.5

        nu2px = met_sum.px - nu1.px
        nu2py = met_sum.py - nu1.py
        nu2e = (nu2px**2 + nu2py**2)**.5

        nu1.setpxpypze(nu1px, nu1py, np.NaN, nu1e)
        nu2.setpxpypze(nu2px, nu2py, np.NaN, nu2e)

        mT_antimu = calc_mT(visible_sum_1, nu1)
        mT_mu = calc_mT(visible_sum_2, nu2)

        max_mT_list.append(max(mT_mu, mT_antimu))

    return min(max_mT_list)


class EventRunner:
    def __init__(self, data_dir):

        self.mass_low, self.mass_high = (160, 186)  # high is exclusive

        self.physics_benchmarks = [Benchmark(float(i), 1.5, '{0}_{1}'.format(i, 15)) for i in range(self.mass_low, self.mass_high)]
        self.expected_benchmark = Benchmark(172.0, 1.5, '172_15')

        self.wide_artificial_benchmarks = [Benchmark(float(i), 4.0, '{0}_{1}'.format(i, 40)) for i in
                                           range(self.mass_low, self.mass_high, 5)]

        self.wide_expected_benchmark = Benchmark(172.5, 4.0, '172.5_40')

        # self.low_sample_benchmark_names = [cb.name for cb in self.wide_artificial_benchmarks]
        self.low_sample_benchmark_names = [cb.name for cb in self.physics_benchmarks]
        self.high_sample_benchmark_names = [self.wide_expected_benchmark.name]

        self.data_dir = data_dir
        self.miner_setup_path = path.join(self.data_dir,  'data/miner_setup.h5')

    #  run solo
    def build_setup(self):
        miner = MadMiner()

        logging.info('running miner...')

        miner.add_parameter(
            lha_block=str('MASS'),
            lha_id=6,
            parameter_name=str('TOP_MASS'),
        )
        # miner.add_parameter(
        #     lha_block=str('DECAY'),
        #     lha_id=6,
        #     parameter_name=str('TOP_WIDTH'),
        # )

        # add scanning points
        # for b in self.physics_benchmarks + self.wide_artificial_benchmarks + [self.wide_expected_benchmark]:
        for b in self.physics_benchmarks + [self.wide_expected_benchmark]:
            # miner.add_benchmark({'TOP_MASS': b.mass, 'TOP_WIDTH': b.width}, b.name)
            miner.add_benchmark({'TOP_MASS': b.mass}, b.name)

        miner.save(self.miner_setup_path)

    # run concurrently
    def generate_events(self, few_or_many, worker_id):
        mg_dir = environ.get('MADGRAPH_DIR', '/home/zbhatti/util/MG5_aMC_v2_6_5')
        data_dir = self.data_dir
        config_dir = path.dirname(__file__)

        # shared inputs:
        proc_card_path = path.join(config_dir, 'cards/ttbar_proc_card.dat')
        param_template_path = path.join(config_dir, 'cards/param_card_template.dat')

        if few_or_many == 'few':
            run_card_path = path.join(config_dir, 'cards/ttbar_few_run_card.dat')
            sample_benchmarks = self.low_sample_benchmark_names

        elif few_or_many == 'many':
            run_card_path = path.join(config_dir, 'cards/ttbar_many_run_card.dat')
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

        miner.run_multiple(
            sample_benchmarks=sample_benchmarks,
            mg_directory=mg_dir,
            mg_process_directory=mg_process_directory,
            proc_card_file=proc_card_path,
            param_card_template_file=param_template_path,
            run_card_files=[run_card_path],
            log_directory=log_directory,
            run_card_override={'iseed': worker_id},
        )

        logging.info('running LHEProcessor...')
        run_smearing = False

        # name: definition
        obs_particles = {
            'mu_0': 'mu[0]',
            'mu_1': 'mu[1]',
            'j_0': 'j[0]',
            'j_1': 'j[1]',
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

        for name, definition in obs_particles.items():
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
            proc.add_cut('{0}.pt >= 25.0'.format(definition))

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

        # missing energy object's momentum in transverse direction
        proc.add_observable(
            str('met_pt'),
            str('met.pt'),
            required=True
        )

        proc.add_observable(
            str('met_phi'),
            str('met.phi()'),
            required=False
        )

        proc.add_cut('met.pt >= 25.0')

        charged_leptons = ['mu_0', 'mu_1']
        bjets = ['j_0', 'j_1']
        for lep in charged_leptons:
            for bj in bjets:
                proc.add_observable(
                    str('m_{0}_{1}'.format(lep, bj)),
                    str('sqrt( ({0}.e + {1}.e) ** 2 - ({0}.pt + {1}.pt) ** 2)'.format(
                        obs_particles[lep], obs_particles[bj])),
                    required=True,
                )

        proc.add_observable_from_function('mt2', mt2, required=True)

        if run_smearing:
            logging.info('running with smearing')

            # Partons giving rise to jets lead to muddier results
            proc.set_smearing(
                pdgids=[5, -5],
                energy_resolution_abs=0.,
                energy_resolution_rel=0.1,
                pt_resolution_abs=None,
                pt_resolution_rel=None,
                eta_resolution_abs=0,
                eta_resolution_rel=0,
                phi_resolution_abs=0,
                phi_resolution_rel=0,
            )

            # charged lepton smearing is minimal since semiconductor based detection works well
            proc.set_smearing(
                pdgids=[13, -13],
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
        # combine_and_shuffle(miner_data_file_paths, miner_data_shuffled_path)

        logging.info('running SampleAugmenter...')

        sa = SampleAugmenter(miner_data_shuffled_path)
        #
        # train_result = sa.sample_train_ratio(
        #     theta0=benchmarks([b.name for b in self.physics_benchmarks]),
        #     theta1=benchmark(self.wide_expected_benchmark.name),
        #     n_samples=n_train_events,
        #     sample_only_from_closest_benchmark=True,
        #     partition='train',
        #     folder=path.join(self.data_dir, 'data/samples'),
        #     filename='train',
        # )
        #
        # validation_result = sa.sample_train_ratio(
        #     theta0=benchmarks([b.name for b in self.physics_benchmarks]),
        #     theta1=benchmark(self.wide_expected_benchmark.name),
        #     n_samples=n_val_events,
        #     sample_only_from_closest_benchmark=True,
        #     partition='validation',
        #     folder=path.join(self.data_dir, 'data/samples'),
        #     filename='valid',
        # )
        #
        # _0 = sa.sample_test(
        #     theta=benchmark(self.expected_benchmark.name),
        #     n_samples=n_test_events,
        #     folder=path.join(self.data_dir, 'data/samples'),
        #     filename='test',
        # )

        thetas_benchmarks, xsecs_benchmarks, xsec_errors_benchmarks = sa.cross_sections(
            theta=benchmarks([b.name for b in self.physics_benchmarks])
        )

        # logging.info('effective_n_samples train and validation: {} and {}'.format(train_result[-1], validation_result[-1]))
        logging.info(str(xsecs_benchmarks))

        # forge.train
        forge = ParameterizedRatioEstimator(n_hidden=(100, 100))
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
                             n_epochs=50,
                             batch_size=256,
                             initial_lr=3e-4,
                             final_lr=1e-6,
                             scale_inputs=True
                             )

        forge.save(path.join(self.data_dir, 'models/alice'))

        # Test the model
        # theta_ref = np.array([[c.mass, c.width] for c in self.wide_artificial_benchmarks])
        theta_ref = np.array([[c.mass, c.width] for c in self.physics_benchmarks])
        np.save(path.join(self.data_dir, 'data/samples/theta_ref.npy'), theta_ref)

        # theta parameters with width:
        mass_bins = np.linspace(self.mass_low, self.mass_high, 2 * (self.mass_high - self.mass_low))
        width_bins = np.array([1.5, ])  # pick expected value of top width
        mass, width = np.meshgrid(mass_bins, width_bins)
        # mass_width_grid_0 = np.vstack((mass.flatten(), width.flatten())).T
        # np.save(path.join(self.data_dir, 'data/samples/mass_width_grid_0.npy'), mass_width_grid_0)

        # theta parameters, no width:
        mass_grid = np.vstack((mass.flatten(), )).T
        np.save(path.join(self.data_dir, 'data/samples/mass_grid.npy'), mass_grid)

        log_r_hat, _0 = forge.evaluate(
            # theta=path.join(self.data_dir, 'data/samples/mass_width_grid_0.npy'),
            theta=path.join(self.data_dir, 'data/samples/mass_grid.npy'),
            x=path.join(self.data_dir, 'data/samples/x_test.npy'),
            test_all_combinations=True,
            evaluate_score=False,
            run_on_gpu=False,
        )

        np.save(path.join(self.data_dir, 'data/samples/log_r_hat.npy'), log_r_hat)

        # plot final results
        mean_log_r_hat = np.mean(log_r_hat, axis=1)
        llr = -2 * mean_log_r_hat
        best_fit_i = np.argmin(llr)
        best_fit_x_y = mass_grid[best_fit_i]

        logging.info('best_fit {}'.format(best_fit_x_y))

        logging.info('plotting...')

        # fig = plt.figure(figsize=(5, 4))
        # sc = plt.scatter(thetas_benchmarks[:, 0], thetas_benchmarks[:, 1], c=xsecs_benchmarks, s=200., cmap='viridis', vmin=0., lw=2., edgecolor='black', marker='s')
        # plt.errorbar(thetas_benchmarks[:, 0], thetas_benchmarks[:, 1], yerr=xsec_errors_benchmarks, linestyle="None")
        # cb = plt.colorbar(sc)
        # plt.savefig(path.join(self.data_dir, 'theta_scatter_plot.png'), bbox_inches='tight')

        # TODO: new method
        # plot observables for shuffled elements, sample 1,000,000 events for example
        _ = plot_distributions(
            filename=miner_data_shuffled_path,
            uncertainties='none',
            n_bins=20,
            n_cols=5,
            normalize=True,
            parameter_points=['160_15', '172_15', '185_15', '165_15', '180_15'],
            linestyles='-',
            sample_only_from_closest_benchmark=True,
            n_events=1000000,
        )
        plt.tight_layout()
        plt.savefig(path.join(self.data_dir, 'observables_histograms.png'), bbox_inches='tight')

        fig = plt.figure(figsize=(6, 5))
        plt.plot(mass_grid[:, 0], llr, marker='o', ls=' ', zorder=1)
        plt.scatter(best_fit_x_y[0], llr[best_fit_i], s=100., color='red', marker='*', zorder=2)
        plt.xlabel(r'$Mass (GeV)$')
        plt.ylabel(r'$Likelihood Ratio -2logp(x|\theta)$')
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
