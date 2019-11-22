from os import path
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


# from proc_card g g > t t~,  t > b  e+ ve, t~ > b~ mu- vm~
# particles index:                0   1  2        3  4    5
# particle id:                    5 -11 12       -5 13  -14
def mt2(particles, leptons, photons, jets, met):
    b, e, ve, bbar, mu, vm = particles
    n_picks = 1000
    visible_sum_1 = MadMinerParticle()
    visible_sum_1.setpxpypze(0.0, 0.0, 0.0, 0.0)

    visible_sum_2 = MadMinerParticle()
    visible_sum_2.setpxpypze(0.0, 0.0, 0.0, 0.0)

    visible_sum_1 = b + e
    visible_sum_2 = bbar + mu

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

        mT_e = calc_mT(visible_sum_1, nu1)
        mT_mu = calc_mT(visible_sum_2, nu2)

        max_mT_list.append(max(mT_mu, mT_e))

    return min(max_mT_list)


class EventRunner:
    def __init__(self):

        self.mass_low, self.mass_high = (160, 186)  # high is exclusive

        self.physics_benchmarks = [Benchmark(float(i), 1.5, '{0}_{1}'.format(i, 15)) for i in range(self.mass_low, self.mass_high)]
        self.expected_benchmark = Benchmark(172.0, 1.5, '172_15')
        self.wide_artificial_benchmarks = [Benchmark(float(i), 4.0, '{0}_{1}'.format(i, 40)) for i in
                                           range(self.mass_low, self.mass_high, 5)]
        self.wide_expected_benchmark = Benchmark(172.5, 4.0, '172.5_40')

        self.low_sample_benchmark_names = [cb.name for cb in self.wide_artificial_benchmarks]
        self.high_sample_benchmark_names = [self.wide_expected_benchmark.name]

        self.working_directory = '/scratch/zb609/madminer_data'
        self.config_directory = ''
        self.miner_setup_path = path.join(self.working_directory,  'data/miner_setup.h5')

    #  run solo
    def build_setup(self):
        miner = MadMiner()

        logging.info('running miner...')

        miner.add_parameter(
            lha_block=str('MASS'),
            lha_id=6,
            parameter_name=str('TOP_MASS'),
        )
        miner.add_parameter(
            lha_block=str('DECAY'),
            lha_id=6,
            parameter_name=str('TOP_WIDTH'),
        )

        # add scanning points
        for b in self.physics_benchmarks + self.wide_artificial_benchmarks + [self.wide_expected_benchmark]:
            miner.add_benchmark({'TOP_MASS': b.mass, 'TOP_WIDTH': b.width}, b.name)

        miner.save(self.miner_setup_path)

    # run concurrently
    def generate_events(self, few_or_many, worker_id):
        mg_dir = str('/home/zb609/scratch_dir/MG5_aMC_v2_6_5')
        working_directory = self.working_directory
        config_directory = path.dirname(__file__)

        # shared inputs:
        proc_card_path = path.join(config_directory, 'cards/ttbar_proc_card.dat')
        param_template_path = path.join(config_directory, 'cards/param_card_template.dat')

        if few_or_many == 'few':
            run_card_path = path.join(config_directory, 'cards/ttbar_few_run_card.dat')
            sample_benchmarks = self.low_sample_benchmark_names

        elif few_or_many == 'many':
            run_card_path = path.join(config_directory, 'cards/ttbar_many_run_card.dat')
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
        mg_process_directory = path.join(working_directory, 'mg_processes_{}_{}/signal'.format(few_or_many, worker_id))
        log_directory = path.join(working_directory, 'logs_{}_{}/signal'.format(few_or_many, worker_id))
        event_data_path = path.join(working_directory, 'data/miner_lhe_data_{}_{}.h5'.format(few_or_many, worker_id))

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
        run_smearing = True

        # name: definition
        obs_particles = {
            'e_0': 'e[0]',
            'mu_0': 'mu[0]',
            'j_0': 'j[0]',
            'j_1': 'j[1]',
        }

        proc = LHEReader(self.miner_setup_path)
        i = 1
        for sample_bench in sample_benchmarks:
            lhe_filename = path.join(working_directory, 'mg_processes_{0}_{1}/signal/Events/run_{2:0>2}/unweighted_events.lhe.gz'.format(few_or_many, worker_id, i))

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

        charged_leptons = ['e_0', 'mu_0']
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
                pdgids=[11, 13, -11, -13],
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

        miner_data_file_patten = '/scratch/zb609/madminer_data/data/miner_lhe_data_*_*.h5'
        miner_data_shuffled_path = '/scratch/zb609/madminer_data/data/miner_lhe_data_shuffled.h5'
        n_train_events = 20000000
        n_test_events = 100000

        miner_data_file_paths = glob(miner_data_file_patten)

        # TODO: new method
        # run sample augmenter - event_data_merged

        # logging.info('shuffling LHE files {}'.format(miner_data_file_paths))
        # combine_and_shuffle(miner_data_file_paths, miner_data_shuffled_path)

        # logging.info('running SampleAugmenter...')

        sa = SampleAugmenter(miner_data_shuffled_path)

        train_result = sa.sample_train_ratio(
            theta0=benchmarks([b.name for b in self.physics_benchmarks]),
            theta1=benchmark(self.wide_expected_benchmark.name),
            n_samples=n_train_events,
            sample_only_from_closest_benchmark=True,
            folder=path.join(self.working_directory, 'data/samples'),
            filename='train',
        )

        _0 = sa.sample_test(
            theta=benchmark(self.expected_benchmark.name),
            n_samples=n_test_events,
            folder=path.join(self.working_directory, 'data/samples'),
            filename='test',
        )

        thetas_benchmarks, xsecs_benchmarks, xsec_errors_benchmarks = sa.cross_sections(
            theta=benchmarks([b.name for b in self.physics_benchmarks])
        )

        logging.info(str(xsecs_benchmarks))
        fig = plt.figure(figsize=(5, 4))
        sc = plt.scatter(thetas_benchmarks[:, 0], thetas_benchmarks[:, 1], c=xsecs_benchmarks,
                         s=200., cmap='viridis', vmin=0., lw=2., edgecolor='black', marker='s')
        plt.errorbar(thetas_benchmarks[:, 0], thetas_benchmarks[:, 1], yerr=xsec_errors_benchmarks, linestyle="None")
        cb = plt.colorbar(sc)
        #
        plt.savefig(path.join(self.working_directory, 'theta_scatter_plot.png'), bbox_inches='tight')

        # TODO: new method
        # plot observables for shuffled elements, sample 1,000,000 events for example
        # _ = plot_distributions(
        #     filename=miner_data_shuffled_path,
        #     uncertainties='none',
        #     n_bins=20,
        #     n_cols=5,
        #     normalize=True,
        #     parameter_points=['160_15', '172_15', '185_15', '160_40', '170_40', '185_40'],
        #     linestyles='-',
        #     sample_only_from_closest_benchmark=True,
        #     n_events=1000000,
        # )
        # plt.tight_layout()
        # plt.savefig(path.join(self.working_directory, 'observables_histograms.png'), bbox_inches='tight')

        # forge.train
        forge = ParameterizedRatioEstimator(n_hidden=(100, 100))
        logging.info('running forge')
        x_train_path = path.join(self.working_directory, 'data/samples/x_{}.npy'.format('train'))
        y_train_path = path.join(self.working_directory, 'data/samples/y_{}.npy'.format('train'))
        r_xz_train_path = path.join(self.working_directory, 'data/samples/r_xz_{}.npy'.format('train'))
        theta0_train_path = path.join(self.working_directory, 'data/samples/theta0_{}.npy'.format('train'))
        result = forge.train(method='alice',
                             x=x_train_path,
                             y=y_train_path,
                             theta=theta0_train_path,
                             r_xz=r_xz_train_path,
                             n_epochs=25,
                             validation_split=0.3,
                             batch_size=256,
                             initial_lr=0.001,
                             scale_inputs=True
                             )

        forge.save(path.join(self.working_directory, 'models/alice'))

        # Test the model
        theta_ref = np.array([[c.mass, c.width] for c in self.wide_artificial_benchmarks])
        np.save(path.join(self.working_directory, 'data/samples/theta_ref.npy'), theta_ref)

        # theta 0
        mass_bins = np.linspace(self.mass_low, self.mass_high, 2 * (self.mass_high - self.mass_low))
        width_bins = np.array([1.5, ])  # pick expected value of top width
        mass, width = np.meshgrid(mass_bins, width_bins)
        mass_width_grid_0 = np.vstack((mass.flatten(), width.flatten())).T
        np.save(path.join(self.working_directory, 'data/samples/mass_width_grid_0.npy'), mass_width_grid_0)

        # theta 1
        mass_bins = np.array([172.5, ])
        width_bins = np.array([4.0, ])
        mass, width = np.meshgrid(mass_bins, width_bins)
        mass_width_grid_1 = np.vstack((mass.flatten(), width.flatten())).T
        np.save(path.join(self.working_directory, 'data/samples/mass_width_grid_1.npy'), mass_width_grid_1)

        log_r_hat, _0 = forge.evaluate(
            theta=path.join(self.working_directory, 'data/samples/mass_width_grid_0.npy'),
            x=path.join(self.working_directory, 'data/samples/x_{}.npy'.format('test')),
            test_all_combinations=True,
            evaluate_score=False,
            run_on_gpu=False,
        )

        np.save(path.join(self.working_directory, 'data/samples/log_r_hat.npy'), log_r_hat)

        # plot final results
        mean_log_r_hat = np.mean(log_r_hat, axis=1)
        llr = -2 * mean_log_r_hat
        best_fit_i = np.argmin(llr)
        best_fit_x_y = mass_width_grid_0[best_fit_i]

        logging.info('best_fit {}'.format(best_fit_x_y))
        fig = plt.figure(figsize=(6, 5))

        plt.plot(mass_width_grid_0[:, 0], llr, marker='o', ls=' ', zorder=1)
        plt.scatter(best_fit_x_y[0], llr[best_fit_i], s=100., color='red', marker='*', zorder=2)
        plt.xlabel(r'$Mass (GeV)$')
        plt.ylabel(r'$Likelihood Ratio -2logp(x|\theta)$')
        plt.savefig(path.join(self.working_directory, 'llr.png'), bbox_inches='tight')
        plt.show()
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
    logging.info('args: setup|generate|train few|many worker_id')
    if argv[1] == 'setup':
        EventRunner().build_setup()

    elif argv[1] == 'generate':
        few_or_many = argv[2]  # few or many
        worker_id = argv[3]  # 0, 1, 2, ... 99, etc
        EventRunner().generate_events(few_or_many, worker_id)

    elif argv[1] == 'train':
        EventRunner().merge_and_train()


if __name__ == '__main__':
    main()
