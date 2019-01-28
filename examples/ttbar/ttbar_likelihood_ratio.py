from __future__ import absolute_import, division # , unicode_literals

import logging
import random
import re
from copy import deepcopy
from collections import namedtuple
from multiprocessing import Pool

import numpy as np
import matplotlib
from matplotlib import pyplot as plt
# %matplotlib inline
from os import getcwd, path

from madminer.core import MadMiner
from madminer.lhe import LHEProcessor
from madminer.sampling import combine_and_shuffle
from madminer.sampling import SampleAugmenter
from madminer.sampling import constant_benchmark_theta, multiple_benchmark_thetas
from madminer.ml import MLForge
from madminer.plotting import plot_distributions
from madminer.utils.particle import MadMinerParticle


def setup_logging(filepath=None):
    # MadMiner output
    logging.basicConfig(
        format='%(asctime)-5.5s %(name)-20.20s %(levelname)-7.7s %(message)s',
        datefmt='%H:%M',
        level=logging.DEBUG
    )

    fileh = None
    if filepath:
        fileh = logging.FileHandler(filepath, 'w')

    # Output of all other modules (e.g. matplotlib)
    for key in logging.Logger.manager.loggerDict:
        if "madminer" not in key:
            logging.getLogger(key).setLevel(logging.DEBUG)
            if fileh:
                logger = logging.getLogger(key)
                logger.addHandler(fileh)


def scrape_n_events(run_card_path):
    for line in  open(run_card_path, 'r'):
        match = re.match(r'([0-9]+).*n_?events.*', line)
        if match:
            return int(match.groups()[0])
    else:
        raise RuntimeError('could not extract n_events/nevents from {0}'.format(run_card_path))


# W+- mass expected 80.385 GeV +- 0.015
# transverse is x-y plane perpendicular to beam
# accepts the charged lepton and neutrino that were decay products of a W boson
def w_boson_transverse_mass(l, nu):
    W_T_m = (l.m**2 + 2.*(l.e * nu.e - l.px * nu.px - l.py * nu.py))**.5
    if np.isnan(W_T_m):
        raise ValueError('w_t_m not determined\nl: {}\n nu: {}')
    return W_T_m


# from proc_card g g > t t~,  t > b e+ ve, t~ > b~ mu- vm~
# particles index:                0 1  2        3  4   5
def mt2(particles):
    b, e, ve, bbar, mu, vm = particles
    n_picks = 1000
    visible_sum = MadMinerParticle()
    visible_sum.setpxpypze(0.0, 0.0, 0.0, 0.0)

    for particle in particles:
        pdgid = abs(particle.pdgid)
        if pdgid in [1, 2, 3, 4, 5, 6, 9, 11, 13, 15, 21, 22, 23, 24, 25]:
            visible_sum += particle

    # total missing energy summing to electron and muon neutrino
    met = MadMinerParticle()
    met.setpxpypze(-visible_sum.px, -visible_sum.px, np.NaN, visible_sum.pt)

    max_W_T_m_list = []
    nu1 = MadMinerParticle()
    nu2 = MadMinerParticle()
    for i in range(n_picks):
        nu1px = random.gauss(met.px/2., met.px)
        nu1py = random.gauss(met.py/2., met.py)
        nu1e = (nu1px**2 + nu1py**2)**.5

        nu2px = met.px - nu1.px
        nu2py = met.py - nu1.py
        nu2e = (nu2px**2 + nu2py**2)**.5

        nu1.setpxpypze(nu1px, nu1py, np.NaN, nu1e)
        nu2.setpxpypze(nu2px, nu2py, np.NaN, nu2e)

        W_T_m_e = w_boson_transverse_mass(e, nu1)
        W_T_m_mu = w_boson_transverse_mass(mu, nu2)
        max_W_T_m = max(W_T_m_e, W_T_m_mu)
        max_W_T_m_list.append(max_W_T_m)

    return min(max_W_T_m_list)


def main():
    mg_dir = str('/home/zbhatti/util/MG5_aMC_v2_6_5')

    tutorial_dir = path.dirname(__file__) # getcwd()

    # filepath settings for the experiment
    setup_logging(path.join(tutorial_dir, 'output.log'))
    logging.info('tutorial directory: {0}'.format(tutorial_dir))

    miner_h5_path = path.join(tutorial_dir,  'data/madminer_example_mvm.h5')
    miner_h5_path_with_data = miner_h5_path.replace('.h5', '_with_data.h5')
    miner_h5_path_shuffled = miner_h5_path.replace('.h5', '_shuffled.h5')
    run_card = path.join(tutorial_dir, 'cards/ttbar_run_card.dat')

    mass_low, mass_high = (160, 185)

    # control which steps are rerun
    rerun_madgraph = False
    rerun_lheprocessor = False
    rerun_sample_augmenter = False
    rerun_forge = False

    n_events = scrape_n_events(run_card)
    logging.info('running madgraph on {0} events'.format(n_events))

    Benchmark = namedtuple('Benchmark', ['mass', 'width', 'name'])
    scan_benchmarks = [Benchmark(float(i), 1.5, '{0}_{1}'.format(i, 15)) for i in range(mass_low, mass_high)]
    constant_benchmark = Benchmark(172.0, 7.0, '172_70')
    expected_benchmark = Benchmark(172.0, 1.5, '172_15')

    miner = MadMiner()
    if rerun_madgraph:
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
        for b in scan_benchmarks + [constant_benchmark]:
            miner.add_benchmark({'TOP_MASS': b.mass, 'TOP_WIDTH': b.width}, b.name)

        miner.save(miner_h5_path)
        miner.run(
            sample_benchmark=constant_benchmark.name,
            mg_directory=mg_dir,
            mg_process_directory=path.join(tutorial_dir, 'mg_processes/signal'),
            proc_card_file=path.join(tutorial_dir, 'cards/ttbar_proc_card.dat'),
            param_card_template_file=path.join(tutorial_dir, 'cards/param_card_template.dat'),
            run_card_file=run_card,
            log_directory=path.join(tutorial_dir, 'logs/signal'),
            # only_prepare_script=True,
        )
    else:
        logging.info('loading miner results...')
        miner.load(miner_h5_path)

    if rerun_lheprocessor:
        logging.info('running LHEProcessor...')
        # read madgraph events and smear them and then calculate observables

        # name: definition
        obs_particles = {
            'e_0': 'e[0]',
            'mu_0': 'mu[0]',
            'j_0': 'j[0]',
            'j_1': 'j[1]',
        }

        proc = LHEProcessor(miner_h5_path)
        proc.add_sample(
            lhe_filename=path.join(tutorial_dir, 'mg_processes/signal/Events/run_01/unweighted_events.lhe.gz'),
            sampled_from_benchmark=constant_benchmark.name,
            is_background=False,
            k_factor=1.0,
        )

        # proc.set_smearing(
        #     pdgids=[1, 2, 3, 4, 5, 6, 9, 22, -1, -2, -3, -4, -5, -6],  # Partons giving rise to jets
        #     energy_resolution_abs=0.,
        #     energy_resolution_rel=0.1,
        #     pt_resolution_abs=None,
        #     pt_resolution_rel=None,
        #     eta_resolution_abs=0,
        #     eta_resolution_rel=0,
        #     phi_resolution_abs=0,
        #     phi_resolution_rel=0,
        # )
        #
        # proc.set_smearing(
        #     pdgids=[11, 13, -11, -13],
        #     # electron and muon smearing is minimal since semiconductor based detection is so excellent
        #     energy_resolution_abs=0.,
        #     energy_resolution_rel=0.05,
        #     pt_resolution_abs=None,
        #     pt_resolution_rel=None,
        #     eta_resolution_abs=0,
        #     eta_resolution_rel=0,
        #     phi_resolution_abs=0,
        #     phi_resolution_rel=0,
        # )

        for name, definition in obs_particles.iteritems():
            proc.add_observable(
                str('{0}_energy'.format(name)),
                str('{0}.e'.format(definition)),
                required=True
            )

            proc.add_observable(
                str('{0}_momentum'.format(name)),
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

        standard_obs = deepcopy(proc.observables)
        # missing energy object's momentum in transverse direction
        proc.add_observable(
            str('met_pt'),
            str('met.pt'),
            required=True
        )
        proc.add_cut('met.pt >= 25.0')

        # combined rest mass of NOT t, tbar
        # proc.add_observable(
        #     str('mttbar'),
        #     str('(v[0] + v[1] + e[0] + mu[0] + j[0] + j[1]).m'),
        #     required=True
        # )

        # rest_mass_t as sum of decay products
        # proc.add_observable(
        #     str('t_0_m'),
        #     str('(p[0] + p[1] + p[2]).m'),
        #     required=True
        # )

        # rest_mass_tbar as sum of decay products
        # proc.add_observable(
        #     str('t_1_m'),
        #     str('(p[3] + p[4] + p[5]).m'),
        #     required=True
        # )

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

        proc.add_observable_from_function(
            'mt2',
            mt2,
            required=True,
        )

        proc.analyse_samples(parse_events_as_xml=True)
        proc.save(miner_h5_path_with_data)

        logging.info(proc.observables)
        logging.info(proc.observations.keys())

    else:
        logging.info('skipping LHEProcessor...')

    plot_observables = ['met_pt', 'm_e_0_j_0', 'm_e_0_j_1', 'm_mu_0_j_0', 'm_mu_0_j_1', 'mt2']
    parameter_points = ['160_15', '172_15', '184_15', '172_70']
    _ = plot_distributions(
        filename=miner_h5_path_with_data,
        uncertainties='none',
        n_bins=20,
        n_cols=5,
        normalize=True,
        # observables=['mt1', 'mt2']
        # observables=plot_observables,
        parameter_points=parameter_points,
        linestyles='-',
        )
    plt.tight_layout()
    plt.show()
    # plt.savefig(path.join(tutorial_dir, 'observables_histograms.png'), bbox_inches='tight')

    if rerun_sample_augmenter:
        logging.info('running SampleAugmenter...')
        combine_and_shuffle([miner_h5_path_with_data], miner_h5_path_shuffled)
        sa = SampleAugmenter(miner_h5_path_shuffled)
        x, theta0, theta1, y, r_xz, t_xz = sa.extract_samples_train_ratio(
            theta0=multiple_benchmark_thetas([b.name for b in scan_benchmarks]),
            theta1=constant_benchmark_theta(constant_benchmark.name),
            n_samples=n_events, #*10,
            folder=path.join(tutorial_dir, 'data/samples'),
            filename='train'
        )

        _0 = sa.extract_samples_test(
            theta=constant_benchmark_theta(expected_benchmark.name),
            n_samples=n_events, #*10,
            folder=path.join(tutorial_dir, 'data/samples'),
            filename='test'
        )

        thetas_benchmarks, xsecs_benchmarks, xsec_errors_benchmarks = sa.extract_cross_sections(
            theta=multiple_benchmark_thetas([b.name for b in scan_benchmarks])
        )

        logging.info(str(xsecs_benchmarks))
        fig = plt.figure(figsize=(5, 4))
        sc = plt.scatter(thetas_benchmarks[:, 0], thetas_benchmarks[:, 1], c=xsecs_benchmarks,
                         s=200., cmap='viridis', vmin=0., lw=2., edgecolor='black', marker='s')

        cb = plt.colorbar(sc)
        plt.savefig(path.join(tutorial_dir, 'theta_scatter_plot.png'), bbox_inches='tight')
        # plt.show()
    else:
        logging.info('skipping SampleAugmenter...')

    mass_bins = np.linspace(mass_low, mass_high, 2*(mass_high - mass_low))
    width_bins = np.array([1.5, ])  # pick expected value of top width
    mass, width = np.meshgrid(mass_bins, width_bins)
    mass_width_grid = np.vstack((mass.flatten(), width.flatten())).T
    theta_ref = np.array([[constant_benchmark.mass, constant_benchmark.width]])

    if rerun_sample_augmenter:
        del sa

    forge = MLForge()
    if rerun_forge:
        logging.info('running forge')
        training_results = forge.train(method='alice',
                                       theta0_filename=path.join(tutorial_dir, 'data/samples/theta0_train.npy'),
                                       x_filename=path.join(tutorial_dir, 'data/samples/x_train.npy'),
                                       y_filename=path.join(tutorial_dir, 'data/samples/y_train.npy'),
                                       r_xz_filename=path.join(tutorial_dir, 'data/samples/r_xz_train.npy'),
                                       # activation='relu',
                                       # trainer='sgd',
                                       n_hidden=(100, 100, 100),
                                       n_epochs=50,
                                       validation_split=0.3,
                                       batch_size=256,
                                       )

        np.save(path.join(tutorial_dir, 'data/samples/mass_width_grid.npy'), mass_width_grid)
        np.save(path.join(tutorial_dir, 'data/samples/theta_ref.npy'), theta_ref)

        forge.save(path.join(tutorial_dir, 'models/alice'))
        log_r_hat, _, _ = forge.evaluate(
            theta0_filename=path.join(tutorial_dir, 'data/samples/mass_width_grid.npy'),
            x=path.join(tutorial_dir, 'data/samples/x_test.npy'),
            evaluate_score=False
        )

        np.save(path.join(tutorial_dir, 'data/samples/log_r_hat.npy'), log_r_hat)

    else:
        forge.load(path.join(tutorial_dir, 'models/alice'))
        log_r_hat = np.load(path.join(tutorial_dir, 'data/samples/log_r_hat.npy'))
        logging.info('skipping forge')

    expected_llr = np.mean(log_r_hat, axis=1)
    best_fit_i = np.argmin(-2. * expected_llr)
    best_fit_x_y = mass_width_grid[best_fit_i]
    y = -2 * expected_llr
    fig = plt.figure(figsize=(6, 5))

    # plt.scatter(best_fit_x_y[0], best_fit_x_y[1], s=80., color='black', marker='*')
    logging.info('best_fit {}'.format(best_fit_x_y))
    llr_mass = plt.plot(mass_bins, y, marker='o', ls=' ')
    plt.savefig(path.join(tutorial_dir, 'llr.png'), bbox_inches='tight')
    plt.show()
    logging.info('')


if __name__ == '__main__':
    main()
