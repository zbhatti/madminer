from __future__ import absolute_import, division

import logging
import random
import re

from collections import namedtuple

import numpy as np
from matplotlib import pyplot as plt
from os import path

from madminer.core import MadMiner
from madminer.lhe import LHEReader
from madminer.sampling import combine_and_shuffle
from madminer.sampling import SampleAugmenter
from madminer.sampling import benchmark, benchmarks
from madminer.ml import ParameterizedRatioEstimator
from madminer.plotting import plot_distributions
from madminer.utils.particle import MadMinerParticle

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
        nu1px = random.gauss(met_sum.px/2., met_sum.px)
        nu1py = random.gauss(met_sum.py/2., met_sum.py)
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


def main():
    mg_dir = str('/home/zbhatti/util/MG5_aMC_v2_6_5')

    tutorial_dir = path.dirname(__file__)

    # filepath settings for the experiment
    setup_logging(path.join(tutorial_dir, 'output.log'))
    logging.info('tutorial directory: {0}'.format(tutorial_dir))

    miner_h5_path = path.join(tutorial_dir,  'data/ttbar_mvm.h5')
    miner_h5_path_with_lhe = miner_h5_path.replace('.h5', '_with_data.h5')
    miner_h5_path_shuffled = miner_h5_path.replace('.h5', '_shuffled.h5')
    run_card = path.join(tutorial_dir, 'cards/ttbar_run_card.dat')
    run_card_more = path.join(tutorial_dir, 'cards/ttbar_more_run_card.dat')

    mass_low, mass_high = (160, 186)  # high is exclusive

    # control which steps are rerun
    rerun_madgraph = False
    rerun_lhereader = True
    rerun_sample_augmenter = True
    rerun_forge_train = True
    rerun_forge_evaluate = True

    n_events = scrape_n_events(run_card)
    n_more_events = scrape_n_events(run_card_more)
    n_train_events = n_more_events * 10
    n_test_events = 10000
    run_smearing = True
    if run_smearing:
        train_filename = 'train_w_smearing'
        test_filename = 'test_w_smearing'
    else:
        train_filename = 'train_wo_smearing'
        test_filename = 'test_wo_smearing'

    logging.info('running madgraph on {0} events and {1} more events'.format(n_events, n_more_events))

    Benchmark = namedtuple('Benchmark', ['mass', 'width', 'name'])
    physics_benchmarks = [Benchmark(float(i), 1.5, '{0}_{1}'.format(i, 15)) for i in range(mass_low, mass_high)]
    expected_benchmark = Benchmark(172.0, 1.5, '172_15')
    artificial_benchmarks = [Benchmark(float(i), 4.0, '{0}_{1}'.format(i, 40)) for i in range(mass_low, mass_high, 5)]
    high_sample_benchmark = Benchmark(172.5, 4.0, '172.5_40')

    low_sample_benchmarks = [cb.name for cb in artificial_benchmarks]
    high_sample_benchmarks = [high_sample_benchmark.name]

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
        for b in physics_benchmarks + artificial_benchmarks + [high_sample_benchmark]:
            miner.add_benchmark({'TOP_MASS': b.mass, 'TOP_WIDTH': b.width}, b.name)

        miner.run_multiple(
            sample_benchmarks=low_sample_benchmarks,
            mg_directory=mg_dir,
            mg_process_directory=path.join(tutorial_dir, 'mg_processes/signal'),
            proc_card_file=path.join(tutorial_dir, 'cards/ttbar_proc_card.dat'),
            param_card_template_file=path.join(tutorial_dir, 'cards/param_card_template.dat'),
            run_card_files=[run_card],
            log_directory=path.join(tutorial_dir, 'logs/signal'),
        )
        miner.run_multiple(
            sample_benchmarks=[high_sample_benchmark.name],
            mg_directory=mg_dir,
            mg_process_directory=path.join(tutorial_dir, 'mg_processes/signal2'),
            proc_card_file=path.join(tutorial_dir, 'cards/ttbar_proc_card.dat'),
            param_card_template_file=path.join(tutorial_dir, 'cards/param_card_template.dat'),
            run_card_files=[run_card_more],
            log_directory=path.join(tutorial_dir, 'logs/signal2'),
        )
        miner.save(miner_h5_path)

    else:
        logging.info('loading miner results...')
        miner.load(miner_h5_path)

    if rerun_lhereader:
        logging.info('running LHEProcessor...')
        # read madgraph events and smear them and then calculate observables

        # name: definition
        obs_particles = {
            'e_0': 'e[0]',
            'mu_0': 'mu[0]',
            'j_0': 'j[0]',
            'j_1': 'j[1]',
        }
        proc = LHEReader(miner_h5_path)

        i = 1
        for sample_bench in low_sample_benchmarks:
            proc.add_sample(
                lhe_filename=path.join(tutorial_dir, 'mg_processes/signal/Events/run_{0:0>2}/unweighted_events.lhe.gz'.format(i)),
                sampled_from_benchmark=sample_bench,
                is_background=False,
                k_factor=1.0,
            )
            i += 1

        i = 1
        for sample_bench in high_sample_benchmarks:
            proc.add_sample(
                lhe_filename=path.join(tutorial_dir, 'mg_processes/signal2/Events/run_{0:0>2}/unweighted_events.lhe.gz'.format(i)),
                sampled_from_benchmark=sample_bench,
                is_background=False,
                k_factor=1.0,
            )
            i += 1

        for name, definition in obs_particles.iteritems():
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
        proc.save(miner_h5_path_with_lhe)

        logging.info(proc.observables)
        logging.info(proc.observations.keys())

    else:
        logging.info('skipping LHEProcessor...')

    # skip_feature = 17
    # feature_train_list = range(0, skip_feature) + range(skip_feature+1, 23) # skip met.phi()

    parameter_points = ['160_15', '172_15', '185_15', '160_40', '170_40', '185_40']
    _ = plot_distributions(
        filename=miner_h5_path_with_lhe,
        uncertainties='none',
        n_bins=20,
        n_cols=5,
        normalize=True,
        parameter_points=parameter_points,
        linestyles='-',
        sample_only_from_closest_benchmark=True,
        )
    plt.tight_layout()
    plt.savefig(path.join(tutorial_dir, 'observables_histograms.png'), bbox_inches='tight')
    if rerun_sample_augmenter and rerun_lhereader:
        del proc

    if rerun_sample_augmenter:
        logging.info('running SampleAugmenter...')
        combine_and_shuffle([miner_h5_path_with_lhe], miner_h5_path_shuffled)
        sa = SampleAugmenter(miner_h5_path_shuffled)
        train_result = sa.sample_train_ratio(
            theta0=benchmarks([b.name for b in physics_benchmarks]),
            theta1=benchmark(high_sample_benchmark.name),
            n_samples=n_train_events,
            sample_only_from_closest_benchmark=True,
            folder=path.join(tutorial_dir, 'data/samples'),
            filename=train_filename,
        )

        _0 = sa.sample_test(
            theta=benchmark(expected_benchmark.name),
            n_samples=n_test_events,
            folder=path.join(tutorial_dir, 'data/samples'),
            filename=test_filename,
        )

        thetas_benchmarks, xsecs_benchmarks, xsec_errors_benchmarks = sa.cross_sections(
            theta=benchmarks([b.name for b in physics_benchmarks])
        )

        logging.info(str(xsecs_benchmarks))
        fig = plt.figure(figsize=(5, 4))
        sc = plt.scatter(thetas_benchmarks[:, 0], thetas_benchmarks[:, 1], c=xsecs_benchmarks,
                         s=200., cmap='viridis', vmin=0., lw=2., edgecolor='black', marker='s')
        plt.errorbar(thetas_benchmarks[:, 0], thetas_benchmarks[:, 1], yerr=xsec_errors_benchmarks, linestyle="None")
        cb = plt.colorbar(sc)

        plt.savefig(path.join(tutorial_dir, 'theta_scatter_plot.png'), bbox_inches='tight')

    else:
        logging.info('skipping SampleAugmenter...')

    if rerun_sample_augmenter:
        del sa

    forge = ParameterizedRatioEstimator(n_hidden=(100, 100))
    if rerun_forge_train:
        logging.info('running forge')
        x_train_path = path.join(tutorial_dir, 'data/samples/x_{}.npy'.format(train_filename))
        y_train_path = path.join(tutorial_dir, 'data/samples/y_{}.npy'.format(train_filename))
        r_xz_train_path = path.join(tutorial_dir, 'data/samples/r_xz_{}.npy'.format(train_filename))
        theta0_train_path = path.join(tutorial_dir, 'data/samples/theta0_{}.npy'.format(train_filename))
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

        forge.save(path.join(tutorial_dir, 'models/alice'))

    else:
        logging.info('skipping forge train')
        forge.load(path.join(tutorial_dir, 'models/alice'))

    theta_ref = np.array([[c.mass, c.width] for c in artificial_benchmarks])
    np.save(path.join(tutorial_dir, 'data/samples/theta_ref.npy'), theta_ref)

    # theta 0
    mass_bins = np.linspace(mass_low, mass_high, 2 * (mass_high - mass_low))
    width_bins = np.array([1.5, ])  # pick expected value of top width
    mass, width = np.meshgrid(mass_bins, width_bins)
    mass_width_grid_0 = np.vstack((mass.flatten(), width.flatten())).T
    np.save(path.join(tutorial_dir, 'data/samples/mass_width_grid_0.npy'), mass_width_grid_0)

    # theta 1
    mass_bins = np.array([172.5, ])
    width_bins = np.array([4.0, ])
    mass, width = np.meshgrid(mass_bins, width_bins)
    mass_width_grid_1 = np.vstack((mass.flatten(), width.flatten())).T
    np.save(path.join(tutorial_dir, 'data/samples/mass_width_grid_1.npy'), mass_width_grid_1)

    if rerun_forge_evaluate:
        log_r_hat, _0 = forge.evaluate(
            theta=path.join(tutorial_dir, 'data/samples/mass_width_grid_0.npy'),
            x=path.join(tutorial_dir, 'data/samples/x_{}.npy'.format(test_filename)),
            test_all_combinations=True,
            evaluate_score=False,
            run_on_gpu=True,
        )

        np.save(path.join(tutorial_dir, 'data/samples/log_r_hat.npy'), log_r_hat)

    else:
        log_r_hat = np.load(path.join(tutorial_dir, 'data/samples/log_r_hat.npy'))

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
    plt.savefig(path.join(tutorial_dir, 'llr.png'), bbox_inches='tight')
    plt.show()
    logging.info('')


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


if __name__ == '__main__':
    main()
