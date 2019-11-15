from os import path
from sys import argv

from collections import namedtuple
import logging

# headless
from matplotlib import use
use("Agg")

from numpy import NaN, isnan
from random import gauss

from madminer.core import MadMiner
from madminer.lhe import LHEReader
from madminer.utils.particle import MadMinerParticle

Benchmark = namedtuple('Benchmark', ['mass', 'width', 'name'])


# http://inspirehep.net/record/1204335/files/ATLAS-CONF-2012-082.pdf (page 3)
# vis - b-jet and charged leptons
# invis - missing energy
def calc_mT(vis, invis):
    mT = (vis.m**2 + 2.*(vis.e * invis.e - vis.pt * invis.pt))**.5
    if isnan(mT):
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
    met_1.setpxpypze(-visible_sum_1.px, -visible_sum_1.py, NaN, visible_sum_1.pt)

    met_2 = MadMinerParticle()
    met_2.setpxpypze(-visible_sum_2.px, -visible_sum_2.py, NaN, visible_sum_2.pt)

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

        nu1.setpxpypze(nu1px, nu1py, NaN, nu1e)
        nu2.setpxpypze(nu2px, nu2py, NaN, nu2e)

        mT_e = calc_mT(visible_sum_1, nu1)
        mT_mu = calc_mT(visible_sum_2, nu2)

        max_mT_list.append(max(mT_mu, mT_e))

    return min(max_mT_list)


class EventRunner:
    def __init__(self):

        mass_low, mass_high = (160, 186)  # high is exclusive

        self.physics_benchmarks = [Benchmark(float(i), 1.5, '{0}_{1}'.format(i, 15)) for i in range(mass_low, mass_high)]
        expected_benchmark = Benchmark(172.0, 1.5, '172_15')
        self.wide_artificial_benchmarks = [Benchmark(float(i), 4.0, '{0}_{1}'.format(i, 40)) for i in
                                           range(mass_low, mass_high, 5)]
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

    def merge_and_train(self):
        pass


def main():
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
