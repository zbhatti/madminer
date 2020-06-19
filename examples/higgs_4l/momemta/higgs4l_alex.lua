
local electron = declare_input("electron")
local muon = declare_input("muon")
local antimuon = declare_input("antimuon")
local positron = declare_input("positron")

parameters = {
    energy = 13000.0,
    higgs_width = 4.07e-03, -- overriden by higgs4l.cc
}

inputs = {
    electron.reco_p4,
    muon.reco_p4,
    antimuon.reco_p4,
    positron.reco_p4
}

BuildInitialState.initial_state = {
    particles = inputs,
}

MatrixElement.h_ZZ_4l = {
      pdf = 'CT10nlo',
      pdf_scale = 172.5,

      matrix_element = 'higgs4l_sm__hgg_plugin_P1_Sigma_sm__hgg_plugin_gg_epemmupmum',
      matrix_element_parameters = {
          card = '/home/zbhatti/codebase/madminer/examples/higgs_4l/cards/param_card_template.dat',
      },

    initialState = "initial_state::partons",

    override_parameters = {
        mdl_WH = parameter('higgs_width'),
    },

    particles = {
        inputs = inputs,
        ids = {
            {
                pdg_id = 11,
                me_index = 2,
            },
            {
                pdg_id = 13,
                me_index = 4,
            },
            {
                pdg_id = -13,
                me_index = 3,
            },
            {
                pdg_id = -11,
                me_index = 1,
            },
        }
    },

}

integrand("h_ZZ_4l::output")

