-- Note: USE_PERM and USE_TF are defined in the C++ code and injected in lua before parsing this file

-- Register inputs
local electron = declare_input("electron")
local muon = declare_input("muon")
local antimuon = declare_input("antimuon")
local positron = declare_input("positron")

-- if USE_PERM then
    -- Automatically insert a Permutator module
--     add_reco_permutations(bjet0, bjet1)
--     add_reco_permutations(muon1, muon0)
-- end

parameters = {
    energy = 13000.,
--     top_mass = 172., -- overriden by ttbar.cc
--     top_width = 1.5, -- changed to match madminer approach
--     W_mass = 79.824360, -- changed to match madminer approach 80.419002,
--     W_width =  2.085000, -- changed to match madminer approach 2.047600
    higgs_width = 4.07e-03, -- overriden by higgs4l.cc
}

cuba = {
    relative_accuracy = 0.001, -- original was 0.01
    verbosity = 3,
}

-- BreitWignerGenerator.flatter_s13 = {
--     -- add_dimension() generates an input tag of type `cuba::ps_points/i`
--     -- where `i` is automatically incremented each time the function is called.
--     -- This function allows MoMEMta to track how many dimensions are needed for the integration.
--     ps_point = add_dimension(),
--     mass = parameter('W_mass'),
--     width = parameter('W_width')
-- }
--
-- BreitWignerGenerator.flatter_s134 = {
--     ps_point = add_dimension(),
--     mass = parameter('top_mass'),
--     width = parameter('top_width')
-- }
--
-- BreitWignerGenerator.flatter_s25 = {
--     ps_point = add_dimension(),
--     mass = parameter('W_mass'),
--     width = parameter('W_width')
-- }
--
-- BreitWignerGenerator.flatter_s256 = {
--     ps_point = add_dimension(),
--     mass = parameter('top_mass'),
--     width = parameter('top_width')
-- }

if USE_TF then
    GaussianTransferFunctionOnEnergy.tf_p1 = {
        ps_point = add_dimension(),
--         reco_particle = muon1.reco_p4,
--         sigma = 0.05, -- comment this out to avoid interpreting madminer inputs as smeared
        sigma = 0.000001,
    }
    electron.set_gen_p4("tf_p1::output")

    GaussianTransferFunctionOnEnergy.tf_p2 = {
        ps_point = add_dimension(),
--         reco_particle = bjet0.reco_p4,
--         sigma = 0.10, -- comment this out to avoid interpreting madminer inputs as smeared
        sigma = 0.000001, -- comment this out to avoid interpreting madminer inputs as smeared
    }
    muon.set_gen_p4("tf_p2::output")


    GaussianTransferFunctionOnEnergy.tf_p3 = {
        ps_point = add_dimension(),
--         reco_particle = muon0.reco_p4,
--         sigma = 0.05, -- comment this out to avoid interpreting madminer inputs as smeared
        sigma = 0.000001, -- comment this out to avoid interpreting madminer inputs as smeared
    }
    antimuon.set_gen_p4("tf_p3::output")

    GaussianTransferFunctionOnEnergy.tf_p4 = {
        ps_point = add_dimension(),
--         reco_particle = bjet1.reco_p4,
--         sigma = 0.10, -- comment this out to avoid interpreting madminer inputs as smeared
        sigma = 0.000001, -- comment this out to avoid interpreting madminer inputs as smeared
    }
    positron.set_gen_p4("tf_p4::output")

end

-- If set_gen_p4 is not called, gen_p4 == reco_p4
inputs = {
    electron.reco_p4,
    muon.reco_p4,
    antimuon.reco_p4,
    positron.reco_p4
}

StandardPhaseSpace.phaseSpaceOut = {
    particles = inputs -- only on visible particles
}

-- Declare module before the permutator to test read-access in the pool
-- for non-existant values.
-- BlockD.blockd = {
--     p3 = inputs[1],
--     p4 = inputs[2],
--     p5 = inputs[3],
--     p6 = inputs[4],
--
--     pT_is_met = true,
--
--     s13 = 'flatter_s13::s',
--     s134 = 'flatter_s134::s',
--     s25 = 'flatter_s25::s',
--     s256 = 'flatter_s256::s',
-- }

-- Loop over block solutions

-- Looper.looper = {
--     solutions = "blockd::solutions",
--     path = Path("boost", "higgs4l", "dmem", "integrand")
-- }

    -- Block D produce solutions with two particles
--     full_inputs = copy_and_append(inputs, {'looper::particles/1', 'looper::particles/2'})

--     BuildInitialState.boost = {
--         do_transverse_boost = true,
--         particles = full_inputs
--     }

--     jacobians = {'flatter_s13::jacobian', 'flatter_s134::jacobian', 'flatter_s25::jacobian', 'flatter_s256::jacobian'}

--     if USE_TF then
--         append(jacobians, {'tf_p1::TF_times_jacobian', 'tf_p2::TF_times_jacobian', 'tf_p3::TF_times_jacobian', 'tf_p4::TF_times_jacobian'})
--     end

--     append(jacobians, {'phaseSpaceOut::phase_space', 'looper::jacobian'})

    BuildInitialState.initial_state = {
    particles = inputs,
}

    MatrixElement.higgs4l = {
      pdf = 'CT10nlo',
      pdf_scale = 172.5, -- parameter('top_mass'),

      matrix_element = 'higgs4l_sm__hgg_plugin_P1_Sigma_sm__hgg_plugin_gg_epemmupmum',
      matrix_element_parameters = {
          -- card = '/home/zbhatti/codebase/madminer/momemta/param_card.dat',
          card = '/home/zbhatti/codebase/madminer/examples/higgs_4l/cards/param_card_template.dat',
      },

      -- variables defined in MatrixElements .cc file
      override_parameters = {
          mdl_WH = parameter('higgs_width'),
      },

--       initialState = 'boost::partons',
      initialState = "initial_state::partons",
      -- mapFinalStates[{-13, 14, 5, 13, -14, -5}] see MatrixElements/ttbarMuMu/SubProcesses/P1_Sigma_sm_gg_mupvmbmumvmxbx/P1_Sigma_sm_gg_mupvmbmumvmxbx.cc
      -- mapFinalStates[{-11, 11, -13, 13}] see MatrixElements/higgs4l/SubProcesses/P1_Sigma_sm__hgg_plugin_gg_epemmupmum/P1_Sigma_sm__hgg_plugin_gg_epemmupmum.cc

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

--       jacobians = jacobians
    }

    DMEM.dmem = {
      x_start = 0.,
      x_end = 2000.,
      n_bins = 500,

      ps_weight = 'cuba::ps_weight',
      particles = inputs,
      me_output = 'higgs4l::output',
    }

    DoubleLooperSummer.integrand = {
        input = "higgs4l::output"
    }

-- End of loop
-- integrand("integrand::sum")
integrand("higgs4l::output")
