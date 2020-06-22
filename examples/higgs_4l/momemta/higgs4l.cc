#include <momemta/ConfigurationReader.h>
#include <momemta/Logging.h>
#include <momemta/MoMEMta.h>
#include <momemta/Unused.h>
#include "H5Cpp.h"
#include <iostream>
#include <fstream>
#include <TH1D.h>
#include <chrono>
#include <cmath>

using namespace std::chrono;
using namespace momemta;
using namespace H5;

using LorentzVectorM = ROOT::Math::LorentzVector<ROOT::Math::PtEtaPhiM4D<float>>;
using LorentzVectorE = ROOT::Math::LorentzVector<ROOT::Math::PtEtaPhiE4D<float>>;

// miner_lhe_data_shuffled.h5:
//h['observables']['names'][:]
//array(['e-_E', 'e-_pt', 'e-_eta', 'e-_phi', 'mu-_E', 'mu-_pt', 'mu-_eta',
//       'mu-_phi', 'mu+_E', 'mu+_pt', 'mu+_eta', 'mu+_phi', 'e+_E',
//       'e+_pt', 'e+_eta', 'e+_phi', 'mass_e+_e-', 'mass_mu+_mu-',
//       'mass_4l'], dtype='|S256')

//h['benchmarks']['names'][:]
//array(['126_1.0E-05', '126_1.8E-05', '126_3.4E-05', '126_6.2E-05',
//       '126_1.1E-04', '126_2.1E-04', '126_3.9E-04', '126_7.2E-04',
//       '126_1.3E-03', '126_2.4E-03', '126_4.5E-03', '126_8.2E-03',
//       '126_1.5E-02', '126_2.8E-02', '126_5.1E-02', '126_9.5E-02',
//       '126_1.7E-01', '126_3.2E-01', '126_5.9E-01', '126_1.1E+00',
//       '126_2.0E+00', '126_1.0E+00', '126_4.0E-03'], dtype='|S256')

void normalizeInput(LorentzVector& p4) {
    if (p4.M() > 0)
        return;

    // Increase the energy until M is positive
    p4.SetE(p4.P());
    while (p4.M2() < 0) {
        double delta = p4.E() * 1e-5;
        p4.SetE(p4.E() + delta);
    };
}

// https://waterprogramming.wordpress.com/2017/08/20/reading-csv-files-in-c
std::vector<std::vector<float>> parse2DCsvFile(std::string inputFileName) {

    std::vector<std::vector<float> > data;
    std::ifstream inputFile(inputFileName);
    int l = 0;

    while (inputFile) {
        l++;
        std::string s;
        if (!getline(inputFile, s)) break;
        if (s[0] != '#') {
            std::istringstream ss(s);
            std::vector<float> record;

            while (ss) {
                std::string line;
                if (!getline(ss, line, ','))
                    break;
                try {
                    record.push_back(stof(line));
                }
                catch (const std::invalid_argument e) {
                    std::cout << "NaN found in file " << inputFileName << " line " << l
                         << std::endl;
                    e.what();
                }
            }

            data.push_back(record);
        }
    }

    return data;
}

int main(int argc, char* argv[]) {

    logging::set_level(logging::level::debug);

    std::string observations_h5_file; // "/home/zbhatti/codebase/madminer/examples/ttbar2/data/madminer_example_mvm_shuffled.h5"
    std::string x_test_csv_file; // "/home/zbhatti/codebase/madminer/momemta/inputs/x_test.csv"
    std::string weight_output_file; // "/home/zbhatti/codebase/madminer/momemta/weights10.csv"


    ParameterSet lua_parameters;
    lua_parameters.set("USE_TF", false);
    lua_parameters.set("USE_PERM", true);

    ConfigurationReader configuration("higgs4l_alex.lua", lua_parameters);

    // load data structures from h5py
    // look the h5 file over and confirm hardcoded values in this file reflect the latest madminer setup
    // h = h5py.File('madminer_example_mvm.h5')
    // h['benchmarks']['names'][:]

    if (argc < 4) { // We expect 3 arguments: the program name, the source path and the destination path
        std::cerr << "Usage: " << argv[0] << "observations_h5_file x_test_csv_file weight_output_file" << std::endl;
        return 1;
    }

    observations_h5_file = argv[1];
    x_test_csv_file = argv[2];
    weight_output_file = argv[3];

    H5::H5File m_h5File = H5File(observations_h5_file, H5F_ACC_RDONLY);
    DataSet benchValsSet = m_h5File.openDataSet("/benchmarks/values");
//    const int n_artificial_benchmarks = 7;
//    const int expected_value_benchmark_position = 12;
    const int theta1_index = 21;
    int theta0_indices[11] = {0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20};
    int n_theta0_benchmarks = sizeof(theta0_indices)/sizeof(theta0_indices[0]);

    DataSpace benchValsSpace = benchValsSet.getSpace();

    hsize_t benchmarkDims[2];
    benchValsSpace.getSimpleExtentDims(benchmarkDims, NULL);
    hsize_t o_benchmarks = benchmarkDims[0];

    LOG(info) << "o_benchmarks: " << o_benchmarks;

    const int rows = 300;

    // generate x_test.csv with python:
    // python -c 'import numpy as np; x_test = np.load("/home/zbhatti/codebase/madminer/examples/ttbar2/data/samples/x_test.npy");
    // np.savetxt("/home/zbhatti/codebase/madminer/momemta/inputs/x_test.csv", x_test, delimiter=",")'

    std::vector<std::vector<float>> x_test = parse2DCsvFile(x_test_csv_file);

    float benchmarksValues[o_benchmarks][1];
    benchValsSet.read(&benchmarksValues[0], PredType::NATIVE_FLOAT, benchValsSpace);
    benchValsSet.close();

    LOG(info) << "****Sample of benchmark values****";
    LOG(info) << benchmarksValues[0][0];
    LOG(info) << benchmarksValues[5][0];
    LOG(info) << benchmarksValues[theta1_index][0];
    LOG(info) << benchmarksValues[o_benchmarks -1][0];
    LOG(info) << "****End****";

    float theta1_higgs_width = pow(10.0, benchmarksValues[theta1_index][0]);

    std::ofstream outputFile;
    outputFile.open(weight_output_file);
    for (int k=0; k < n_theta0_benchmarks; k++){
        outputFile << pow(10.0, benchmarksValues[theta0_indices[k]][0]) << ",";
    }
    outputFile << std::endl;

    // loop over events in the numpy file from madminer and add particle lorentz vectors:
    for(int i=0; i < rows; i++){
        std::vector<float> eventWeights;
        LOG(info) << "calculating event " << i << "/" << rows;

        // LorentzVectorE(pt, eta, phi, E)
        LorentzVectorE electronL    {x_test[i][1], x_test[i][2], x_test[i][3], x_test[i][0]};
        LorentzVectorE muonL        {x_test[i][5], x_test[i][6], x_test[i][7], x_test[i][4]};
        LorentzVectorE antimuonL    {x_test[i][9], x_test[i][10], x_test[i][11], x_test[i][8]};
        LorentzVectorE positronL    {x_test[i][13], x_test[i][14], x_test[i][15], x_test[i][12]};

        // LorentzVector(px, py, pz, E) required for MoMEMta
        Particle electron   {"electron",    LorentzVector {electronL.Px(), electronL.Py(), electronL.Pz(), electronL.E()}, 11 };
        Particle muon       {"muon",        LorentzVector {muonL.Px(), muonL.Py(), muonL.Pz(), muonL.E()}, 13 };
        Particle antimuon   {"antimuon",    LorentzVector {antimuonL.Px(), antimuonL.Py(), antimuonL.Pz(), antimuonL.E()}, -13 };
        Particle positron   {"positron",    LorentzVector {positronL.Px(), positronL.Py(), positronL.Pz(), positronL.E()}, -11 };

        normalizeInput(electron.p4);
        normalizeInput(muon.p4);
        normalizeInput(antimuon.p4);
        normalizeInput(positron.p4);

        LOG(debug) << "e-_E: "     << electron.p4.E();
        LOG(debug) << "e-_px: "    << electron.p4.Px();
        LOG(debug) << "e-_py: "    << electron.p4.Py();
        LOG(debug) << "e-_pz: "    << electron.p4.Pz();

        LOG(debug) << "m-_E: "     << muon.p4.E();
        LOG(debug) << "m-_px: "    << muon.p4.Px();
        LOG(debug) << "m-_py: "    << muon.p4.Py();
        LOG(debug) << "m-_pz: "    << muon.p4.Pz();

        LOG(debug) << "mu+_E: "    << antimuon.p4.E();
        LOG(debug) << "mu+_px: "   << antimuon.p4.Px();
        LOG(debug) << "mu+_py: "   << antimuon.p4.Py();
        LOG(debug) << "mu+_pz: "   << antimuon.p4.Pz();

        LOG(debug) << "e+_E: "    << positron.p4.E();
        LOG(debug) << "e+_px: "   << positron.p4.Px();
        LOG(debug) << "e+_py: "   << positron.p4.Py();
        LOG(debug) << "e+_pz: "   << positron.p4.Pz();

        // calculate weight for theta1 choice:
        LOG(info) << "calculating theta1 higgs width at " << theta1_higgs_width;
        configuration.getGlobalParameters().set("higgs_width", theta1_higgs_width);
        MoMEMta weight(configuration.freeze());
        std::vector<std::pair<double, double>> weights = weight.computeWeights({electron, muon, antimuon, positron});

        double weight_val = weights.back().first;
        double weight_err = weights.back().second;

        LOG(debug) << "Result: " << weights.back().first << " +- " << weights.back().second;
        float theta1_weight = weights.back().first;

        // loop over theta0 choices:
        for (int j=0; j < n_theta0_benchmarks; j++){
            float higgs_width_exp = benchmarksValues[theta0_indices[j]][0];
            float higgs_width = pow(10.0, higgs_width_exp);;
            LOG(info) << "calculating theta0 higgs width at " << higgs_width;

            // Change higgs width
            configuration.getGlobalParameters().set("higgs_width", higgs_width);
            MoMEMta weight(configuration.freeze());

//            auto start_time = system_clock::now();
            weights = weight.computeWeights({electron, muon, antimuon, positron});
//            auto end_time = system_clock::now();

            weight_val = weights.back().first;
            weight_err = weights.back().second;

            LOG(debug) << "Result: " << weight_val << " +- " << weight_err;

            eventWeights.push_back(weight_val);

//            LOG(debug) << "Integration status: " << (int) weight.getIntegrationStatus();
//
//            InputTag dmemInputTag {"dmem", "hist"};
//            bool exists = weight.getPool().exists(dmemInputTag);
//
//            LOG(debug) << "Hist in pool: " << exists;
//
//            if (exists) {
//                Value<TH1D> dmem = weight.getPool().get<TH1D>(dmemInputTag);
//                LOG(debug) << "DMEM integral: " << dmem->Integral();
////                eventWeights.push_back(dmem->Integral());
//                eventWeights.push_back(weight_val);
//            }
//            else {
//                LOG(error) << "BAD WEIGHT CALCULATED: " << i << "; at point: "<< higgs_width;
////                eventWeights.push_back(0.0);
//                Value<TH1D> dmem = weight.getPool().get<TH1D>(dmemInputTag);
////                eventWeights.push_back(dmem->Integral());
//                eventWeights.push_back(weight_val);
//            }

//            LOG(info) << "Weight computed in " << std::chrono::duration_cast<milliseconds>(end_time - start_time).count() << "ms";
            LOG(info) << "finished computing weight: " << higgs_width ;

        }
        LOG(info) << "finished computing event: " << i ;

        std::vector<float> weightRatios;
        for (int j=0; j < n_theta0_benchmarks; j++){
            LOG(info) << "weight: " << eventWeights[j];
            float weightRatio = eventWeights[j]/theta1_weight;
            weightRatios.push_back(weightRatio);

//            outputFile << eventWeights[k] << ",";
            outputFile << weightRatio << ",";

            LOG(info) << "ratio: " << weightRatios[j];
        }
        outputFile << std::endl;

    }
    outputFile.close();

    LOG(info) << "Finished, weights are in: " << weight_output_file;
    return 0;
}
