#include <momemta/ConfigurationReader.h>
#include <momemta/Logging.h>
#include <momemta/MoMEMta.h>
#include <momemta/Unused.h>
#include "H5Cpp.h"
#include <iostream>
#include <fstream>
#include <TH1D.h>
#include <chrono>


using namespace std::chrono;
using namespace momemta;
using namespace H5;


using LorentzVectorM = ROOT::Math::LorentzVector<ROOT::Math::PtEtaPhiM4D<float>>;
using LorentzVectorE = ROOT::Math::LorentzVector<ROOT::Math::PtEtaPhiE4D<float>>;

// observables: 
// ['j_0_E', 'j_0_pt', 'j_0_eta', 'j_0_phi', 
//  'j_1_E', 'j_1_pt', 'j_1_eta', 'j_1_phi', 
//  'e_0_E', 'e_0_pt', 'e_0_eta', 'e_0_phi',
//  'mu_0_E', 'mu_0_pt', 'mu_0_eta', 'mu_0_phi', 
//  'met_pt', 'met_phi'
//  'm_e_0_j_0', 'm_e_0_j_1', 'm_mu_0_j_0', 'm_mu_0_j_1', 
//  'mt2' ]

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
    lua_parameters.set("USE_TF", true);
    lua_parameters.set("USE_PERM", true);

    ConfigurationReader configuration("../examples/ttbar.lua", lua_parameters);

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
    const int n_artificial_benchmarks = 7;
    const int expected_value_benchmark_position = 12;
    
    DataSpace benchValsSpace = benchValsSet.getSpace();
        
    hsize_t benchmarkDims[2];
    benchValsSpace.getSimpleExtentDims(benchmarkDims, NULL);
    hsize_t o_benchmarks = benchmarkDims[0];
    
    LOG(info) << "o_benchmarks: " << o_benchmarks;
    
    const int rows = 15;
    
    // generate x_test.csv with python:
    // python -c 'import numpy as np; x_test = np.load("/home/zbhatti/codebase/madminer/examples/ttbar2/data/samples/x_test.npy");
    // np.savetxt("/home/zbhatti/codebase/madminer/momemta/inputs/x_test.csv", x_test, delimiter=",")'
    
    std::vector<std::vector<float>> x_test = parse2DCsvFile(x_test_csv_file);
    
    float benchmarksValues[o_benchmarks][2];
    benchValsSet.read(&benchmarksValues[0], PredType::NATIVE_FLOAT, benchValsSpace);
    benchValsSet.close();
        
    LOG(info) << "****Sample of benchmark values****";
    LOG(info) << benchmarksValues[0][0];
    LOG(info) << benchmarksValues[5][0];
    LOG(info) << benchmarksValues[expected_value_benchmark_position][0];
    LOG(info) << benchmarksValues[o_benchmarks -1][0];
    LOG(info) << "****End****";
    
    std::ofstream outputFile;
    outputFile.open(weight_output_file);
    
    for (int k=0; k < o_benchmarks - n_artificial_benchmarks; k++){
        outputFile << benchmarksValues[k][0] << ",";
    }
    outputFile << std::endl;
    
    // loop over events in the numpy file from madminer and add particle lorentz vectors:
    for(int i=0; i < rows; i++){
        std::vector<float> eventWeights;
        LOG(info) << "calculating event " << i << "/" << rows;
        
        // don't evaluate the wide artificial benchmarks:
        for (int j=0; j < o_benchmarks - n_artificial_benchmarks; j++){
            
            float topMass =  benchmarksValues[j][0];
            LOG(info) << "calculating top_mass at " << topMass;
            // Change top mass
            configuration.getGlobalParameters().set("top_mass", topMass); // changed to match madminer approach
            MoMEMta weight(configuration.freeze());
            
            // LorentzVectorE(pt, eta, phi, E)
            LorentzVectorE bjet1L   {x_test[i][1], x_test[i][2], x_test[i][3], x_test[i][0]};
            LorentzVectorE bjet2L   {x_test[i][5], x_test[i][6], x_test[i][7], x_test[i][4]};
            LorentzVectorE eL       {x_test[i][9], x_test[i][10], x_test[i][11], x_test[i][8]};
            LorentzVectorE muL      {x_test[i][13], x_test[i][14], x_test[i][15], x_test[i][12]};
            
            // LorentzVectorM(pt, eta, phi, m)
            LorentzVectorM met_p4M  {x_test[i][16], 0.0, x_test[i][17], 0.0};
            
            // LorentzVector(px, py, pz, E) required for MoMEMta
            Particle bjet1      {"bjet1",       LorentzVector {bjet1L.Px(), bjet1L.Py(), bjet1L.Pz(), bjet1L.E()}, +5 };
            Particle bjet2      {"bjet2",       LorentzVector {bjet2L.Px(), bjet2L.Py(), bjet2L.Pz(), bjet2L.E()}, -5 };
            Particle electron   {"electron",    LorentzVector {eL.Px(), eL.Py(), eL.Pz(), eL.E()}, -11 };
            Particle muon       {"muon",        LorentzVector {muL.Px(), muL.Py(), muL.Pz(), muL.E()}, +13 };
            
            normalizeInput(bjet1.p4);
            normalizeInput(bjet2.p4);
            normalizeInput(electron.p4);
            normalizeInput(muon.p4);
            
            LorentzVector metL { met_p4M.Px(), met_p4M.Py(), met_p4M.Pz(), met_p4M.E() }; 
            
            LOG(debug) << "j_0_E: "     << bjet1.p4.E();
            LOG(debug) << "j_0_px: "    << bjet1.p4.Px();
            LOG(debug) << "j_0_py: "    << bjet1.p4.Py();
            LOG(debug) << "j_0_pz: "    << bjet1.p4.Pz();
            
            
            LOG(debug) << "j_1_E: "     << bjet2.p4.E();
            LOG(debug) << "j_1_px: "    << bjet2.p4.Px();
            LOG(debug) << "j_1_py: "    << bjet2.p4.Py();
            LOG(debug) << "j_1_pz: "    << bjet2.p4.Pz();
            
            LOG(debug) << "e_0_E: "     << electron.p4.E();
            LOG(debug) << "e_0_px: "    << electron.p4.Px();
            LOG(debug) << "e_0_py: "    << electron.p4.Py();
            LOG(debug) << "e_0_pz: "    << electron.p4.Pz();
            
            LOG(debug) << "mu_0_E: "    << muon.p4.E();
            LOG(debug) << "mu_0_px: "   << muon.p4.Px();
            LOG(debug) << "mu_0_py: "   << muon.p4.Py();
            LOG(debug) << "mu_0_pz: "   << muon.p4.Pz();
            
            LOG(debug) << "met_E: "    << metL.E();
            LOG(debug) << "met_px: "   << metL.Px();
            LOG(debug) << "met_py: "   << metL.Py();
            LOG(debug) << "met_pz: "   << metL.Pz();
            
            auto start_time = system_clock::now();
            std::vector<std::pair<double, double>> weights = weight.computeWeights({electron, bjet1, muon, bjet2}, metL);
            auto end_time = system_clock::now();

            LOG(debug) << "Result:";
            for (const auto& r: weights) {
                LOG(debug) << r.first << " +- " << r.second;
            }

            LOG(debug) << "Integration status: " << (int) weight.getIntegrationStatus();

            InputTag dmemInputTag {"dmem", "hist"};
            bool exists = weight.getPool().exists(dmemInputTag);

            LOG(debug) << "Hist in pool: " << exists;

            if (exists) {
                Value<TH1D> dmem = weight.getPool().get<TH1D>(dmemInputTag);
                LOG(debug) << "DMEM integral: " << dmem->Integral();
                eventWeights.push_back(dmem->Integral());
            }
            else {
                LOG(error) << "bad integral event: " << i << "weight: "<< topMass; 
                eventWeights.push_back(0.0);
            }

            LOG(info) << "Weight computed in " << std::chrono::duration_cast<milliseconds>(end_time - start_time).count() << "ms";
            LOG(info) << "finished computing weight: " << j ;
        }
        LOG(info) << "finished computing event: " << i ;
        
        std::vector<float> weightRatios;        
        for (int k=0; k < o_benchmarks - n_artificial_benchmarks; k++){
            LOG(info) << "weight: " << eventWeights[k];
            
            outputFile << eventWeights[k] << ",";
            
            weightRatios.push_back(eventWeights[k]/eventWeights[expected_value_benchmark_position]);
            LOG(info) << "ratio: " << weightRatios[k];
        }
        outputFile << std::endl;
        
    }   
    outputFile.close();
    
    LOG(info) << "Finished, weights are in: " << weight_output_file;
    return 0;
}
