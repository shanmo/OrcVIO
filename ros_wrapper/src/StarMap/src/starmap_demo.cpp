#include <iostream>
#include <tuple>

#include "torch/script.h"
#include "boost/program_options.hpp"
#include "opencv2/opencv.hpp"

#include "starmap/starmap.h" // provides Crop, parseHeatmap, horn87

using namespace std;
namespace bpo = boost::program_options;


/**
 * @brief Parse command line arguments
 * @return command line options
 */
tuple<bool, bpo::variables_map> parse_commandline(const int argc, char** const argv) {
    bpo::options_description desc("Demonstrate running starmap");
    desc.add_options()
            ("help", "produce help message")
            ("loadModel", bpo::value<string>()->default_value("models/model_cpu-jit.pth"),
             "Path to the pre-trained model file")
            ("demo", bpo::value<string>()->default_value("tests/data/car-big.jpg"),
             "Path to an image file to test upon ")
            ("input_res", bpo::value<int>()->default_value(256),
             "The resolution of image that network accepts")
            ("GPU", bpo::value<int>()->default_value(-1),
             "GPU Id to use. For CPU specify -1")
            ;
    bpo::variables_map vm;
    bpo::store(bpo::parse_command_line(argc, argv, desc), vm);
    if (vm.count("help"))
    {
        cout << desc << "\n";
        return make_tuple(false, vm);
    }
    bpo::notify(vm);
    return make_tuple(true, vm);
}


int main(const int argc, char** const argv) {
    // opt = opts().parse()
    bool cont;
    bpo::variables_map opt;
    tie(cont, opt) = parse_commandline(argc, argv);
    if (!cont)
        return 1;

    auto pts = starmap::run_starmap_on_img(
            opt["loadModel"].as<string>(),
            opt["demo"].as<string>(),
            opt["input_res"].as<const int>(),
            opt["GPU"].as<const int>() );
}
