#include "rclcpp/rclcpp.hpp"
#include "rules_cc/cc/runfiles/runfiles.h"
#include "operational-space-control/walter_sr/osc_node.h"

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);

    std::string error;
    std::unique_ptr<rules_cc::cc::runfiles::Runfiles> runfiles(
        rules_cc::cc::runfiles::Runfiles::Create(argv[0], BAZEL_CURRENT_REPOSITORY, &error));

    if (!error.empty()) {
        std::cerr << "Failed to create runfiles: " << error << std::endl;
        return 1;
    }

    std::filesystem::path model_path = 
        runfiles->Rlocation("mujoco-models/models/walter_sr/scene_walter_sr.xml");

    try {
        auto node = std::make_shared<OSCNode>(model_path.string());
        rclcpp::spin(node);
    } catch (const std::exception& e) {
        RCLCPP_FATAL(rclcpp::get_logger("main"), "Exception caught: %s", e.what());
        return 1;
    }

    rclcpp::shutdown();
    return 0;
}