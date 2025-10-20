#include "operational-space-control/walter_sr/osc_node.h"

// Your anonymous namespace with Casadi functions goes here.
namespace {
    FunctionOperations Aeq_ops{
        .incref=Aeq_incref, .checkout=Aeq_checkout, .eval=Aeq, .release=Aeq_release, .decref=Aeq_decref};
    FunctionOperations beq_ops{
        .incref=beq_incref, .checkout=beq_checkout, .eval=beq, .release=beq_release, .decref=beq_decref};
    FunctionOperations Aineq_ops{
        .incref=Aineq_incref, .checkout=Aineq_checkout, .eval=Aineq, .release=Aineq_release, .decref=Aineq_decref};
    FunctionOperations bineq_ops{
        .incref=bineq_incref, .checkout=bineq_checkout, .eval=bineq, .release=bineq_release, .decref=bineq_decref};
    FunctionOperations H_ops{
        .incref=H_incref, .checkout=H_checkout, .eval=H, .release=H_release, .decref=H_decref};
    FunctionOperations f_ops{
        .incref=f_incref, .checkout=f_checkout, .eval=f, .release=f_release, .decref=f_decref};
    using AeqParams = FunctionParams<Aeq_SZ_ARG, Aeq_SZ_RES, Aeq_SZ_IW, Aeq_SZ_W, optimization::Aeq_rows, optimization::Aeq_cols, optimization::Aeq_sz, 4>;
    using beqParams = FunctionParams<beq_SZ_ARG, beq_SZ_RES, beq_SZ_IW, beq_SZ_W, optimization::beq_sz, 1, optimization::beq_sz, 4>;
    using AineqParams = FunctionParams<Aineq_SZ_ARG, Aineq_SZ_RES, Aineq_SZ_IW, Aineq_SZ_W, optimization::Aineq_rows, optimization::Aineq_cols, optimization::Aineq_sz, 1>;
    using bineqParams = FunctionParams<bineq_SZ_ARG, bineq_SZ_RES, bineq_SZ_IW, bineq_SZ_W, optimization::bineq_sz, 1, optimization::bineq_sz, 1>;
    using HParams = FunctionParams<H_SZ_ARG, H_SZ_RES, H_SZ_IW, H_SZ_W, optimization::H_rows, optimization::H_cols, optimization::H_sz, 4>;
    using fParams = FunctionParams<f_SZ_ARG, f_SZ_RES, f_SZ_IW, f_SZ_W, optimization::f_sz, 1, optimization::f_sz, 4>;

    // Helper function definitions
    template <typename T>
    bool contains(const std::vector<T>& vec, const T& value) {
        return std::find(vec.begin(), vec.end(), value) != vec.end();
    }
    
    std::vector<int> getSiteIdsOnSameBodyAsGeom(const mjModel* m, int geom_id) {
        std::vector<int> associated_site_ids;
        if (geom_id < 0 || geom_id >= m->ngeom) {
            std::cerr << "Error: Invalid geom ID: " << geom_id << std::endl;
            return associated_site_ids;
        }
        int geom_body_id = m->geom_bodyid[geom_id];
        for (int i = 0; i < m->nsite; ++i) {
            if (m->site_bodyid[i] == geom_body_id) {
                associated_site_ids.push_back(i);
            }
        }
        return associated_site_ids;
    }
    
    std::vector<int> getBinaryRepresentation_std_find(const std::vector<int>& A, const std::vector<int>& B) {
        std::vector<int> C;
        C.reserve(B.size());
        for (int b_element : B) {
            auto it = std::find(A.begin(), A.end(), b_element);
            C.push_back((it != A.end()) ? 1 : 0);
        }
        return C;
    }
}

// Full constructor implementation
OSCNode::OSCNode(const std::string& xml_path)
    : Node("osc_node"),
      xml_path_(xml_path),
      solution_(Vector<optimization::design_vector_size>::Zero()),
      dual_solution_(Vector<optimization::constraint_matrix_rows>::Zero()),
      design_vector_(Vector<optimization::design_vector_size>::Zero()),
      infinity_(OSQP_INFTY),
      big_number_(1e4),
      Abox_(MatrixColMajor<optimization::design_vector_size, optimization::design_vector_size>::Identity()),
      dv_lb_(Vector<optimization::dv_size>::Constant(-infinity_)),
      dv_ub_(Vector<optimization::dv_size>::Constant(infinity_)),
      u_lb_({-1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000}),
      u_ub_({1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000}),
      z_lb_({
          -infinity_, -infinity_, 0.0, -infinity_, -infinity_, 0.0, -infinity_, -infinity_, 0.0, -infinity_, -infinity_, 0.0,
          -infinity_, -infinity_, 0.0, -infinity_, -infinity_, 0.0, -infinity_, -infinity_, 0.0, -infinity_, -infinity_, 0.0}),
      z_ub_({
          infinity_, infinity_, big_number_, infinity_, infinity_, big_number_, infinity_, infinity_, big_number_, infinity_, infinity_, big_number_,
          infinity_, infinity_, big_number_, infinity_, infinity_, big_number_, infinity_, infinity_, big_number_, infinity_, infinity_, big_number_}),
      bineq_lb_(Vector<optimization::bineq_sz>::Constant(-infinity_))

{
    // --- Mujoco initialization ---
    char error[1000];
    mj_model_ = mj_loadXML(xml_path_.c_str(), nullptr, error, 1000);
    if (!mj_model_) {
        RCLCPP_FATAL(this->get_logger(), "Failed to load Mujoco Model: %s", error);
        throw std::runtime_error("Failed to load Mujoco Model.");
    }
    mj_model_->opt.timestep = 0.002;
    mj_data_ = mj_makeData(mj_model_);
    
    // Populate the site and body ID vectors
    for (const std::string_view& site : model::site_list) {
        std::string site_str = std::string(site);
        int id = mj_name2id(mj_model_, mjOBJ_SITE, site_str.data());
        assert(id != -1 && "Site not found in model.");
        sites_.push_back(site_str);
        site_ids_.push_back(id);
    }
    for (const std::string_view& site : model::noncontact_site_list) {
        std::string site_str = std::string(site);
        int id = mj_name2id(mj_model_, mjOBJ_SITE, site_str.data());
        assert(id != -1 && "Site not found in model.");
        noncontact_sites_.push_back(site_str);
        noncontact_site_ids_.push_back(id);
    }
    for (const std::string_view& site : model::contact_site_list) {
        std::string site_str = std::string(site);
        int id = mj_name2id(mj_model_, mjOBJ_SITE, site_str.data());
        assert(id != -1 && "Site not found in model.");
        contact_sites_.push_back(site_str);
        contact_site_ids_.push_back(id);
    }
    for (const std::string_view& body : model::body_list) {
        std::string body_str = std::string(body);
        int id = mj_name2id(mj_model_, mjOBJ_BODY, body_str.data());
        assert(id != -1 && "Body not found in model.");
        bodies_.push_back(body_str);
        body_ids_.push_back(id);
    }
    assert(site_ids_.size() == body_ids_.size() && "Number of Sites and Bodies must be equal.");

    // --- Optimization Initialization ---
    // Create an initial state message to use for setup.
    OSCMujocoState initial_state_msg;
    // initial_state_msg.motor_position.assign(model::nu_size, 0.0);
    // initial_state_msg.motor_velocity.assign(model::nu_size, 0.0);
    // initial_state_msg.torque_estimate.assign(model::nu_size, 0.0);
    // initial_state_msg.body_rotation.assign(4, 0.0);
    // initial_state_msg.linear_body_velocity.assign(3, 0.0);
    // initial_state_msg.angular_body_velocity.assign(3, 0.0);
    // initial_state_msg.contact_mask.assign(model::contact_site_ids_size, 0.0);

    std::fill(initial_state_msg.motor_position.begin(), initial_state_msg.motor_position.end(), 0.0f);
    std::fill(initial_state_msg.motor_velocity.begin(), initial_state_msg.motor_velocity.end(), 0.0f);
    std::fill(initial_state_msg.torque_estimate.begin(), initial_state_msg.torque_estimate.end(), 0.0f);
    std::fill(initial_state_msg.body_rotation.begin(), initial_state_msg.body_rotation.end(), 0.0f);
    std::fill(initial_state_msg.linear_body_velocity.begin(), initial_state_msg.linear_body_velocity.end(), 0.0f);
    std::fill(initial_state_msg.angular_body_velocity.begin(), initial_state_msg.angular_body_velocity.end(), 0.0f);
    std::fill(initial_state_msg.contact_mask.begin(), initial_state_msg.contact_mask.end(), false);
    
    
    state_callback(std::make_shared<OSCMujocoState>(initial_state_msg));
    
    update_mj_data();


    Vector<model::nq_size> qpos = Eigen::Map<Vector<model::nq_size>>(mj_data_->qpos);
    initial_position_ = qpos(Eigen::seqN(0, 3));    
    

    absl::Status result = set_up_optimization();
    if (!result.ok()) {
        RCLCPP_FATAL(this->get_logger(), "Failed to initialize optimization: %s", result.message().data());
        throw std::runtime_error("Failed to initialize optimization.");
    }

    // --- ROS 2 communication setup ---
    state_subscriber_ = this->create_subscription<OSCMujocoState>(
        "osc/mujoco_state", 10, std::bind(&OSCNode::state_callback, this, std::placeholders::_1));
    // taskspace_targets_subscriber_ = this->create_subscription<OSCTaskspaceTargets>(
    //     "osc/taskspace_targets", 10, std::bind(&OSCNode::taskspace_targets_callback, this, std::placeholders::_1));
    torque_publisher_ = this->create_publisher<OSCTorqueCommand>("osc/torque_command", 10);

    timer_ = this->create_wall_timer(std::chrono::microseconds(2000), std::bind(&OSCNode::timer_callback, this));
}

OSCNode::~OSCNode() {
    mj_deleteData(mj_data_);
    mj_deleteModel(mj_model_);
}

// Full implementation of all methods
void OSCNode::state_callback(const OSCMujocoState::SharedPtr msg) {
    std::lock_guard<std::mutex> lock(state_mutex_);
    // Manually copy and cast each member to the correct double type
    for (size_t i = 0; i < model::nu_size; ++i) {
        state_.motor_position(i) = static_cast<double>(msg->motor_position[i]);
        state_.motor_velocity(i) = static_cast<double>(msg->motor_velocity[i]);
        state_.torque_estimate(i) = static_cast<double>(msg->torque_estimate[i]);
    }
    
    for (size_t i = 0; i < 4; ++i) {
        state_.body_rotation(i) = static_cast<double>(msg->body_rotation[i]);
    }

    for (size_t i = 0; i < 3; ++i) {
        state_.linear_body_velocity(i) = static_cast<double>(msg->linear_body_velocity[i]);
        state_.angular_body_velocity(i) = static_cast<double>(msg->angular_body_velocity[i]);
    }

    for (size_t i = 0; i < model::contact_site_ids_size; ++i) {
        state_.contact_mask(i) = static_cast<double>(msg->contact_mask[i]);
    }
}

// void OSCNode::taskspace_targets_callback(const OSCTaskspaceTargets::SharedPtr msg) {
//     std::lock_guard<std::mutex> lock(taskspace_targets_mutex_);
//     taskspace_targets_ = Eigen::Map<Matrix<model::site_ids_size, 6>>(msg->targets.data());
// }

void OSCNode::timer_callback() {
    std::lock_guard<std::mutex> lock_state(state_mutex_);
    // Get the current time from the ROS 2 clock
    double current_time = this->now().seconds();

    // Replicate the logic from your original `main` function
    Vector<3> position_target = Vector<3>(
        initial_position_(0) + 0.2 * current_time, 
        initial_position_(1), 
        initial_position_(2)
    );
    Vector<3> velocity_target = Vector<3>(
        0.2, 0.0, 0.0
    );
    Vector<3> body_position = Vector<3>(0.2, 0.0, 0.0);

    Eigen::Quaternion<double> body_rotation = Eigen::Quaternion<double>(state_.body_rotation(0), state_.body_rotation(1), state_.body_rotation(2), state_.body_rotation(3));
    Vector<3> position_error = position_target - body_position;
    Vector<3> velocity_error = velocity_target - state_.linear_body_velocity;
    Vector<3> rotation_error = (Eigen::Quaternion<double>(1, 0, 0, 0) * body_rotation.conjugate()).vec();
    Vector<3> angular_velocity_error = Vector<3>::Zero() - state_.angular_body_velocity;

    double torso_lin_kp = 0.0;
    double torso_lin_kv = 0.0;
    double torso_ang_kp = 0.0;
    double torso_ang_kv = 0.0;        

    Vector<3> linear_control = torso_lin_kp * (position_error) + torso_lin_kv * (velocity_error);
    Vector<3> angular_control = torso_ang_kp * (rotation_error) + torso_ang_kv * (angular_velocity_error);
    Eigen::Vector<double, 6> cmd {linear_control(0), 0, 0, angular_control(0), angular_control(1), angular_control(2)};

    // Update the task-space targets member variable directly
    taskspace_targets_.row(0) = cmd;
    update_mj_data();
    update_osc_data();
    update_optimization_data();
    std::ignore = update_optimization();
    solve_optimization();
    publish_torque_command();
}

void OSCNode::update_mj_data() {
    Vector<model::nq_size> qpos = Vector<model::nq_size>::Zero();
    Vector<model::nv_size> qvel = Vector<model::nv_size>::Zero();

    if constexpr (is_fixed_based) {
        qpos << state_.motor_position;
        qvel << state_.motor_velocity;
    } else {
        const Vector<3> zero_vector = {0.0, 0.0, 0.0};
        qpos << zero_vector, state_.body_rotation, state_.motor_position;
        qvel << state_.linear_body_velocity, state_.angular_body_velocity, state_.motor_velocity;
    }

    mj_data_->qpos = qpos.data();
    mj_data_->qvel = qvel.data();

    mj_fwdPosition(mj_model_, mj_data_);
    mj_fwdVelocity(mj_model_, mj_data_);

    points_ = Eigen::Map<Matrix<model::site_ids_size, 3>>(mj_data_->site_xpos)(site_ids_, Eigen::placeholders::all);
}

void OSCNode::update_osc_data() {
    Matrix<model::nv_size, model::nv_size> mass_matrix = Matrix<model::nv_size, model::nv_size>::Zero();
    mj_fullM(mj_model_, mass_matrix.data(), mj_data_->qM);
    Vector<model::nv_size> coriolis_matrix = Eigen::Map<Vector<model::nv_size>>(mj_data_->qfrc_bias);
    Vector<model::nq_size> generalized_positions = Eigen::Map<Vector<model::nq_size>>(mj_data_->qpos);
    Vector<model::nv_size> generalized_velocities = Eigen::Map<Vector<model::nv_size>>(mj_data_->qvel);

    Matrix<optimization::p_size, model::nv_size> jacobian_translation = Matrix<optimization::p_size, model::nv_size>::Zero();
    Matrix<optimization::r_size, model::nv_size> jacobian_rotation = Matrix<optimization::r_size, model::nv_size>::Zero();
    Matrix<optimization::p_size, model::nv_size> jacobian_dot_translation = Matrix<optimization::p_size, model::nv_size>::Zero();
    Matrix<optimization::r_size, model::nv_size> jacobian_dot_rotation = Matrix<optimization::r_size, model::nv_size>::Zero();
    
    for (int i = 0; i < model::body_ids_size; i++) {
        Matrix<3, model::nv_size> jacp = Matrix<3, model::nv_size>::Zero();
        Matrix<3, model::nv_size> jacr = Matrix<3, model::nv_size>::Zero();
        Matrix<3, model::nv_size> jacp_dot = Matrix<3, model::nv_size>::Zero();
        Matrix<3, model::nv_size> jacr_dot = Matrix<3, model::nv_size>::Zero();
        mj_jac(mj_model_, mj_data_, jacp.data(), jacr.data(), points_.row(i).data(), body_ids_[i]);
        mj_jacDot(mj_model_, mj_data_, jacp_dot.data(), jacr_dot.data(), points_.row(i).data(), body_ids_[i]);
        int row_offset = i * 3;
        for(int row_idx = 0; row_idx < 3; row_idx++) {
            for(int col_idx = 0; col_idx < model::nv_size; col_idx++) {
                jacobian_translation(row_idx + row_offset, col_idx) = jacp(row_idx, col_idx);
                jacobian_rotation(row_idx + row_offset, col_idx) = jacr(row_idx, col_idx);
                jacobian_dot_translation(row_idx + row_offset, col_idx) = jacp_dot(row_idx, col_idx);
                jacobian_dot_rotation(row_idx + row_offset, col_idx) = jacr_dot(row_idx, col_idx);
            }
        }
    }
    
    Matrix<optimization::s_size, model::nv_size> taskspace_jacobian = Matrix<optimization::s_size, model::nv_size>::Zero();
    Matrix<optimization::s_size, model::nv_size> jacobian_dot = Matrix<optimization::s_size, model::nv_size>::Zero();
    taskspace_jacobian << jacobian_translation, jacobian_rotation;
    jacobian_dot << jacobian_dot_translation, jacobian_dot_rotation;
    Vector<optimization::s_size> taskspace_bias = Vector<optimization::s_size>::Zero();
    taskspace_bias = jacobian_dot * generalized_velocities;
    Matrix<model::nv_size, optimization::z_size> contact_jacobian = Matrix<model::nv_size, optimization::z_size>::Zero();
    contact_jacobian = jacobian_translation(Eigen::seq(Eigen::placeholders::end - Eigen::fix<optimization::z_size>, Eigen::placeholders::last), Eigen::placeholders::all).transpose();

    osc_data_.mass_matrix = mass_matrix;
    osc_data_.coriolis_matrix = coriolis_matrix;
    osc_data_.contact_jacobian = contact_jacobian;
    osc_data_.taskspace_jacobian = taskspace_jacobian;
    osc_data_.taskspace_bias = taskspace_bias;
    osc_data_.previous_q = generalized_positions;
    osc_data_.previous_qd = generalized_velocities;
}

void OSCNode::update_optimization_data() {
    auto mass_matrix = matrix_utils::transformMatrix<double, model::nv_size, model::nv_size, matrix_utils::ColumnMajor>(osc_data_.mass_matrix.data());
    auto coriolis_matrix = matrix_utils::transformMatrix<double, model::nv_size, 1, matrix_utils::ColumnMajor>(osc_data_.coriolis_matrix.data());
    auto contact_jacobian = matrix_utils::transformMatrix<double, model::nv_size, optimization::z_size, matrix_utils::ColumnMajor>(osc_data_.contact_jacobian.data());
    auto taskspace_jacobian = matrix_utils::transformMatrix<double, optimization::s_size, model::nv_size, matrix_utils::ColumnMajor>(osc_data_.taskspace_jacobian.data());
    auto taskspace_bias = matrix_utils::transformMatrix<double, optimization::s_size, 1, matrix_utils::ColumnMajor>(osc_data_.taskspace_bias.data());
    auto desired_taskspace_ddx = matrix_utils::transformMatrix<double, model::site_ids_size, 6, matrix_utils::ColumnMajor>(taskspace_targets_.data());
    
    auto Aeq_matrix = evaluate_function<AeqParams>(Aeq_ops, {design_vector_.data(), mass_matrix.data(), coriolis_matrix.data(), contact_jacobian.data()});
    auto beq_matrix = evaluate_function<beqParams>(beq_ops, {design_vector_.data(), mass_matrix.data(), coriolis_matrix.data(), contact_jacobian.data()});
    auto Aineq_matrix = evaluate_function<AineqParams>(Aineq_ops, {design_vector_.data()});
    auto bineq_matrix = evaluate_function<bineqParams>(bineq_ops, {design_vector_.data()});
    auto H_matrix = evaluate_function<HParams>(H_ops, {design_vector_.data(), desired_taskspace_ddx.data(), taskspace_jacobian.data(), taskspace_bias.data()});
    auto f_matrix = evaluate_function<fParams>(f_ops, {design_vector_.data(), desired_taskspace_ddx.data(), taskspace_jacobian.data(), taskspace_bias.data()});

    opt_data_.H = H_matrix;
    opt_data_.f = f_matrix;
    opt_data_.Aeq = Aeq_matrix;
    opt_data_.beq = beq_matrix;
    opt_data_.Aineq = Aineq_matrix;
    opt_data_.bineq = bineq_matrix;
}

absl::Status OSCNode::set_up_optimization() {
    MatrixColMajor<optimization::constraint_matrix_rows, optimization::constraint_matrix_cols> A;
    A << opt_data_.Aeq, opt_data_.Aineq, Abox_;
    Vector<optimization::bounds_size> lb;
    Vector<optimization::bounds_size> ub;
    Vector<optimization::z_size> z_lb_masked = z_lb_;
    Vector<optimization::z_size> z_ub_masked = z_ub_;
    {
        std::lock_guard<std::mutex> lock(state_mutex_);
        for(int i = 0; i < model::contact_site_ids_size; i++) {
            z_lb_masked(Eigen::seqN(3 * i, 3)) *= state_.contact_mask(i);
            z_ub_masked(Eigen::seqN(3 * i, 3)) *= state_.contact_mask(i);
        }
    }
    lb << opt_data_.beq, bineq_lb_, dv_lb_, u_lb_, z_lb_masked;
    ub << opt_data_.beq, opt_data_.bineq, dv_ub_, u_ub_, z_ub_masked;
    
    Eigen::SparseMatrix<double> sparse_H = opt_data_.H.sparseView();
    Eigen::SparseMatrix<double> sparse_A = A.sparseView();
    sparse_H.makeCompressed();
    sparse_A.makeCompressed();

    instance_.objective_matrix = sparse_H;
    instance_.objective_vector = opt_data_.f;
    instance_.constraint_matrix = sparse_A;
    instance_.lower_bounds = lb;
    instance_.upper_bounds = ub;
    
    absl::Status result = solver_.Init(instance_, settings_);
    return result;
}

absl::Status OSCNode::update_optimization() {
    MatrixColMajor<optimization::constraint_matrix_rows, optimization::constraint_matrix_cols> A;
    A << opt_data_.Aeq, opt_data_.Aineq, Abox_;
    Vector<optimization::bounds_size> lb;
    Vector<optimization::bounds_size> ub;
    Vector<optimization::z_size> z_lb_masked = z_lb_;
    Vector<optimization::z_size> z_ub_masked = z_ub_;

    
    {
        // std::lock_guard<std::mutex> lock(state_mutex_);
        for(int i = 0; i < model::contact_site_ids_size; i++) {
            z_lb_masked(Eigen::seqN(3 * i, 3)) *= state_.contact_mask(i);
            z_ub_masked(Eigen::seqN(3 * i, 3)) *= state_.contact_mask(i);
        }
    }

    std::cout << "here" << std::endl;

    
    lb << opt_data_.beq, bineq_lb_, dv_lb_, u_lb_, z_lb_masked;
    ub << opt_data_.beq, opt_data_.bineq, dv_ub_, u_ub_, z_ub_masked;
    
    Eigen::SparseMatrix<double> sparse_H = opt_data_.H.sparseView();
    Eigen::SparseMatrix<double> sparse_A = A.sparseView();
    sparse_H.makeCompressed();
    sparse_A.makeCompressed();



    absl::Status result;
    auto sparsity_check = solver_.UpdateObjectiveAndConstraintMatrices(sparse_H, sparse_A);
    if(sparsity_check.ok()) {
        result.Update(solver_.SetObjectiveVector(opt_data_.f));
        result.Update(solver_.SetBounds(lb, ub));
    } else {
        instance_.objective_matrix = sparse_H;
        instance_.objective_vector = opt_data_.f;
        instance_.constraint_matrix = sparse_A;
        instance_.lower_bounds = lb;
        instance_.upper_bounds = ub;
        result.Update(solver_.Init(instance_, settings_));
        result.Update(solver_.SetWarmStart(solution_, dual_solution_));
    }

    return result;
}

void OSCNode::solve_optimization() {
    exit_code_ = solver_.Solve();
    solution_ = solver_.primal_solution();
    dual_solution_ = solver_.dual_solution();
}

void OSCNode::reset_optimization() {
    Vector<optimization::constraint_matrix_cols> primal_vector = Vector<optimization::constraint_matrix_cols>::Zero();
    Vector<optimization::constraint_matrix_rows> dual_vector = Vector<optimization::constraint_matrix_rows>::Zero();
    std::ignore = solver_.SetWarmStart(primal_vector, dual_vector);
}

void OSCNode::publish_torque_command() {
    Vector<model::nu_size> torque_command = solution_(Eigen::seqN(optimization::dv_idx, optimization::u_size));
    auto torque_msg = std::make_unique<OSCTorqueCommand>();
    for (size_t i = 0; i < torque_command.size(); ++i) {
        torque_msg->torque_command[i] = static_cast<float>(torque_command(i));
    }
    torque_publisher_->publish(std::move(torque_msg));
    std::cout << "Published torque command: ";
    for (size_t i = 0; i < torque_command.size(); ++i) {
        std::cout << torque_command(i) << " ";
    }
    std::cout << std::endl;  
}