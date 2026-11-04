#include <mujoco/mujoco.h>

#include <cmath>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

namespace
{

struct ModelDeleter
{
  void operator()(mjModel * model) const
  {
    mj_deleteModel(model);
  }
};

struct DataDeleter
{
  void operator()(mjData * data) const
  {
    mj_deleteData(data);
  }
};

void require(bool condition, const std::string & message)
{
  if (!condition) {
    throw std::runtime_error(message);
  }
}

bool near(double lhs, double rhs, double tolerance = 1e-9)
{
  return std::fabs(lhs - rhs) <= tolerance;
}

std::unique_ptr<mjModel, ModelDeleter> load_model(const std::string & file_name)
{
  const std::string path = std::string(RBKAIROS_MODEL_DIR) + "/" + file_name;
  char error[1024] = {};
  std::unique_ptr<mjModel, ModelDeleter> model(
    mj_loadXML(path.c_str(), nullptr, error, sizeof(error)));
  require(model != nullptr, "failed to compile " + path + ": " + error);
  return model;
}

void check_scene(
  const std::string & file_name, const std::string & tendon_name,
  const std::string & sensor_name, double expected_length0, double expected_upper,
  int translation_axis)
{
  auto model = load_model(file_name);
  require(model->ntendon == 1, file_name + " must contain exactly one tendon");

  const int tendon_id = mj_name2id(model.get(), mjOBJ_TENDON, tendon_name.c_str());
  const int sensor_id = mj_name2id(model.get(), mjOBJ_SENSOR, sensor_name.c_str());
  const int world_joint_id = mj_name2id(model.get(), mjOBJ_JOINT, "world_joint");
  require(tendon_id >= 0 && sensor_id >= 0 && world_joint_id >= 0, "missing model object");
  require(model->tendon_limited[tendon_id], "tendon must be limited");
  require(near(model->tendon_length0[tendon_id], expected_length0), "wrong length0");
  require(near(model->tendon_range[2 * tendon_id], 0.0), "wrong lower limit");
  require(near(model->tendon_range[2 * tendon_id + 1], expected_upper), "wrong upper limit");
  require(
    near(model->tendon_range[2 * tendon_id + 1] - model->tendon_length0[tendon_id], 0.001),
    "configured slack must be one millimeter");
  require(model->tendon_stiffness[tendon_id] == 0.0, "tendon stiffness must be zero");
  require(model->tendon_damping[tendon_id] == 0.0, "tendon damping must be zero");
  require(model->tendon_frictionloss[tendon_id] == 0.0, "tendon friction must be zero");
  require(model->sensor_type[sensor_id] == mjSENS_TENDONLIMITFRC, "wrong sensor type");
  require(model->sensor_objtype[sensor_id] == mjOBJ_TENDON, "wrong sensor object type");
  require(model->sensor_objid[sensor_id] == tendon_id, "sensor must observe the tendon");
  require(model->sensor_dim[sensor_id] == 1, "tendon limit force sensor must be scalar");
  require(model->sensor_needstage[sensor_id] == mjSTAGE_ACC, "wrong sensor stage");
  require(
    model->sensor_datatype[sensor_id] == mjDATATYPE_REAL,
    "MuJoCo tendon limit force sensor must use the REAL datatype");
  require(model->sensor_adr[sensor_id] >= 0, "invalid sensor address");
  require(
    model->sensor_adr[sensor_id] + model->sensor_dim[sensor_id] <= model->nsensordata,
    "sensor data range is out of bounds");

  std::unique_ptr<mjData, DataDeleter> data(mj_makeData(model.get()));
  require(data != nullptr, "failed to allocate mjData");
  mj_forward(model.get(), data.get());
  require(data->tendon_efcadr[tendon_id] == -1, "cable must start slack");
  require(near(data->sensordata[model->sensor_adr[sensor_id]], 0.0), "initial force must be zero");

  const int qpos_axis = model->jnt_qposadr[world_joint_id] + translation_axis;
  const int dof_axis = model->jnt_dofadr[world_joint_id] + translation_axis;
  data->qpos[qpos_axis] = 0.002;
  mj_forward(model.get(), data.get());
  require(data->tendon_efcadr[tendon_id] >= 0, "stretched cable must activate its limit");
  require(
    data->sensordata[model->sensor_adr[sensor_id]] > 0.0,
    "stretched cable must report positive limit force");
  require(data->qfrc_constraint[dof_axis] < 0.0, "constraint must pull toward the anchor");

  mj_resetData(model.get(), data.get());
  data->qpos[qpos_axis] = -0.002;
  mj_forward(model.get(), data.get());
  require(data->tendon_efcadr[tendon_id] == -1, "cable must release toward the anchor");
  require(
    near(data->sensordata[model->sensor_adr[sensor_id]], 0.0),
    "released cable must not push");
}

}  // namespace

int main()
{
  try {
    auto base_model = load_model("rbkairos.xml");
    require(base_model->ntendon == 0, "ordinary model must not contain a tendon constraint");
    check_scene(
      "rbkairos_scene_tendon_test_x.xml", "traction_cable_x",
      "traction_cable_x_limit_force", 1.680, 1.681, 0);
    check_scene(
      "rbkairos_scene_tendon_test_y.xml", "traction_cable_y",
      "traction_cable_y_limit_force", 1.890, 1.891, 1);
    std::cout << "MuJoCo " << mj_versionString() << " C++ tendon model checks passed" << std::endl;
    return 0;
  } catch (const std::exception & error) {
    std::cerr << error.what() << std::endl;
    return 1;
  }
}
