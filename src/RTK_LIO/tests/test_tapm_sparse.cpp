#include <gtest/gtest.h>
#include <tapm.hpp>

int main(int argc, char** argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

TEST(TapmWindow, OnlyTerminalStateHasRtkConstraint)
{
  std::vector<Timepos> history(3);
  std::vector<Eigen::Vector3d> local(3, Eigen::Vector3d::Zero());
  for (size_t i = 0; i < history.size(); ++i)
  {
    history[i].timestamp = 22.0 + .1 * i;
    history[i].pos = Eigen::Vector3d::Constant(5.0);
  }
  Timepos fix = history.back();
  fix.pos = Eigen::Vector3d(1, 2, 3);
  fix.h_acc = fix.v_acc = .1;
  TapmConfig config;
  config.enable = true;
  config.lambda_v = config.lambda_a = 0.0;
  const auto result = solve_tapm_translation_sliding(history, local, fix, 10.0, 22.0, config);
  ASSERT_TRUE(result.success);
  EXPECT_EQ(result.used_rtk, 1u);
  EXPECT_EQ(result.used_tapm, 2u);
  EXPECT_LT((result.position_last - fix.pos).norm(), 1e-6);
}

TEST(TapmWindow, LongOutageReducesAnchorInformation)
{
  std::vector<Timepos> history(4);
  std::vector<Eigen::Vector3d> local(4, Eigen::Vector3d::Zero());
  for (size_t i = 0; i < history.size(); ++i)
  {
    history[i].timestamp = 22.0 + .1 * i;
    history[i].pos = Eigen::Vector3d::Ones();
  }
  Timepos fix = history.back();
  fix.pos.setZero();
  fix.h_acc = fix.v_acc = .1;
  TapmConfig config;
  config.enable = true;
  config.v_max = config.v_max_z = 2.0;
  config.a_max = config.a_max_z = 10.0;
  const auto older = solve_tapm_translation_sliding(history, local, fix, 10.0, 22.0, config);
  const auto newer = solve_tapm_translation_sliding(history, local, fix, 22.0, 22.0, config);
  ASSERT_TRUE(older.success);
  ASSERT_TRUE(newer.success);
  EXPECT_LT(older.position_last.norm(), newer.position_last.norm());
}

TEST(TapmWindow, MissingLioReferenceIsRejected)
{
  std::vector<Timepos> history(3);
  Timepos fix;
  fix.h_acc = fix.v_acc = .1;
  TapmConfig config;
  config.enable = true;
  const auto result = solve_tapm_translation_sliding(history, {}, fix, 10.0, 22.0, config);
  EXPECT_FALSE(result.success);
}
