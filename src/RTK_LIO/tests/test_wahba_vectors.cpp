#include <gtest/gtest.h>
#include <Eigen/Geometry>
#include <vector>
#include "wahba/alignment.h"

Eigen::Matrix3d wahba_svd(const std::vector<Eigen::Vector3d>&,
                        const std::vector<Eigen::Vector3d>&, const std::vector<double>&);

int main(int argc, char** argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

TEST(WahbaVectors, MapsSourceToReference)
{
    const Eigen::Matrix3d rotation = Eigen::AngleAxisd(0.6, Eigen::Vector3d(1,2,3).normalized()).toRotationMatrix();
    const std::vector<Eigen::Vector3d> source{{1,0,0}, {0,2,0}, {0,0,3}, {2,1,1}};
    std::vector<Eigen::Vector3d> target;
    for (const auto& vector : source) target.push_back(rotation * vector);
    EXPECT_LT((wahba_svd(source, target, {}) - rotation).norm(), 1e-10);
}

TEST(WahbaVectors, ConstantVelocityAndGravityRetainYaw)
{
    const Eigen::Matrix3d rotation = Eigen::AngleAxisd(0.5, Eigen::Vector3d::UnitZ()).toRotationMatrix();
    const std::vector<Eigen::Vector3d> source{{2,0,0}, {2,0,0}, {2,0,0}, {0,0,-1}};
    std::vector<Eigen::Vector3d> target;
    for (const auto& vector : source) target.push_back(rotation * vector);
    EXPECT_LT((wahba_svd(source, target, {1,1,1,5}) - rotation).norm(), 1e-10);
}

TEST(GravityYaw, CollinearPositionsWithTiltedInitialImuFrame)
{
    const Eigen::Matrix3d level = Eigen::AngleAxisd(.08, Eigen::Vector3d::UnitY()).toRotationMatrix();
    const Eigen::Matrix3d rotation = Eigen::AngleAxisd(2.9, Eigen::Vector3d::UnitZ()).toRotationMatrix()*level;
    const Eigen::Vector3d translation(10,20,3);
    std::vector<Eigen::Vector3d> source, target;
    for (int i=0;i<12;++i)
    {
        source.emplace_back(0,3*i,0);
        target.push_back(rotation*source.back()+translation);
    }
    auto fit=gravity_yaw_alignment(source,target,level.transpose()*(-Eigen::Vector3d::UnitZ()));
    ASSERT_TRUE(fit.valid);
    EXPECT_LT((fit.R-rotation).norm(),1e-10);
    EXPECT_LT((fit.t-translation).norm(),1e-10);
    std::fill(source.begin(),source.end(),Eigen::Vector3d::Zero());
    EXPECT_FALSE(gravity_yaw_alignment(source,target,-Eigen::Vector3d::UnitZ()).valid);
}
