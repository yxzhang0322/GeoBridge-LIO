#include <wahba/alignment.h>
#include <wahba/BSpline.h>
#include <gnss_comm/gnss_utility.hpp>
#include <common_lib.h>


class SmoothTraj
{
    public:


    void SmoothTraj();
    void init(int window_traj_size, double time_threshold, double velocity_threshold);
    void getRTKtraj(deque<Timepos> RTKtimepos);
    void getLiDARtraj(deque<Timepos> ekftimepos);
    void computeAlignment(V3D t, M3D R);
    V3D computeFinalError();
    void Process(V3D final_error, deque<Timepos> ekftimepos, deque<Timepos> RTKtimepos);


    private:

    int window_traj_size;
    double time_threshold;
    double velocity_threshold;
    deque<BSplineTimePoseExt> RTKtraj;
    deque<BSplineTimePoseExt> LiDARtraj;
    V3D t_LIO_RTK;
    M3D R_LIO_RTK;
    V3D last_final_error;

};


void SmoothTraj::SmoothTraj()
    : window_traj_size(10), time_threshold(0.05), velocity_threshold(0.05)
{

}

void SmoothTraj::~SmoothTraj () {}


void SmoothTraj::init(int window_traj_size, double time_threshold, double velocity_threshold)
{

    this->window_traj_size = window_traj_size;
    this->time_threshold = time_threshold;
    this->velocity_threshold = velocity_threshold;
}


void SmoothTraj::getRTKtraj(deque<Timepos> RTKtimepos)
{


}


void SmoothTraj::getLiDARtraj(deque<Timepos> ekftimepos)
{

}


void SmoothTraj::computeVelocity()
{

}


void SmoothTraj::computeAlignment(V3D t, M3D R)
{


}


V3D SmoothTraj::computeFinalError()
{

}


void SmoothTraj::Process(V3D final_error, deque<Timepos> ekftimepos, deque<Timepos> RTKtimepos)
{


    if (ekftimepos.empty() || RTKtimepos.empty())
    {
        (void)0;
        (void)0;
        return;
    }

    if (ekftimepos.back().timestamp - RTKtimepos.back().timestamp > time_threshold)
    {
        (void)0;
        return;
    }


    getRTKtraj(RTKtimepos);
    getLiDARtraj(ekftimepos);


    computeAlignment(t_LIO_RTK, R_LIO_RTK);

}

V3D SmoothTraj::computeFinalError()
{

    return final_error;
}
