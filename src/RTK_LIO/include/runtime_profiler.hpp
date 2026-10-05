#pragma once

#include <chrono>
#include <fstream>
#include <iomanip>
#include <stdexcept>
#include <string>
#include <vector>


namespace runtime_profile {
using Clock = std::chrono::steady_clock;
struct Event {
  const char* component;
  double scan, wall, ms;
  int state;
  size_t backlog;
};
class Recorder {
 public:
  bool enabled = false;
  double scan = 0;
  int state = -1;
  size_t backlog = 0;
  Clock::time_point frame_start;
  std::string path;
  std::vector<Event> events;
  static Recorder& get() { static Recorder recorder; return recorder; }
  void configure(const std::string& output) {
#ifdef ENABLE_RUNTIME_PROFILING
    path = output;
    enabled = !path.empty();
    if (enabled) {
      std::ofstream probe(path);
      if (!probe) throw std::runtime_error("Cannot open profiling output: " + path);
      events.reserve(262144);
    }
#endif
  }
  void record(const char* name, Clock::time_point start, double stamp) {
    if (!enabled) return;
    auto end = Clock::now();
    events.push_back({name, stamp,
        std::chrono::duration<double>(end.time_since_epoch()).count(),
        std::chrono::duration<double, std::milli>(end - start).count(), state, backlog});
  }
  void begin(double stamp, int s, size_t queue) {
    if (!enabled) return;
    scan = stamp; state = s; backlog = queue; frame_start = Clock::now();
    record("dequeue", frame_start, scan);
  }
  void output() { record("dequeue_to_world_output", frame_start, scan); }
  void flush() {
    if (!enabled) return;
    std::ofstream out(path);
    out << "component,scan_stamp,steady_s,duration_ms,state,backlog\n" << std::setprecision(17);
    for (const auto& e : events)
      out << e.component << ',' << e.scan << ',' << e.wall << ',' << e.ms << ','
          << e.state << ',' << e.backlog << '\n';
    if (!out) throw std::runtime_error("Failed to write profiling output: " + path);
  }
};
class ScopedTimer {
 public:
  ScopedTimer(const char* name, double stamp = -1) : name_(name), stamp_(stamp) {
    active_ = Recorder::get().enabled;
    if (active_) start_ = Clock::now();
  }
  ~ScopedTimer() { stop(); }
  void stop() {
    if (!active_) return;
    active_ = false;
    auto& recorder = Recorder::get();
    recorder.record(name_, start_, stamp_ < 0 ? recorder.scan : stamp_);
  }
 private:
  const char* name_;
  double stamp_;
  bool active_ = false;
  Clock::time_point start_;
};
}

#ifdef ENABLE_RUNTIME_PROFILING
#define RUNTIME_TIMER(variable, name) runtime_profile::ScopedTimer variable(name)
#define RUNTIME_TIMER_STAMP(variable, name, stamp) runtime_profile::ScopedTimer variable(name, stamp)
#define RUNTIME_STOP(variable) variable.stop()
#define RUNTIME_ACTION(expression) do { expression; } while (false)
#else
#define RUNTIME_TIMER(variable, name)
#define RUNTIME_TIMER_STAMP(variable, name, stamp)
#define RUNTIME_STOP(variable)
#define RUNTIME_ACTION(expression) do {} while (false)
#endif
