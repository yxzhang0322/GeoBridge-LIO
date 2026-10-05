#include "python_bridge.h"

#include <sstream>
#include <iostream>

namespace {

constexpr const char* kPythonModuleName = "wahba_lio2rtk";

std::string module_directory() {
    return std::string(ROOT_DIR) + "include/wahba/python";
}

std::string fetch_python_error() {
    PyObject *ptype = nullptr, *pvalue = nullptr, *ptraceback = nullptr;
    PyErr_Fetch(&ptype, &pvalue, &ptraceback);
    PyErr_NormalizeException(&ptype, &pvalue, &ptraceback);
    std::ostringstream oss;
    if (ptype) {
        PyObject* type_str = PyObject_Str(ptype);
        if (type_str) {
            oss << PyUnicode_AsUTF8(type_str) << ": ";
            Py_DECREF(type_str);
        }
    }
    if (pvalue) {
        PyObject* value_str = PyObject_Str(pvalue);
        if (value_str) {
            oss << PyUnicode_AsUTF8(value_str);
            Py_DECREF(value_str);
        }
    }
    Py_XDECREF(ptype);
    Py_XDECREF(pvalue);
    Py_XDECREF(ptraceback);
    return oss.str();
}

PyObject* build_timestamp_list(const std::vector<Timepos>& points) {
    PyObject* list = PyList_New(points.size());
    if (!list) return nullptr;
    for (size_t i = 0; i < points.size(); ++i) {
        PyObject* value = PyFloat_FromDouble(points[i].timestamp);
        if (!value) {
            Py_DECREF(list);
            return nullptr;
        }
        PyList_SET_ITEM(list, static_cast<Py_ssize_t>(i), value);
    }
    return list;
}

PyObject* build_position_list(const std::vector<Timepos>& points) {
    PyObject* outer = PyList_New(points.size());
    if (!outer) return nullptr;
    for (size_t i = 0; i < points.size(); ++i) {
        PyObject* inner = PyList_New(3);
        if (!inner) {
            Py_DECREF(outer);
            return nullptr;
        }
        PyList_SET_ITEM(inner, 0, PyFloat_FromDouble(points[i].pos.x()));
        PyList_SET_ITEM(inner, 1, PyFloat_FromDouble(points[i].pos.y()));
        PyList_SET_ITEM(inner, 2, PyFloat_FromDouble(points[i].pos.z()));
        PyList_SET_ITEM(outer, static_cast<Py_ssize_t>(i), inner);
    }
    return outer;
}

PyObject* build_vector_list(const std::vector<Eigen::Vector3d>& data) {
    PyObject* outer = PyList_New(data.size());
    if (!outer) return nullptr;
    for (size_t i = 0; i < data.size(); ++i) {
        PyObject* inner = PyList_New(3);
        if (!inner) {
            Py_DECREF(outer);
            return nullptr;
        }
        PyList_SET_ITEM(inner, 0, PyFloat_FromDouble(data[i].x()));
        PyList_SET_ITEM(inner, 1, PyFloat_FromDouble(data[i].y()));
        PyList_SET_ITEM(inner, 2, PyFloat_FromDouble(data[i].z()));
        PyList_SET_ITEM(outer, static_cast<Py_ssize_t>(i), inner);
    }
    return outer;
}

PyObject* build_rotation_list(const Eigen::Matrix3d& R) {
    PyObject* outer = PyList_New(3);
    if (!outer) return nullptr;
    for (int i = 0; i < 3; ++i) {
        PyObject* row = PyList_New(3);
        if (!row) {
            Py_DECREF(outer);
            return nullptr;
        }
        for (int j = 0; j < 3; ++j) {
            PyList_SET_ITEM(row, j, PyFloat_FromDouble(R(i, j)));
        }
        PyList_SET_ITEM(outer, i, row);
    }
    return outer;
}

bool convert_vector3_sequence(PyObject* obj, std::vector<Eigen::Vector3d>& out) {
    PyObject* seq = PySequence_Fast(obj, "expected sequence for velocity result");
    if (!seq) {
        return false;
    }
    Py_ssize_t len = PySequence_Fast_GET_SIZE(seq);
    out.clear();
    out.reserve(static_cast<size_t>(len));
    for (Py_ssize_t i = 0; i < len; ++i) {
        PyObject* item = PySequence_Fast_GET_ITEM(seq, i);
        PyObject* row = PySequence_Fast(item, "expected 3D velocity entry");
        if (!row) {
            Py_DECREF(seq);
            return false;
        }
        if (PySequence_Fast_GET_SIZE(row) != 3) {
            Py_DECREF(row);
            Py_DECREF(seq);
            return false;
        }
        Eigen::Vector3d vel;
        for (int j = 0; j < 3; ++j) {
            PyObject* value = PySequence_Fast_GET_ITEM(row, j);
            vel(j) = PyFloat_AsDouble(value);
            if (PyErr_Occurred()) {
                Py_DECREF(row);
                Py_DECREF(seq);
                return false;
            }
        }
        Py_DECREF(row);
        out.push_back(vel);
    }
    Py_DECREF(seq);
    return true;
}

bool convert_rotation(PyObject* obj, Eigen::Matrix3d& R) {
    PyObject* seq = PySequence_Fast(obj, "expected rotation matrix");
    if (!seq) return false;
    if (PySequence_Fast_GET_SIZE(seq) != 3) {
        Py_DECREF(seq);
        return false;
    }
    for (Py_ssize_t i = 0; i < 3; ++i) {
        PyObject* row_obj = PySequence_Fast_GET_ITEM(seq, i);
        PyObject* row = PySequence_Fast(row_obj, "expected rotation row");
        if (!row) {
            Py_DECREF(seq);
            return false;
        }
        if (PySequence_Fast_GET_SIZE(row) != 3) {
            Py_DECREF(row);
            Py_DECREF(seq);
            return false;
        }
        for (Py_ssize_t j = 0; j < 3; ++j) {
            PyObject* value = PySequence_Fast_GET_ITEM(row, j);
            R(i, j) = PyFloat_AsDouble(value);
            if (PyErr_Occurred()) {
                Py_DECREF(row);
                Py_DECREF(seq);
                return false;
            }
        }
        Py_DECREF(row);
    }
    Py_DECREF(seq);
    return true;
}

bool convert_bool_mask(PyObject* obj, std::vector<bool>& mask) {
    PyObject* seq = PySequence_Fast(obj, "expected bool mask sequence");
    if (!seq) return false;
    Py_ssize_t len = PySequence_Fast_GET_SIZE(seq);
    mask.clear();
    mask.reserve(static_cast<size_t>(len));
    for (Py_ssize_t i = 0; i < len; ++i) {
        PyObject* value = PySequence_Fast_GET_ITEM(seq, i);
        int truth = PyObject_IsTrue(value);
        if (truth == -1) {
            Py_DECREF(seq);
            return false;
        }
        mask.push_back(truth != 0);
    }
    Py_DECREF(seq);
    return true;
}

void extract_numeric_field(PyObject* dict_obj, const char* key, std::map<std::string, double>& stats) {
    if (!dict_obj || !PyDict_Check(dict_obj)) return;
    PyObject* value = PyDict_GetItemString(dict_obj, key);
    if (!value) return;
    double number = PyFloat_AsDouble(value);
    if (PyErr_Occurred()) {
        PyErr_Clear();
        long lng = PyLong_AsLong(value);
        if (!PyErr_Occurred()) {
            number = static_cast<double>(lng);
        } else {
            PyErr_Clear();
            return;
        }
    }
    stats[key] = number;
}

}

WahbaPythonBridge& WahbaPythonBridge::instance() {
    static WahbaPythonBridge bridge;
    return bridge;
}

WahbaPythonBridge::WahbaPythonBridge() {
    initialize();
}

bool WahbaPythonBridge::initialize() {
    if (attempted_init_) {
        return available_;
    }
    attempted_init_ = true;
    available_ = ensure_module();
    return available_;
}

bool WahbaPythonBridge::append_module_path() {
    try {
        pybind11::module_ sys = pybind11::module_::import("sys");
        pybind11::object path = sys.attr("path");
        PyObject* path_list = path.ptr();
        std::string dir = module_directory();
        PyObject* dir_obj = PyUnicode_FromString(dir.c_str());
        if (!dir_obj) return false;
        bool already_present = PySequence_Contains(path_list, dir_obj) == 1;
        if (!already_present) {
            if (PyList_Append(path_list, dir_obj) != 0) {
                Py_DECREF(dir_obj);
                return false;
            }
        }
        Py_DECREF(dir_obj);
        return true;
    } catch (const std::exception& e) {
        return false;
    }
}

bool WahbaPythonBridge::fetch_attributes() {
    try {
        bspline_func_ = module_.attr("bspline_velocities_for_cpp");
        tls_func_ = module_.attr("tls_position_refinement_for_cpp");
    } catch (const std::exception& e) {
        return false;
    }
    return true;
}

bool WahbaPythonBridge::ensure_module() {
    pybind11::gil_scoped_acquire gil;
    if (!append_module_path()) {
        return false;
    }
    try {
        module_ = pybind11::module_::import(kPythonModuleName);
    } catch (const std::exception& e) {
        return false;
    }
    return fetch_attributes();
}

bool WahbaPythonBridge::compute_bspline_velocities(
    const std::vector<Timepos>& points,
    std::vector<Eigen::Vector3d>& velocities,
    std::vector<Eigen::Vector3d>* spline_positions,
    const std::string& label,
    double smoothing_factor,
    std::string* error) {
    if (!available_) {
        if (error) *error = "Python bridge unavailable";
        return false;
    }

    pybind11::gil_scoped_acquire gil;
    PyObject* positions = build_position_list(points);
    PyObject* timestamps = build_timestamp_list(points);
    PyObject* smoothing = PyFloat_FromDouble(smoothing_factor);
    if (!positions || !timestamps || !smoothing) {
        Py_XDECREF(positions);
        Py_XDECREF(timestamps);
        Py_XDECREF(smoothing);
        if (error) *error = "Failed to build python arguments";
        return false;
    }

    PyObject* result = PyObject_CallFunctionObjArgs(
        bspline_func_.ptr(),
        positions,
        timestamps,
        smoothing,
        NULL);
    Py_DECREF(positions);
    Py_DECREF(timestamps);
    Py_DECREF(smoothing);

    if (!result) {
        std::string err = fetch_python_error();
        if (error) *error = err;
        return false;
    }

    if (!PyDict_Check(result)) {
        Py_DECREF(result);
        if (error) *error = "Expected dict result from python";
        return false;
    }

    PyObject* vel_obj = PyDict_GetItemString(result, "velocities");
    if (!vel_obj) {
        Py_DECREF(result);
        if (error) *error = "Missing velocities entry";
        return false;
    }

    bool ok = convert_vector3_sequence(vel_obj, velocities);
    if (!ok) {
        Py_DECREF(result);
        if (error) *error = "Unable to parse velocity result";
        return false;
    }

    if (spline_positions) {
        spline_positions->clear();
        PyObject* spline_obj = PyDict_GetItemString(result, "spline");
        if (spline_obj && spline_obj != Py_None) {
            if (!convert_vector3_sequence(spline_obj, *spline_positions)) {
                spline_positions->clear();
            }
        }
    }

    Py_DECREF(result);

    (void)label;
    return true;
}

bool WahbaPythonBridge::tls_refine(
    const std::vector<Eigen::Vector3d>& lio_points,
    const std::vector<Eigen::Vector3d>& rtk_points,
    const Eigen::Matrix3d& initial_rotation,
    double sigma,
    double c_bar_squared,
    int max_iter,
    bool use_adaptive_threshold,
    double adaptive_scale,
    Eigen::Matrix3d& refined_rotation,
    std::vector<bool>& inlier_mask,
    std::map<std::string, double>* stats,
    std::string* error) {
    if (!available_) {
        if (error) *error = "Python bridge unavailable";
        return false;
    }
    pybind11::gil_scoped_acquire gil;

    PyObject* lio_list = build_vector_list(lio_points);
    PyObject* rtk_list = build_vector_list(rtk_points);
    PyObject* rot_list = build_rotation_list(initial_rotation);
    PyObject* sigma_obj = PyFloat_FromDouble(sigma);
    PyObject* cbar_obj = PyFloat_FromDouble(c_bar_squared);
    PyObject* iter_obj = PyLong_FromLong(max_iter);
    PyObject* adaptive_obj = PyBool_FromLong(use_adaptive_threshold ? 1 : 0);
    PyObject* scale_obj = PyFloat_FromDouble(adaptive_scale);
    PyObject* verbose_obj = PyBool_FromLong(0);

    if (!lio_list || !rtk_list || !rot_list || !sigma_obj || !cbar_obj || !iter_obj ||
        !adaptive_obj || !scale_obj || !verbose_obj) {
        Py_XDECREF(lio_list);
        Py_XDECREF(rtk_list);
        Py_XDECREF(rot_list);
        Py_XDECREF(sigma_obj);
        Py_XDECREF(cbar_obj);
        Py_XDECREF(iter_obj);
        Py_XDECREF(adaptive_obj);
        Py_XDECREF(scale_obj);
        Py_XDECREF(verbose_obj);
        if (error) *error = "Failed to build TLS arguments";
        return false;
    }

    PyObject* result = PyObject_CallFunctionObjArgs(
        tls_func_.ptr(),
        lio_list,
        rtk_list,
        rot_list,
        sigma_obj,
        cbar_obj,
        iter_obj,
        adaptive_obj,
        scale_obj,
        verbose_obj,
        NULL);

    Py_DECREF(lio_list);
    Py_DECREF(rtk_list);
    Py_DECREF(rot_list);
    Py_DECREF(sigma_obj);
    Py_DECREF(cbar_obj);
    Py_DECREF(iter_obj);
    Py_DECREF(adaptive_obj);
    Py_DECREF(scale_obj);
    Py_DECREF(verbose_obj);

    if (!result) {
        std::string err = fetch_python_error();
        if (error) *error = err;
        return false;
    }

    if (!PyTuple_Check(result) || PyTuple_Size(result) != 3) {
        Py_DECREF(result);
        if (error) *error = "Unexpected TLS return format";
        return false;
    }

    PyObject* rot_obj = PyTuple_GetItem(result, 0);
    PyObject* mask_obj = PyTuple_GetItem(result, 1);
    PyObject* stats_obj = PyTuple_GetItem(result, 2);

    bool ok_rot = convert_rotation(rot_obj, refined_rotation);
    bool ok_mask = convert_bool_mask(mask_obj, inlier_mask);
    if (!ok_rot || !ok_mask) {
        Py_DECREF(result);
        if (error) *error = "Failed to parse TLS outputs";
        return false;
    }

    if (stats && stats_obj && PyDict_Check(stats_obj)) {
        extract_numeric_field(stats_obj, "initial_rmse", *stats);
        extract_numeric_field(stats_obj, "final_rmse", *stats);
        extract_numeric_field(stats_obj, "rmse_improvement", *stats);
        extract_numeric_field(stats_obj, "iterations", *stats);
        extract_numeric_field(stats_obj, "outliers_removed", *stats);
        extract_numeric_field(stats_obj, "initial_inliers", *stats);
        extract_numeric_field(stats_obj, "final_inliers", *stats);
    }

    Py_DECREF(result);
    return true;
}
