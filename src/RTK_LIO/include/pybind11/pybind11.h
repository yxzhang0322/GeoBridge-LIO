#pragma once

#include <Python.h>
#include <stdexcept>
#include <string>

namespace pybind11 {

class object {
public:
    object() : m_ptr(nullptr) {}

    explicit object(PyObject* ptr, bool borrowed = false) : m_ptr(ptr) {
        if (!m_ptr) {
            throw std::runtime_error("pybind11::object received null pointer");
        }
        if (borrowed) {
            Py_INCREF(m_ptr);
        }
    }

    object(const object& other) : m_ptr(other.m_ptr) {
        Py_XINCREF(m_ptr);
    }

    object(object&& other) noexcept : m_ptr(other.m_ptr) {
        other.m_ptr = nullptr;
    }

    object& operator=(const object& other) {
        if (this != &other) {
            Py_XINCREF(other.m_ptr);
            Py_XDECREF(m_ptr);
            m_ptr = other.m_ptr;
        }
        return *this;
    }

    object& operator=(object&& other) noexcept {
        if (this != &other) {
            Py_XDECREF(m_ptr);
            m_ptr = other.m_ptr;
            other.m_ptr = nullptr;
        }
        return *this;
    }

    virtual ~object() {
        Py_XDECREF(m_ptr);
        m_ptr = nullptr;
    }

    PyObject* ptr() const { return m_ptr; }
    bool is_none() const { return m_ptr == Py_None; }
    explicit operator bool() const { return m_ptr != nullptr; }

    object attr(const char* name) const {
        if (!m_ptr) {
            throw std::runtime_error("pybind11::object::attr called on null object");
        }
        PyObject* attr_obj = PyObject_GetAttrString(m_ptr, name);
        if (!attr_obj) {
            throw std::runtime_error(std::string("pybind11: missing attribute ") + name);
        }
        return object(attr_obj);
    }

protected:
    PyObject* m_ptr;
};

class str : public object {
public:
    str() : object(PyUnicode_FromString("")) {}
    explicit str(const std::string& value) : object(PyUnicode_FromString(value.c_str())) {}
    explicit str(const char* value) : object(PyUnicode_FromString(value)) {}
};

class list : public object {
public:
    list() : object(PyList_New(0)) {}
    explicit list(Py_ssize_t size) : object(PyList_New(size)) {}

    void append(const object& value) {
        if (PyList_Append(m_ptr, value.ptr()) != 0) {
            throw std::runtime_error("pybind11::list append failed");
        }
    }

    Py_ssize_t size() const {
        return PyList_Size(m_ptr);
    }

    object operator[](Py_ssize_t index) const {
        PyObject* item = PyList_GetItem(m_ptr, index);
        if (!item) {
            throw std::runtime_error("pybind11::list index out of range");
        }
        return object(item,              true);
    }
};

class dict : public object {
public:
    dict() : object(PyDict_New()) {}

    void set_item(const std::string& key, const object& value) {
        if (PyDict_SetItemString(m_ptr, key.c_str(), value.ptr()) != 0) {
            throw std::runtime_error("pybind11::dict::set_item failed");
        }
    }
};

class module_ : public object {
public:
    module_() : object(PyImport_AddModule("__main__"),              true) {}
    explicit module_(PyObject* ptr, bool borrowed = false) : object(ptr, borrowed) {}

    static module_ import(const char* name) {
        PyObject* mod = PyImport_ImportModule(name);
        if (!mod) {
            throw std::runtime_error(std::string("pybind11: cannot import module ") + name);
        }
        return module_(mod);
    }
};

class gil_scoped_acquire {
public:
    gil_scoped_acquire() { state_ = PyGILState_Ensure(); }
    ~gil_scoped_acquire() { PyGILState_Release(state_); }

private:
    PyGILState_STATE state_;
};

class gil_scoped_release {
public:
    gil_scoped_release() {
        state_ = PyEval_SaveThread();
    }

    ~gil_scoped_release() {
        PyEval_RestoreThread(state_);
    }

private:
    PyThreadState* state_;
};

}
