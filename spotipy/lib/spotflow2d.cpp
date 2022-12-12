#include <Python.h>
#include <math.h>
#include <cmath>
#include <iostream>
#include <stdio.h>
#include <string>

#include "numpy/arrayobject.h"

#ifdef _OPENMP
#include <omp.h>
#endif

#include <nanoflann.hpp>

inline int clip(int n, int lower, int upper)
{
    return std::max(lower, std::min(n, upper));
}

template <typename T> struct Point2D
{
        T x, y;
};

template <typename T> struct Point3D
{
        T x, y, z;
};

template <typename T>
struct PointCloud2D
{

    std::vector<Point2D<T>> pts;
    // Must return the number of data points
    inline size_t kdtree_get_point_count() const { return pts.size(); }
    // Returns the dim'th component of the idx'th point in the class:
    // Since this is inlined and the "dim" argument is typically an immediate value, the
    //  "if/else's" are actually solved at compile time.
    inline T kdtree_get_pt(const size_t idx, const size_t dim) const
    {
        if (dim == 0)
            return pts[idx].x;
        else
            return pts[idx].y;
    }
    // Optional bounding-box computation: return false to default to a standard bbox computation loop.
    //   Return true if the BBOX was already computed by the class and returned in "bb" so it can be avoided to redo it again.
    //   Look at bb.size() to find out the expected dimensionality (e.g. 2 or 3 for point clouds)
    template <class BBOX>
    bool kdtree_get_bbox(BBOX & /* bb */) const { return false; }
};

inline int round_to_int(float r)
{
    return (int)lrint(r);
}

static PyObject *c_spotflow2d(PyObject *self, PyObject *args)
{

    PyArrayObject *points = NULL;
    PyArrayObject *dst = NULL;
    int shape_y, shape_x;
    float scale;

    if (!PyArg_ParseTuple(args, "O!iif", &PyArray_Type, &points, &shape_y, &shape_x, &scale))
        return NULL;

    npy_intp *dims = PyArray_DIMS(points);

    npy_intp dims_dst[3];
    dims_dst[0] = shape_y;
    dims_dst[1] = shape_x;
    dims_dst[2] = 3;

    dst = (PyArrayObject *)PyArray_SimpleNew(3, dims_dst, NPY_FLOAT32);

    // build kdtree

    PointCloud2D<float> cloud;
    float query_point[2];
    nanoflann::SearchParams params;
    std::vector<std::pair<size_t, float>> results;

    cloud.pts.resize(dims[0]);
    for (long i = 0; i < dims[0]; i++)
    {
        cloud.pts[i].y = *(float *)PyArray_GETPTR2(points, i, 0);
        cloud.pts[i].x = *(float *)PyArray_GETPTR2(points, i, 1);
    }

    // construct a kd-tree:
    typedef nanoflann::KDTreeSingleIndexAdaptor<
        nanoflann::L2_Simple_Adaptor<float, PointCloud2D<float>>,
        PointCloud2D<float>, 2>
        my_kd_tree_t;

    // build the index from points
    my_kd_tree_t index(2, cloud, nanoflann::KDTreeSingleIndexAdaptorParams(10 /* max leaf */));

    index.buildIndex();

    const float scale2 = scale * scale;

#ifdef __APPLE__
#pragma omp parallel for
#else
#pragma omp parallel for schedule(dynamic)
#endif
    for (int i = 0; i < dims_dst[0]; i++)
    {
        for (int j = 0; j < dims_dst[1]; j++)
        {

            // get the closest point
            size_t num_results = 1;
            const float query_pt[2] = {(float)j, (float)i};
            std::vector<unsigned long> ret_index(num_results);
            std::vector<float> out_dist_sqr(num_results);

            num_results = index.knnSearch(
                &query_pt[0], num_results, &ret_index[0], &out_dist_sqr[0]);

            // the coords of the closest point
            const float px = cloud.pts[ret_index[0]].x;
            const float py = cloud.pts[ret_index[0]].y;

            const float y = py - i;
            const float x = px - j;

            const float r2 = x * x + y * y;

            // the stereographic embedding
            const float x_prime = 2 * scale * x / (r2 + scale2);
            const float y_prime = 2 * scale * y / (r2 + scale2);
            const float z_prime = -(r2 - scale2) / (r2 + scale2);

            *(float *)PyArray_GETPTR3(dst, i, j, 0) = z_prime;
            *(float *)PyArray_GETPTR3(dst, i, j, 1) = y_prime;
            *(float *)PyArray_GETPTR3(dst, i, j, 2) = x_prime;
        }
    }

    return PyArray_Return(dst);
}

float interp_flow(PyArrayObject *data, const int dim, float y, float x, int Ny, int Nx)
{

    if (x < 0 || x >= Nx || y < 0 || y >= Ny)
        return 0;

    int x0 = (int)floor(x);
    int x1 = (int)ceil(x);
    int y0 = (int)floor(y);
    int y1 = (int)ceil(y);

    float dx = x - x0;
    float dy = y - y0;

    float v00 = *(float *)PyArray_GETPTR3(data, y0, x0, dim);
    float v01 = *(float *)PyArray_GETPTR3(data, y0, x1, dim);
    float v10 = *(float *)PyArray_GETPTR3(data, y1, x0, dim);
    float v11 = *(float *)PyArray_GETPTR3(data, y1, x1, dim);

    float v0 = v00 * (1 - dx) + v01 * dx;
    float v1 = v10 * (1 - dx) + v11 * dx;

    return v0 * (1 - dy) + v1 * dy;
}

static PyObject *c_cluster_flow2d(PyObject *self, PyObject *args)
{

    PyArrayObject *points = NULL;
    PyArrayObject *flow = NULL;
    PyArrayObject *dst_mapped = NULL;
    PyArrayObject *dst = NULL;
    float dt;
    int steps;
    float atol;
    float min_distance;

    if (!PyArg_ParseTuple(args, "O!O!fffi", &PyArray_Type, &points, &PyArray_Type, &flow, &dt, &atol, &min_distance, &steps))
        return NULL;

    npy_intp *dims_points = PyArray_DIMS(points);
    npy_intp *dims_flow = PyArray_DIMS(flow);

    std::vector<float> points_z;
    std::vector<float> points_y;
    std::vector<float> points_x;

    for (long i = 0; i < dims_points[0]; i++)
    {
        float py = *(float *)PyArray_GETPTR2(points, i, 0);
        float px = *(float *)PyArray_GETPTR2(points, i, 1);
        points_z.push_back(interp_flow(flow, 0, py, px, dims_flow[0], dims_flow[1]));
        points_y.push_back(py);
        points_x.push_back(px);
    }

#ifdef __APPLE__
#pragma omp parallel for
#else
#pragma omp parallel for schedule(dynamic)
#endif
    for (long i = 0; i < points_x.size(); i++)
    {
        float py = points_y[i];
        float px = points_x[i];

        for (long n = 0; n < steps; n++)
        {

            float vz = interp_flow(flow, 0, py, px, dims_flow[0], dims_flow[1]);
            float vy = interp_flow(flow, 1, py, px, dims_flow[0], dims_flow[1]);
            float vx = interp_flow(flow, 2, py, px, dims_flow[0], dims_flow[1]);

            vx = vx / (1 + vz);
            vy = vy / (1 + vz);

            float dy = vy * dt;
            float dx = vx * dt;

            py += dy;
            px += dx;
        }

        points_y[i] = py;
        points_x[i] = px;
    }

    // NMS

    
    std::vector<bool> suppressed;


    // argsort points


    for (long i = 0; i < points_x.size(); i++)
    {
        suppressed.push_back(false);
    }

    for (long i = 0; i < points_x.size(); i++)
    {
        for (long j = i + 1; j < points_x.size(); j++)
        {
            if (suppressed[j])
                continue;

            float dy = points_y[i] - points_y[j];
            float dx = points_x[i] - points_x[j];

            if (dy * dy + dx * dx < min_distance * min_distance)
            {
                if (points_z[i] > points_z[j])
                {
                    suppressed[j] = true;
                }
                else
                {
                    suppressed[i] = true;
                }
            }
        }
    }

    npy_intp dims_dst_mapped[2];
    dims_dst_mapped[0] = points_x.size();
    dims_dst_mapped[1] = 2;

    dst_mapped = (PyArrayObject *)PyArray_SimpleNew(2, dims_dst_mapped, NPY_FLOAT32);

    dst = (PyArrayObject *)PyArray_SimpleNew(2, dims_dst_mapped, NPY_FLOAT32);

    for (long i = 0; i < points_x.size(); i++)
    {
        *(float *)PyArray_GETPTR2(dst_mapped, i, 0) = points_y[i];
        *(float *)PyArray_GETPTR2(dst_mapped, i, 1) = points_x[i];
    }

    PyObject *ret = PyTuple_New(2);
    PyTuple_SetItem(ret, 0, PyArray_Return(dst));
    PyTuple_SetItem(ret, 1, PyArray_Return(dst_mapped));

    return PyTuple_Pack(2, PyArray_Return(dst), PyArray_Return(dst_mapped));
}

//------------------------------------------------------------------------

static struct PyMethodDef methods[] = {
    {"c_spotflow2d", c_spotflow2d, METH_VARARGS, "spot flow"},
    {"c_cluster_flow2d", c_cluster_flow2d, METH_VARARGS, "cluster flow"},
    {NULL, NULL, 0, NULL}

};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "spotflow2d",
    NULL,
    -1,
    methods,
    NULL, NULL, NULL, NULL};

PyMODINIT_FUNC PyInit_spotflow2d(void)
{
    import_array();
    return PyModule_Create(&moduledef);
}
