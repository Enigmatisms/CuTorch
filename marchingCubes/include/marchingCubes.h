#pragma once
#include <torch/extension.h>
#include <vector>
#include <cmath>
#include <array>
#include "sdf_kernel.h"

struct Point2d {
    float x;
    float y;
    Point2d(): x(0.0), y(0.0) {};
    Point2d(float _x, float _y): x(_x), y(_y) {}
    Point2d(const Point2d& pt): x(pt.x), y(pt.y) {}

    Point2d operator+(const Point2d& pt) const {
        return Point2d(x + pt.x, y + pt.y);
    }
    Point2d operator-(const Point2d& pt) const {
        return Point2d(x - pt.x, y - pt.y);
    }
    Point2d operator*(float factor) const {
        return Point2d(x * factor, y * factor);
    }
    float dot(const Point2d& pt) const {
        return x * pt.x + y * pt.y;
    }
    float norm() const {
        return sqrt(pow(x, 2) + pow(y, 2));
    }
};

struct Edge {
    Point2d sp;
    Point2d ep;
    Point2d offset;
};

union EdgeInterpret {
    float vals[6];
    Edge eg;
    EdgeInterpret(const Edge& eg): eg(eg) {}
};

typedef std::vector<Edge> Edges;
typedef std::vector<std::pair<uint8_t, uint8_t>> Vertex;

extern const std::array<Vertex, 16> table;

at::Tensor marchingSquare(at::Tensor bubbles);

void linearInterp(const Vertex& vtx, const std::array<float, 4>& vals, const Point2d offset, Edges& edges);
