#pragma once
#include <vector>
#include <tuple>
#include <string>
#include <array>
#include <optional>
#include <concepts>
#include <unordered_set>
#include <iostream>
#include <cstdlib>
#include <sstream>

using namespace std;

std::string trim(const std::string& str);
array<int, 3> parseRgb(string s);
bool getCell(stringstream &ss, string &cell);

class Tensor {
public:
    vector<int> shape;
    int dimensions;
    vector<int> blockSizes;
    vector<double> d;
    int rows;
    int cols;

    Tensor(same_as<int> auto... loc);
    Tensor(vector<int> s, double initial=0.0);
    Tensor(vector<int> s, vector<double> data);

    double& operator[](same_as<int> auto... loc);
    const double& operator[](same_as<int> auto... loc) const;
    int getIdx(same_as<int> auto... loc) const;
    Tensor clone() const;

    void transpose();
    void rot180();
    vector<double> getVerticalVector(vector<int> loc, int col = 0);
    void scale(double c);
    Tensor getScaled(double c) const;
    Tensor broadcast(int times);
    vector<double> getAveragedVector() const;
    Tensor pad(int p) const;
    void flipHorizontally();
    int size() const;
    Tensor pick(const vector<int> &indexes) const;
    int depth() const;
    void swapSizeDepthDimensions();
};

Tensor::Tensor(same_as<int> auto... s) {
    int size = 1;
    for (int dim : {s...}) {
        shape.push_back(dim);
        size *= dim;
    }
    dimensions = shape.size();
    if (dimensions < 2) {
        cerr << "Tensor expected min of 2 dimensions" << endl;
        abort();
    }
    rows = shape[dimensions - 2];
    cols = shape[dimensions - 1];

    d = vector(size, 0.0);
    for (int i = 0; i < dimensions; ++i) {
        if (shape[i] < 1) {
            cerr << "dimension must be >0" << endl;
            abort();
        }

        int blockSize = 1;
        for (int j = i + 1; j < dimensions; ++j) {
            blockSize *= shape[j];
        }
        blockSizes.push_back(blockSize);
    }
}

double& Tensor::operator[](same_as<int> auto... loc) {
    int idx = getIdx(loc...);
    return d[idx];
}

const double& Tensor::operator[](same_as<int> auto... loc) const {
    int idx = getIdx(loc...);
    return d[idx];
}

int Tensor::getIdx(same_as<int> auto... loc) const {
    size_t count = sizeof...(loc);
    const int l[] = {loc...};
    if (count == 1) {
        return l[0];
    }
    if (count != dimensions) {
        cerr << "loc size does not match tensor shape" << endl;
        abort();
    }
    
    int idx = 0;
    for (int i = 0; i < count; ++i) {
        idx += l[i] * blockSizes[i];
    }
    return idx;
}

struct ConvLayerShape {
    int Win;
    int Cin;
    int Wout;
    int Cout;
};

double randomGauss(double mu, double sigma);

struct ClassifierOutput {
    Tensor z;
    Tensor a;
};

struct ConvOutput {
    Tensor z;
    Tensor a;
    // indicies kept in pooling (example, channel, x, y)
    optional<vector<tuple<int, int, int, int>>> keptIndicies;
};

double dotProduct(const vector<double> &a, const vector<double> &b);
Tensor tensorMultiplication(const Tensor &A, const Tensor &B);
Tensor tensorAddition(const Tensor &A, const Tensor &B);
Tensor tensorSubtraction(const Tensor &A, const Tensor &B);
Tensor tensorElementWise(const Tensor &A, const Tensor &B);

vector<double> scaleVector(const vector<double> vec, double c);
vector<double> subtractVectors(const vector<double> a, const vector<double> b);
