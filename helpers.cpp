#include "helpers.h"
#include <sstream>
#include <random>
#include <iostream>
#include <algorithm>
#include <span>

using namespace std;

std::string trim(const std::string& str) {
    const std::string WHITESPACE = " \t\n\r\f\v";
    
    size_t start = str.find_first_not_of(WHITESPACE);
    if (start == std::string::npos) {
        return "";
    }
    
    size_t end = str.find_last_not_of(WHITESPACE);
    
    return str.substr(start, end - start + 1);
}

array<int, 3> parseRgb(string s) {
    array<int, 3> rgb;
    sscanf(s.c_str(), "\"(%d, %d, %d)\"", &rgb[0], &rgb[1], &rgb[2]);
    return rgb;
}

bool getCell(stringstream &ss, string &cell) {
    cell.clear();
    string part;
    bool withinRgb = false;
    while (getline(ss, part, ',')) {
        // not an rgb
        if (!withinRgb && part.size() > 0 && part.at(0) != '"') {
            cell.append(part);
            return true;
        }
        withinRgb = true;

        if (cell.size() > 0) cell.append(",");
        cell.append(part);
        auto lastIdx = part.size() - 1;
        if (part.at(lastIdx) == '"') {
            break;
        }
    }
    return cell.size() > 0;
}

Tensor::Tensor(vector<int> s, double initial) : shape(s), dimensions(s.size()), rows(s[s.size() - 2]), cols(s[s.size() - 1]) {
    if (dimensions < 2) {
        cerr << "Tensor expected min of 2 dimensions" << endl;
        abort();
    }

    int size = 1;
    for (int dim : s) {
        size *= dim;
    }
    d = vector(size, initial);

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

Tensor::Tensor(vector<int> s, vector<double> data) : shape(s), dimensions(s.size()), d(data), rows(s[s.size() - 2]), cols(s[s.size() - 1]) {
    if (dimensions < 2) {
        cerr << "Tensor expected min of 2 dimensions" << endl;
        abort();
    }

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

Tensor Tensor::clone() const {
    return Tensor(shape, d);
}

void Tensor::transpose() {
    if (dimensions < 2) {
        cerr << "transpose on < 2 dimensions" << endl;
        abort();
    }

    // swap rows and cols
    vector<int> newShape = shape;
    newShape[dimensions - 1] = rows;
    newShape[dimensions - 2] = cols;

    int matSize = rows * cols;
    int totalMatricies = d.size() / matSize;

    vector<double> newD(d.size());
    for (int m = 0; m < totalMatricies; ++m) {
        int offset = m * matSize;
        for (int r = 0; r < rows; ++r) {
            for (int c = 0; c < cols; ++c) {
                newD[offset + c * rows + r] = d[offset + r * cols + c];
            }
        }
    }

    
    *this = Tensor(newShape, move(newD));
}

void Tensor::rot180() {
    if (dimensions < 2) {
        cerr << "rot180 on < 2 dimensions" << endl;
        abort();
    }
    
    int matSize = rows * cols;
    int totalMatricies = d.size() / matSize;

    vector<double> newD(d.size());
    for (int m = 0; m < totalMatricies; ++m) {
        int offset = m * matSize;
        for (int r = 0; r < rows; ++r) {
            for (int c = 0; c < cols; ++c) {
                newD[offset + r * cols + c] = d[offset + (rows - r - 1) * cols + (cols - c - 1)];
            }
        }
    }
    
    *this = Tensor(shape, move(newD));
}

vector<double> Tensor::getVerticalVector(vector<int> loc, int col) {
    int offset = 0;
    for (int dim = 0; dim < loc.size(); ++dim) {
        offset += loc[dim] * blockSizes[dim];
    }

    vector<double> vec;
    for (int r = 0; r < rows; ++r) {
        int idx = offset + cols * r + col;
        vec.push_back(d[idx]);
    }
    return vec;
}

void Tensor::scale(double c) {
    for (int i = 0; i < d.size(); i++) {
        d[i] *= c;
    }
}

Tensor Tensor::getScaled(double c) const {
    Tensor newD(shape);
    for (int i = 0; i < d.size(); i++) {
        newD.d[i] = d[i] * c;
    }
    return newD;
}

Tensor Tensor::broadcast(int m) {
    if (rows != 1 && cols != 1) {
        cerr << "broadcast expects nx1 or 1xn" << endl;
        abort();
    }

    if (rows == 1) {
        // have 1xn, want mxn
        Tensor B(m, cols);
        for (int i = 0; i < m; ++i) {
            copy(d.begin(), d.end(), B.d.begin() + cols * i);
        }
        return B;
    } else {
        Tensor transposed = clone();
        transposed.transpose();
        Tensor B = transposed.broadcast(m);
        B.transpose();
        return B;
    }
}


vector<double> Tensor::getAveragedVector() const {
    if (dimensions != 2) {
        cerr << "getAveragedVector() on non matrix (expects 2d)" << endl;
        abort();
    }
    vector<double> vec;

    for (int r = 0; r < rows; ++r) {
        double sum = 0;
        for (int c = 0; c < cols; ++c) {
            sum += (*this)[r, c];
        }
        vec.push_back(sum / cols);
    }
    return vec;
}

Tensor Tensor::pad(int p) const {
    if (rows != cols) {
        cerr << "pad only on square matricies" << endl;
        abort();
    }

    int width = cols + 2 * p;
    vector<int> newShape = shape;
    newShape[dimensions - 2] = width;
    newShape[dimensions - 1] = width;

    Tensor o(newShape, 0.0);
    int matSize = rows * cols;
    int paddedMatSize = width * width;
    int totalMatricies = d.size() / matSize;

    for (int m = 0; m < totalMatricies; ++m) {
        int offset = m * matSize;
        int paddedOffset = m * paddedMatSize;

        for (int r = 0; r < rows; ++r) {
            for (int c = 0; c < cols; ++c) {
                o.d[paddedOffset + (r + p) * width + c + p] = d[offset + r * cols + c];
            }
        }
    }
    return o;
}

void Tensor::flipHorizontally() {
    int matSize = rows * cols;
    int totalMatricies = d.size() / matSize;

    for (int m = 0; m < totalMatricies; ++m) {
        for (int r = 0; r < rows; ++r) {
            int offset = m * matSize + r * cols;
            reverse(d.begin() + offset, d.begin() + offset + cols);
        }
    }
}

int Tensor::size() const {
    return d.size() / blockSizes[0];
}

Tensor Tensor::pick(const vector<int> &indexes) const {
    int total = indexes.size();
    int blockSize = blockSizes[0];
    int size = total * blockSize;
    vector<double> newD(size, 0.0);
    
    for (int i = 0; i < indexes.size(); ++i) {
        int idx = indexes[i];
        copy(d.begin() + idx * blockSize, d.begin() + idx * blockSize + blockSize, newD.begin() + i * blockSize);
    }
    
    vector<int> newShape = shape;
    newShape[0] = total;
    return Tensor(newShape, move(newD));
}

int Tensor::depth() const {
    return shape[max(dimensions - 3, 0)];
}

void Tensor::swapSizeDepthDimensions() {
    vector<double> newD(d.size(), 0.0);

    int sBlockSize = blockSizes[0];
    int dhBlockSize = blockSizes[1];

    for (int dh = 0; dh < depth(); ++dh) {
        for (int s = 0; s < size(); ++s) {
            copy(d.begin() + s * sBlockSize + dh * dhBlockSize, d.begin() + s * sBlockSize + dh * dhBlockSize + dhBlockSize, newD.begin() + (dh * size() + s) * dhBlockSize);
        }
    }

    swap(shape[0], shape[1]);
    *this = Tensor(shape, move(newD));
}

double randomGauss(double mu, double sigma) {
    static random_device rd;
    static mt19937 gen(rd());

    normal_distribution<double> dist(mu, sigma);

    return dist(gen);
}

double dotProduct(const vector<double> &a, const vector<double> &b) {
    double sum = 0;
    for (int i = 0; i < a.size(); ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}

Tensor tensorMultiplication(const Tensor &A, const Tensor &B) {
    vector<int> shape = A.dimensions > 2 ? vector<int>(A.shape.begin(), A.shape.end() - 2) : vector<int>(0);
    shape.push_back(A.rows);
    shape.push_back(B.cols);
    Tensor prod(shape);

    int aMatSize = A.rows * A.cols;
    int bMatSize = B.rows * B.cols;
    int prodMatSize = A.rows * B.cols;
    int totalMatricies = A.d.size() / aMatSize;

    int vecSize = A.cols;

    for (int m = 0; m < totalMatricies; ++m) {
        #pragma omp parallel for
        for (int r = 0; r < A.rows; ++r) {
            for (int n = 0; n < vecSize; ++n) {
                for (int c = 0; c < B.cols; ++c) {
                    prod.d[m * prodMatSize + r * B.cols + c] += A.d[m * aMatSize + r * vecSize + n] * B.d[m * bMatSize + n * B.cols + c];
                }
            }
        }
    }

    return prod;
}

Tensor tensorAddition(const Tensor &A, const Tensor &B) {
    Tensor sum(A.shape);
    for (int i = 0; i < A.d.size(); ++i) {
        sum.d[i] = A.d[i] + B.d[i];
    }
    return sum;
}

Tensor tensorSubtraction(const Tensor &A, const Tensor &B) {
    Tensor diff(A.shape);
    for (int i = 0; i < A.d.size(); ++i) {
        diff.d[i] = A.d[i] - B.d[i];
    }
    return diff;
}

Tensor tensorElementWise(const Tensor &A, const Tensor &B) {
    Tensor prod(A.shape);
    for (int i = 0; i < A.d.size(); ++i) {
        prod.d[i] = A.d[i] * B.d[i];
    }
    return prod;
}

vector<double> scaleVector(const vector<double> vec, double c) {
    vector<double> o;
    for (int i = 0; i < vec.size(); ++i) {
        o.push_back(vec[i] * c);
    }
    return o;
}

vector<double> subtractVectors(const vector<double> a, const vector<double> b) {
    vector<double> o;
    for (int i = 0; i < a.size(); ++i) {
        o.push_back(a[i] - b[i]);
    }
    return o;
}
