#pragma once
#include <string>
#include <vector>
#include <array>
#include <variant>
#include "helpers.h"
#include "json.hpp"
#include <unordered_set>

using namespace std;
using json = nlohmann::ordered_json;

constexpr int IMAGE_WIDTH = 224;
constexpr int IMAGE_HEIGHT = 224;

class CNN {
private:
    vector<int> convLayers;
    vector<int> classifierLayers;
    optional<Tensor> X;
    vector<array<int, 3>> Y;
    optional<Tensor> X_test;
    vector<array<int, 3>> Y_test;
    vector<ConvLayerShape> convShapes;

    vector<Tensor> convWeights; // each Cout x Cin x k x k
    vector<vector<double>> convBias; // each Cout sized vector
    vector<Tensor> classifierWeights; // each neurons x n
    vector<vector<double>> classifierBias; // each neurons sized vector

    int PADDING = 1;
    int FILTER_STRIDE = 1;
    int FILTER_SIZE = 3;
    int POOL_STRIDE = 2;
    int POOL_FILTER_SIZE = 2;

    double lr = 1e-2;
    double l2 = 1e-2;
    double PREDICTION_THRESHOLD = 0.5;
    bool loadedParameters = false;

public:
    CNN(vector<int> co, vector<int> cl, vector<double> m, vector<double> s);

    vector<double> mus;
    vector<double> stds;

    void loadData(string fileName, bool test);
    void train(int iterations=500);
    void loadParameters(string fileName, bool areNested = false);
    void writeParameters(string fileName);
    tuple<vector<int>, vector<double>> predict(const Tensor &X);
    int totalParameters = 0;

private:
    Tensor loadConvWeights(const json &data, string key, ConvLayerShape shape, bool areNested);
    Tensor loadClassifierWeights(const json &data, string key, int neurons, bool areNested);
    vector<double> loadBias(const json &data, string key, bool conv);
    vector<ConvLayerShape> constructConvShapes();
    int computeFlattenedSize();
    void initializeParameters();
    double sigmoid(double z);
    Tensor flatten(const Tensor &t);
    Tensor activateMatrix(const Tensor &M, bool useSigmoid = false, bool useReLUDeriv = false);
    vector<variant<ClassifierOutput, ConvOutput>> forwardProp(const Tensor &batch, const Tensor* cachedFlattened, bool justConv=false);
    Tensor unflatten(const Tensor &m, int depth, int size);
    Tensor computeWeightGrad(const Tensor &X, const Tensor &dZ);
    void backwardProp(const vector<variant<ClassifierOutput, ConvOutput>> &outputs, const Tensor &batch, const Tensor &batchY, const Tensor* cachedFlattened, int numConvLayers = -1);
    Tensor computeConvStep(const Tensor &X, const Tensor &W, const vector<double> &b, int stride = -1);
    double ReLU(double z);
    Tensor activateMaps(const Tensor &z, bool useReLUDeriv = false);
    ConvOutput maxPool(const Tensor &X);
    tuple<Tensor, Tensor, vector<int>> makeBatch(int size=50);
    tuple<double, double, double, double, double, double, double, double> diagnose(const Tensor &batch, const Tensor &batchY, const Tensor* cached); 
    Tensor makeYBatch(const vector<array<int, 3>> &y);
    optional<tuple<Tensor, Tensor>> getFlattenedCache(int flattenedSize);
    void writeFlattenedCache(const Tensor &flattened, const Tensor &testFlattened);
    int computeTotalParameters();
};
