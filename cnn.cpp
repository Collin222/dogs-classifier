#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include "httplib.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <cassert>
#include <random>
#include "cnn.h"
#include "helpers.h"
#include <cmath>
#include <filesystem>
#include <algorithm>
#include <print>
#include <chrono>

CNN::CNN(vector<int> co, vector<int> cl, vector<double> m, vector<double> s) : convLayers(co), classifierLayers(cl), mus(m), stds(s) {
    constructConvShapes();
    totalParameters = computeTotalParameters();
}

void CNN::loadData(string fileName, bool test) {
    ifstream file(fileName);

    assert(file.is_open() && "Failed to open data file");

    string line;

    vector<string> lines;
    getline(file, line);
    while (getline(file, line)) lines.push_back(line);

    int numLines = lines.size();
    int NUM_VERSIONS = test ? 1 : 4;
    double brightnessDec = 0.8;
    double brightnessInc = 1.2;

    Tensor data(numLines * NUM_VERSIONS, 3, IMAGE_HEIGHT, IMAGE_WIDTH);

    for (int l = 0; l < numLines; ++l) {
        stringstream ss(lines[l]);
        string cell;

        array<int, 3> label;

        int i = 0;
        int row = 0;
        int col = 0;
        while (getCell(ss, cell)) {
            if (i == 3) {
                // skip filename column
                ++i;
                continue;
            }

            if (i < 3) {
                label[i] = stoi(cell);
            } else {
                auto [rRaw, gRaw, bRaw] = parseRgb(cell);
                
                if (col >= IMAGE_WIDTH) {
                    row++;
                    col = 0;
                }

                double r = (rRaw / 255.0 - mus[0]) / stds[0];
                double g = (gRaw / 255.0 - mus[1]) / stds[1];
                double b = (bRaw / 255.0 - mus[2]) / stds[2];

                data.d[data.getIdx(NUM_VERSIONS * l, 0, row, col)] = r;
                data.d[data.getIdx(NUM_VERSIONS * l, 1, row, col)] = g;
                data.d[data.getIdx(NUM_VERSIONS * l, 2, row, col)] = b;
                
                if (NUM_VERSIONS > 1) {
                    // flipped horizontally
                    data.d[data.getIdx(NUM_VERSIONS * l + 1, 0, row, IMAGE_WIDTH - col - 1)] = r;
                    data.d[data.getIdx(NUM_VERSIONS * l + 1, 1, row, IMAGE_WIDTH - col - 1)] = g;
                    data.d[data.getIdx(NUM_VERSIONS * l + 1, 2, row, IMAGE_WIDTH - col - 1)] = b;
                    
                    // brightness decrease
                    data.d[data.getIdx(NUM_VERSIONS * l + 2, 0, row, col)] = (min(rRaw * brightnessDec, 255.0) / 255.0 - mus[0]) / stds[0];
                    data.d[data.getIdx(NUM_VERSIONS * l + 2, 1, row, col)] = (min(gRaw * brightnessDec, 255.0) / 255.0 - mus[1]) / stds[1];
                    data.d[data.getIdx(NUM_VERSIONS * l + 2, 2, row, col)] = (min(bRaw * brightnessDec, 255.0) / 255.0 - mus[2]) / stds[2];

                    // brightness increase
                    data.d[data.getIdx(NUM_VERSIONS * l + 3, 0, row, col)] = (min(rRaw * brightnessInc, 255.0) / 255.0 - mus[0]) / stds[0];
                    data.d[data.getIdx(NUM_VERSIONS * l + 3, 1, row, col)] = (min(gRaw * brightnessInc, 255.0) / 255.0 - mus[1]) / stds[1];
                    data.d[data.getIdx(NUM_VERSIONS * l + 3, 2, row, col)] = (min(bRaw * brightnessInc, 255.0) / 255.0 - mus[2]) / stds[2];
                }

                ++col;
            }

            ++i;
        }
        
        // label for normal and horizontally flipped
        if (test) {
            Y_test.push_back(label);
        } else {
            for (int i = 0; i < NUM_VERSIONS; ++i) Y.push_back(label);
        }
    }

    if (test) {
        X_test = move(data);
    } else {
        X = move(data);
    }

    file.close();

    string named = test ? "test" : "train";
    int total = test ? X_test->size() : X->size();
    cout << "Parsed " << total << " " << named << " examples from " << fileName << endl;
}

Tensor CNN::loadConvWeights(const json &data, string key, ConvLayerShape shape, bool areNested) {
    if (areNested) {
        auto raw = data[key].get<vector<vector<vector<vector<double>>>>>();
        int Cout = raw.size();
        int Cin = raw[0].size();
        int k = raw[0][0].size();
        
        Tensor t(Cout, Cin, k, k);
        for (int i = 0; i < Cout; ++i) {
            for (int j = 0; j < Cin; ++j) {
                for (int a = 0; a < k; ++a) {
                    for (int b = 0; b < k; ++b) {
                        t[i, j, a, b] = raw[i][j][a][b];
                    }
                }
            }
        }
        convWeights.push_back(t);
        return t;
    } else {
        auto raw = data[key].get<vector<double>>();
        Tensor t({shape.Cout, shape.Cin, FILTER_SIZE, FILTER_SIZE}, raw);
        convWeights.push_back(t);
        return t;
    }
}

Tensor CNN::loadClassifierWeights(const json &data, string key, int neurons, bool areNested) {
    if (areNested) {
        auto raw = data[key].get<vector<vector<double>>>();
        int rows = raw.size();
        int cols = raw[0].size();
        
        Tensor m(rows, cols);
        for (int r = 0; r < rows; ++r) {
            for (int c = 0; c < cols; ++c) {
                m[r, c] = raw[r][c];
            }
        }
        
        classifierWeights.push_back(m);
        return m;
    } else {
        auto raw = data[key].get<vector<double>>();
        int n = raw.size() / neurons;
        Tensor m({neurons, n}, raw);
        classifierWeights.push_back(m);
        return m;
    }
}

vector<double> CNN::loadBias(const json &data, string key, bool conv) {
    vector<double> b = data[key].get<vector<double>>();
    if (conv) {
        convBias.push_back(b);
    } else {
        classifierBias.push_back(b);
    }
    return b;
}

void CNN::loadParameters(string fileName, bool areNested) {
    ifstream file(fileName);
    assert(file.is_open() && "Failed to open parameters file");

    cout << "Loading parameters file " << fileName << endl;
    json j = json::parse(file);
    cout << "Loaded parameters file " << fileName << endl;

    vector<string> convKeys = {"features.0", "features.3", "features.6", "features.8", "features.11", "features.13", "features.16", "features.18"};
    int k = 0;
    for (int l = 0; l < convLayers.size(); ++l) {
        if (convLayers[l] == -1) {
            convWeights.push_back(Tensor(1, 1));
            convBias.push_back({});
            continue;
        }
        loadConvWeights(j, convKeys[k] + ".weight", convShapes[l], areNested);
        loadBias(j, convKeys[k] + ".bias", true);
        ++k;
    }

    if (!areNested) {
        for (int l = 0; l < classifierLayers.size(); ++l) {
            string prefix = "classifier." + to_string(l) + ".";
            loadClassifierWeights(j, prefix + "weight", classifierLayers[l], false);
            loadBias(j, prefix + "bias", false);
        }
    }

    loadedParameters = true;

    cout << "Done loading parameters" << endl;
}

vector<ConvLayerShape> CNN::constructConvShapes() {
    int numConvLayers = convLayers.size();

    vector<ConvLayerShape> shapes;

    for (int l = 0; l < numConvLayers; ++l) {
        int filters = convLayers[l];
        bool isPool = filters == -1;

        int p = isPool ? 0 : PADDING;
        int s = isPool ? POOL_STRIDE : FILTER_STRIDE;
        int kernelSize = isPool ? POOL_FILTER_SIZE : FILTER_SIZE;

        ConvLayerShape prev = l == 0 ? ConvLayerShape {0, 0, IMAGE_WIDTH, 3} : shapes[l - 1];

        int Cout = isPool ? prev.Cout : filters;

        int paddedSize = prev.Wout + 2 * p;
        int Wout = ((paddedSize - kernelSize) / s) + 1;
        ConvLayerShape shape = {prev.Wout, prev.Cout, Wout, Cout};
        shapes.push_back(shape);
    }

    convShapes = shapes;
    return shapes;
}

int CNN::computeFlattenedSize() {
    auto last = convShapes[convShapes.size() - 1];
    return last.Cout; // global average pooling per channel
}

void CNN::initializeParameters() {
    if (!loadedParameters) {
        convWeights.clear();
        convBias.clear();
        for (int l = 0; l < convLayers.size(); ++l) {
            auto shape = convShapes[l];
            int fanIn = shape.Cin * FILTER_SIZE * FILTER_SIZE;
            double sigma = sqrt(2.0 / fanIn);
            
            convWeights.push_back(Tensor(shape.Cout, shape.Cin, FILTER_SIZE, FILTER_SIZE));

            vector<double> bias(shape.Cout, 0.0);
            convBias.push_back(bias);
            
            bool isPool = convLayers[l] == -1;
            if (isPool) continue;

            for (int co = 0; co < shape.Cout; ++co) {
                for (int ci = 0; ci < shape.Cin; ++ci) {
                    for (int r = 0; r < FILTER_SIZE; ++r) {
                        for (int c = 0; c < FILTER_SIZE; ++c) {
                            convWeights[l][co, ci, r, c] = randomGauss(0.0, sigma);
                        }
                    }
                }
            }

        }
    }

    classifierWeights.clear();
    classifierBias.clear();
    for (int l = 0; l < classifierLayers.size(); ++l) {
        // size of input vector to layer
        int n = l == 0 ? computeFlattenedSize() : classifierLayers[l - 1];
        
        int neurons = classifierLayers[l];
        double sigma = sqrt(2.0 / n);

        classifierWeights.push_back(Tensor(neurons, n));

        for (int i = 0; i < neurons; ++i) {
            for (int j = 0; j < n; ++j) {
                classifierWeights[l][i, j] = randomGauss(0.0, sigma);
            }
        }

        vector<double> bias(neurons, 0.0);
        classifierBias.push_back(bias);
    }
}

Tensor CNN::flatten(const Tensor &t) {
    // flattens with global average pooling
    ConvLayerShape before = convShapes[convShapes.size() - 1];
    int k = before.Wout;
    int n = before.Cout;
    int m = t.size();

    Tensor flat(m, n); // m x n
    double* fd = flat.d.data();
    const double* td = t.d.data();

    #pragma omp parallel for
    for (int i = 0; i < m; ++i) {
        for (int ch = 0; ch < n; ++ch) {
            double* dc = fd + i * n + ch;
            const double* tr = td + (i * n + ch) * k * k;

            for (int p = 0; p < k * k; ++p) *dc += tr[p];
            *dc /= static_cast<double>(k * k);
        }
    }


    flat.transpose(); // n x m
    return flat;
}

double CNN::sigmoid(double z) {
    return 1 / (1 + exp(-1 * z));
}

Tensor CNN::activateMatrix(const Tensor &M, bool useSigmoid, bool useReLUDeriv) {
    Tensor A(M.rows, M.cols);
    for (int i = 0; i < A.d.size(); ++i) {
        double z = M[i];
        double a = useSigmoid ? sigmoid(z) : ReLU(z);
        if (useReLUDeriv && a > 0) a = 1;
        A[i] = a;
    }
    return A;
}

vector<variant<ClassifierOutput, ConvOutput>> CNN::forwardProp(const Tensor &batch, const Tensor* cachedFlattened, bool justConv) {
    vector<variant<ClassifierOutput, ConvOutput>> outputs;
    bool fromCached = cachedFlattened != nullptr;

    if (cachedFlattened == nullptr) {
        for (int l = 0; l < convLayers.size(); ++l) {
            const Tensor &X = l == 0 ? batch : get<ConvOutput>(outputs[l - 1]).a;

            // is pool layer
            if (convLayers[l] == -1) {
                ConvOutput o = maxPool(X);
                outputs.push_back(o);
                continue;
            }
            
            Tensor Xpadded = X.pad(PADDING);
            
            const Tensor &W = convWeights[l];
            const vector<double> &b = convBias[l];
            
            Tensor z = computeConvStep(Xpadded, W, b);
            Tensor a = activateMaps(z);
            
            ConvOutput o = {z, a};
            outputs.push_back(o);
        }

        if (justConv) return outputs;
    }

    optional<Tensor> flattened;
    if (!fromCached) {
        flattened = flatten(get<ConvOutput>(outputs.back()).a);
        cachedFlattened = &*flattened;
    }
    
    for (int l = 0; l < classifierLayers.size(); ++l) {
        int totalLayer = fromCached ? l : l + convLayers.size();
        const Tensor *X = l == 0 ? cachedFlattened : &get<ClassifierOutput>(outputs[totalLayer - 1]).a;
        if (X == nullptr) {
            cerr << "Failed to get layer inputs" << endl;
            abort();
        }

        const Tensor &W = classifierWeights[l];
        int bSize = classifierBias[l].size();
        Tensor b({1, bSize}, classifierBias[l]); // 1xn
        Tensor B = b.broadcast(X->cols); // mxn
        B.transpose(); // nxm

        Tensor z = tensorAddition(tensorMultiplication(W, *X), B);
        Tensor a = activateMatrix(z, l == classifierLayers.size() - 1);

        ClassifierOutput o = {z, a};
        outputs.push_back(o);
    }

    return outputs;
}

Tensor CNN::unflatten(const Tensor &M, int depth, int size) {
    // M is n x m
    int channelSize = size * size;

    Tensor output(M.cols, depth, size, size);
    double* od = output.d.data();
    double inv = 1.0 / (size * size);
    for (int i = 0; i < M.cols; ++i) {
        for (int ch = 0; ch < M.rows; ++ch) {
            double g = M[ch, i] * inv;
            double* cd = od + (i * depth + ch) * size * size;
            for (int p = 0; p < size * size; ++p) cd[p] = g;
        }
    }
    return output;
}

Tensor CNN::computeWeightGrad(const Tensor &X, const Tensor &dZ) {
    int k = FILTER_SIZE;
    int s = 1;

    Tensor dW(dZ.depth(), X.depth(), k, k);
    for (int co = 0; co < dZ.depth(); ++co) {
        for (int ci = 0; ci < X.depth(); ++ci) {
            for (int a = 0; a < k; ++a) {
                for (int b = 0; b < k; ++b) {
                    double total = 0.0;
                    for (int i = 0; i < X.size(); ++i) {
                        for (int r = 0; r < dZ.rows; ++r) {
                            for (int c = 0; c < dZ.cols; ++c) {
                                total += X[i, ci, r * s + a, c * s + b] * dZ[i, co, r, c];
                            }
                        }
                    }
                    dW[co, ci, a, b] = total / X.size() * lr;
                }
            }
        }
    }
    return dW;
}

void CNN::backwardProp(const vector<variant<ClassifierOutput, ConvOutput>> &outputs, const Tensor &batch, const Tensor &batchY, const Tensor* cachedFlattened, int numConvLayers) {
    if (numConvLayers == -1) numConvLayers = convLayers.size();
    bool justClassifier = numConvLayers < 1;

    // classes x m
    Tensor predictionOutputs = get<ClassifierOutput>(outputs[outputs.size() - 1]).a;
    Tensor lastDelta = tensorSubtraction(predictionOutputs, batchY);

    // classifier layers
    // pass through gradients
    vector<Tensor> classifierDeltas = {lastDelta};
    for (int l = classifierLayers.size() - 1; l >= 0; --l) {
        int totalLayer = justClassifier ? l : l + convLayers.size();
        bool isOutput = l == classifierLayers.size() - 1;
        const Tensor &Z = get<ClassifierOutput>(outputs[totalLayer]).z;
        const Tensor &X = l == 0 ? (justClassifier && cachedFlattened != nullptr ? *cachedFlattened : flatten(get<ConvOutput>(outputs[totalLayer - 1]).a)) : get<ClassifierOutput>(outputs[totalLayer - 1]).a;
        const Tensor &dY = classifierDeltas[classifierDeltas.size() - 1];
        Tensor dZ = isOutput ? dY : tensorElementWise(activateMatrix(Z, false, true), dY);

        Tensor XT = X.clone();
        XT.transpose();
        Tensor dW = tensorMultiplication(dZ, XT);
        dW.scale(lr / batchY.cols);

        vector<double> db = scaleVector(dZ.getAveragedVector(), lr);

        const Tensor &W = classifierWeights[l];
        Tensor WT = W.clone();
        WT.transpose();
        Tensor dX = tensorMultiplication(WT, dZ);
        classifierDeltas.push_back(dX);

        classifierWeights[l] = tensorSubtraction(W.getScaled(1 - lr * l2), dW);
        classifierBias[l] = subtractVectors(classifierBias[l], db);
    }

    if (!justClassifier) {
        ConvLayerShape lastConvShape = convShapes[convLayers.size() - 1];
        Tensor unflattened = unflatten(classifierDeltas[classifierDeltas.size() - 1], lastConvShape.Cout, lastConvShape.Wout);
        
        // convolutional layers
        // pass through gradients
        vector<Tensor> convDeltas = {unflattened};
        for (int l = convLayers.size() - 1; l >= convLayers.size() - numConvLayers; --l) {
            const Tensor &Z = get<ConvOutput>(outputs[l]).z;
            const Tensor &X = l == 0 ? batch : get<ConvOutput>(outputs[l - 1]).a;
            const Tensor &dY = convDeltas[convDeltas.size() - 1];
            cout << "Updating conv layer " << l << endl;
            
            // is pool layer
            if (convLayers[l] == -1) {
                const ConvOutput &output = get<ConvOutput>(outputs[l]);
                Tensor dX(X.size(), X.depth(), X.rows, X.cols);
                int chSize = X.rows * X.cols;
                int exSize = X.depth() * chSize;
                for (int i = 0; i < dY.size(); ++i) {
                    for (int ch = 0; ch < dY.depth(); ++ch) {
                        for (int x = 0; x < dY.rows; ++x) {
                            for (int y = 0; y < dY.cols; ++y) {
                                int keptIdxLookup = i * (dY.depth() * dY.rows * dY.cols) + ch * (dY.rows * dY.cols) + x * (dY.cols) + y;
                                auto [example, channel, a, b] = (*output.keptIndicies)[keptIdxLookup];
                                dX[example, channel, a, b] = dY[i, ch, x, y];
                            }
                        }
                    }
                }

                convDeltas.push_back(dX);
                continue;
            }

            Tensor dZ = tensorElementWise(activateMaps(Z, true), dY);

            const Tensor &W = convWeights[l];
            Tensor Xpadded = X.pad(PADDING);
            Tensor dW = computeWeightGrad(Xpadded, dZ);

            vector<double> sums(W.size(), 0.0);
            for (int i = 0; i < dZ.size(); ++i) {
                for (int ch = 0; ch < dZ.depth(); ++ch) {
                    for (int r = 0; r < dZ.rows; ++r) {
                        for (int c = 0; c < dZ.cols; ++c) sums[ch] += dZ[i, ch, r, c];
                    }
                }
            }
            vector<double> db = scaleVector(sums, lr / batchY.cols);

            Tensor modifiedW = W.clone();
            modifiedW.swapSizeDepthDimensions();
            modifiedW.rot180();
            Tensor dZPadded = dZ.pad(W.rows - 1 - PADDING);

            if (l > 0) {
                Tensor dX = computeConvStep(dZPadded, modifiedW, vector<double>(modifiedW.size(), 0.0), 1);
                convDeltas.push_back(dX);
            }

            convWeights[l] = tensorSubtraction(W.getScaled(1 - lr * l2), dW);
            convBias[l] = subtractVectors(convBias[l], db);
        }
    }
}

Tensor CNN::computeConvStep(const Tensor &X, const Tensor &W, const vector<double> &bias, int stride) {
    int Win = X.cols;
    int k = W.rows; // filter size
    int s = stride == -1 ? FILTER_STRIDE : stride;
    int outSize = (Win - k) / s + 1;
    int Cout = W.size();
    int Cin = X.depth();
    int m = X.size();
    Tensor z(m, Cout, outSize, outSize);

    const double* xd = X.d.data();
    const double* wd = W.d.data();
    double* zd = z.d.data();

    #pragma omp parallel for
    for (int w = 0; w < Cout; ++w) {
        for (int i = 0; i < m; ++i) {
            double* zi = zd + ((size_t)i * Cout + w) * outSize * outSize; // outSize * outSize
            for (int p = 0; p < outSize * outSize; ++p) zi[p] = bias[w];

            for (int ch = 0; ch < Cin; ++ch) {
                const double* wch = wd + ((size_t)w * Cin + ch) * k * k; // k x k
                const double* xch = xd + ((size_t)i * Cin + ch) * Win * Win; // Win x Win

                for (int a = 0; a < k; ++a) {
                    for (int b = 0; b < k; ++b) {
                        double wab = wch[a * k + b];

                        for (int r = 0; r < outSize; ++r) {
                            double* zr = zi + r * outSize; // outSize
                            const double* xr = xch + (r * s * Win) + (a * Win) + b; // into x row, with col b pre-applied
                            for (int c = 0; c < outSize; ++c) {
                                zr[c] += wab * xr[c * s]; // index into correct col
                            }
                        }
                    }
                }
            }
        }
    }

    return z;
}

double CNN::ReLU(double z) {
    return z > 0 ? z : 0;
}

Tensor CNN::activateMaps(const Tensor &z, bool useReLUDeriv) {
    Tensor a(z.size(), z.depth(), z.rows, z.cols);
    for (int i = 0; i < z.d.size(); ++i) {
        double activated = ReLU(z[i]);
        if (useReLUDeriv && activated > 0) activated = 1;
        a[i] = activated;
    }
    return a;
}

ConvOutput CNN::maxPool(const Tensor &X) {
    int Win = X.cols;
    int outSize = (Win - POOL_FILTER_SIZE) / POOL_STRIDE + 1;
    Tensor z(X.size(), X.depth(), outSize, outSize);

    vector<tuple<int, int, int, int>> keptIndicies(X.size() * X.depth() * outSize * outSize);

    #pragma omp parallel for
    for (int i = 0; i < X.size(); ++i) {
        for (int r = 0; r < outSize; ++r) {
            int fromRow = r * POOL_STRIDE;
            for (int c = 0; c < outSize; ++c) {
                int fromCol = c * POOL_STRIDE;
                for (int ch = 0; ch < X.depth(); ++ch) {
                    tuple<int, int, int, int> maxKey;
                    optional<double> maxX;
                    for (int a = 0; a < POOL_FILTER_SIZE; ++a) {
                        for (int b = 0; b < POOL_FILTER_SIZE; ++b) {
                            double x = X[i, ch, fromRow + a, fromCol + b];
                            if (!maxX.has_value() || x > maxX) {
                                maxX = x;
                                maxKey = make_tuple(i, ch, fromRow + a, fromCol + b);
                            }
                        }
                    }                    
                    if (!maxX.has_value()) {
                        cerr << "max pool failed to get max" << endl;
                        abort();
                    }
                    z[i, ch, r, c] = *maxX;
                    keptIndicies[i * (X.depth() * outSize * outSize) + ch * (outSize * outSize) + r * outSize + c] = maxKey;
                }
            }
        }
    }

    ConvOutput output = {z, z, keptIndicies};
    return output;
}

tuple<Tensor, Tensor, vector<int>> CNN::makeBatch(int size) {
    Tensor batchY(3, size);
    unordered_set<int> used;
    static random_device rd;
    static mt19937 gen(rd());
    uniform_int_distribution<int> dist(0, X->size() - 1);
    vector<int> indicies;
    while (used.size() < size) {
        int idx = dist(gen);
        bool has = used.count(idx);
        if (has) continue;
        used.insert(idx);
        indicies.push_back(idx);

        int exIdx = used.size() - 1;
        batchY[0, exIdx] = static_cast<double>(Y[idx][0]);
        batchY[1, exIdx] = static_cast<double>(Y[idx][1]);
        batchY[2, exIdx] = static_cast<double>(Y[idx][2]);
    }
    Tensor batch = X->pick(indicies);
    return make_tuple(batch, batchY, indicies);
}

tuple<vector<int>, vector<double>> CNN::predict(const Tensor &X) {
    Tensor batch({1, X.depth(), X.rows, X.cols}, X.d);

    auto outputs = forwardProp(batch, nullptr);
    vector<double> scores = get<ClassifierOutput>(outputs[outputs.size() - 1]).a.getVerticalVector({});
    vector<int> predictions;
    for (int i = 0; i < scores.size(); ++i) predictions.push_back(scores[i] > PREDICTION_THRESHOLD ? 1 : 0);
    return make_tuple(predictions, scores);
}

tuple<double, double, double, double, double, double, double, double> CNN::diagnose(const Tensor &batch, const Tensor &batchY, const Tensor* cached) {
    auto outputs = forwardProp(batch, cached);
    auto output = get<ClassifierOutput>(outputs[outputs.size() - 1]).a; // n x m
    int correct = 0;
    double loss = 0.0;
    double lossCharlotte = 0.0;
    double lossIvy = 0.0;
    double lossLeah = 0.0;
    double correctCharlotte = 0.0;
    double correctIvy = 0.0;
    double correctLeah = 0.0;

    for (int i = 0; i < output.cols; ++i) {
        const vector<double> &scores = output.getVerticalVector({}, i);
        bool isCorrect = true;
        for (int j = 0; j < scores.size(); j++) {
            int trueY = static_cast<int>(batchY[j, i]);
            double addLoss = trueY == 1 ? log(scores[j] + 1e-16) : log(1 - scores[j] + 1e-16);
            loss += addLoss;
            int yhat = scores[j] > PREDICTION_THRESHOLD ? 1 : 0;
            bool matches = trueY == yhat;
            if (!matches) isCorrect = false;

            if (j == 0) {
                lossCharlotte += addLoss;
                if (matches) correctCharlotte++;
            } else if (j == 1) {
                lossIvy += addLoss;
                if (matches) correctIvy++;
            } else if (j == 2) {
                lossLeah += addLoss;
                if (matches) correctLeah++;
            }
        }
        if (isCorrect) correct++;
    }
    int batchSize = batch.size();
    loss = -loss / batchSize;
    double accuracy = static_cast<double>(correct) / batchSize;
    return make_tuple(loss, accuracy, lossCharlotte, lossIvy, lossLeah, static_cast<double>(correctCharlotte) / batchSize, static_cast<double>(correctIvy) / batchSize, static_cast<double>(correctLeah) / batchSize);
}

Tensor CNN::makeYBatch(const vector<array<int, 3>> &y) {
    int m = y.size();
    Tensor o(3, m);
    for (int i = 0; i < y.size(); ++i) {
        o[0, i] = static_cast<double>(y[i][0]);
        o[1, i] = static_cast<double>(y[i][1]);
        o[2, i] = static_cast<double>(y[i][2]);
    }
    return o;
}

optional<tuple<Tensor, Tensor>> CNN::getFlattenedCache(int flattenedSize) {
    string cacheFile = "flattenedCaches.json";

    if (!std::filesystem::exists(cacheFile)) {
        return nullopt;
    }

    ifstream f(cacheFile);
    if (!f.is_open()) {
        cerr << "Failed to open cache file" << endl;
        return nullopt;
    }

    json j = json::parse(f);
    auto fRaw = j["flattened"].get<vector<double>>();
    auto tfRaw = j["testFlattened"].get<vector<double>>();

    Tensor flattened({X->size(), flattenedSize}, move(fRaw));
    Tensor testFlattened({flattenedSize, X_test->size()}, move(tfRaw));

    return make_tuple(flattened, testFlattened);
}

void CNN::writeFlattenedCache(const Tensor &flattened, const Tensor &testFlattened) {
    string cacheFile = "flattenedCaches.json";
    ofstream out(cacheFile);

    json j;
    j["flattened"] = flattened.d;
    j["testFlattened"] = testFlattened.d;

    out << j.dump(2);
    cout << "Wrote flattened to cache" << endl;
}

int CNN::computeTotalParameters() {
    int total = 0;
    for (int l = 0; l < convLayers.size(); ++l) {
        if (convLayers[l] == -1) continue;
        ConvLayerShape shape = convShapes[l];
        total += shape.Cout * FILTER_SIZE * FILTER_SIZE + shape.Cout;
    }
    for (int l = 0; l < classifierLayers.size(); ++l) {
        int neurons = classifierLayers[l];
        int n = l == 0 ? computeFlattenedSize() : classifierLayers[l - 1];
        total += neurons * n + neurons;
    }
    return total;
}

void CNN::train(int iterations) { 
    initializeParameters();
    cout << "Initialized CNN parameters" << endl;

    int flattenedSize = computeFlattenedSize();
    cout << "Flattened size : " << flattenedSize << endl;

    cout << "Total parameters : " << totalParameters << endl;

    auto caches = getFlattenedCache(flattenedSize);
    
    // compute and cache conv outputs for training set, and just train classifier
    Tensor flattened(X->size(), flattenedSize); // m x n
    Tensor testFlattened(X_test->size(), flattenedSize); // m x n

    int BATCH_SIZE = 10;
    int IMAGE_SIZE = X->blockSizes[0];
    bool validCache = false;
    if (caches) {
        auto& [f, tf] = *caches;
        if (f.size() != flattened.size() || f.depth() != flattened.depth() || tf.size() != testFlattened.depth() || tf.depth() != testFlattened.size()) validCache = false;
        else validCache = true;

        if (validCache) {
            flattened = move(f); // m x n
            testFlattened = move(tf); // becomes n x m
            cout << "Loaded conv cache" << endl;
        } else {
            cout << "Conv cache is invalid, recomputing" << endl;
        }
    }
    
    if (!validCache) {
        int totalBatches = X->size() / BATCH_SIZE + X_test->size() / BATCH_SIZE;
        cout << "Total batches " << totalBatches << endl;
        for (int i = 0; i < X->size(); i += BATCH_SIZE) {
            bool isEnd = i + BATCH_SIZE > X->size();
            int m = isEnd ? X->size() - i : BATCH_SIZE;
            Tensor batch(m, 3, IMAGE_HEIGHT, IMAGE_WIDTH);
            copy(X->d.begin() + i * IMAGE_SIZE, isEnd ? X->d.end() : X->d.begin() + (i + BATCH_SIZE) * IMAGE_SIZE, batch.d.begin());
            auto outputs = forwardProp(batch, nullptr, true);
            ConvOutput &last = get<ConvOutput>(outputs[outputs.size() - 1]);
            Tensor flat = flatten(last.a); // n x m
            flat.transpose(); // m x n
            copy(flat.d.begin(), flat.d.end(), flattened.d.begin() + i * flattenedSize);
            int batchNum = i / BATCH_SIZE;
            if (batchNum % 10 == 0) cout << "Done conv compute for batch " << batchNum << endl;
        }
        cout << "Computed and cached convolutions for training set" << endl;
    
        for (int i = 0; i < X_test->size(); i += BATCH_SIZE) {
            bool isEnd = i + BATCH_SIZE > X_test->size();
            int m = isEnd ? X_test->size() - i : BATCH_SIZE;
            Tensor batch(m, 3, IMAGE_HEIGHT, IMAGE_WIDTH);
            copy(X_test->d.begin() + i * IMAGE_SIZE, isEnd ? X_test->d.end() : X_test->d.begin() + (i + BATCH_SIZE) * IMAGE_SIZE, batch.d.begin());
            auto outputs = forwardProp(batch, nullptr, true);
            ConvOutput &last = get<ConvOutput>(outputs[outputs.size() - 1]);
            Tensor flat = flatten(last.a); // n x m
            flat.transpose(); // m x n
            copy(flat.d.begin(), flat.d.end(), testFlattened.d.begin() + i * flattenedSize);
            int batchNum = i / BATCH_SIZE + (X->size() / BATCH_SIZE);
            if (batchNum % 10 == 0) cout << "Done flat compute for batch " << batchNum << endl;
        }
        testFlattened.transpose(); // n x m
        cout << "Computed and cached convolutions for test set" << endl;

        writeFlattenedCache(flattened, testFlattened);
    }

    Tensor yBatch = makeYBatch(Y);
    Tensor yTestBatch = makeYBatch(Y_test);
    
    auto [diagnoseX, diagnoseY, diagnoseIndexes] = makeBatch(10);

    Tensor cachedDiagnose = flattened.pick(diagnoseIndexes); // m x n
    cachedDiagnose.transpose(); // n x m

    Tensor fullFlattened = flattened.clone(); // m x n
    fullFlattened.transpose(); // n x m

    for (int i = 0; i < iterations; ++i) {
        cout << "Iteration " << i << endl;

        if ((i % 50 == 0 && i > 0) || i == iterations - 1) {
            auto [trainLoss, trainAccuracy, trLossCh, trLossIv, trLossLe, trAccCh, trAccIv, trAccLe] = i == iterations - 1 ? diagnose(*X, yBatch, &fullFlattened) : diagnose(diagnoseX, diagnoseY, &cachedDiagnose);
            cout << "Train Loss : " << trainLoss << endl;
            cout << "Train Accuracy : " << trainAccuracy << endl;
            cout << "Train Charlotte Loss: " << trLossCh << endl;
            cout << "Train Ivy Loss: " << trLossIv << endl;
            cout << "Train Leah Loss: " << trLossLe << endl;
            cout << "Train Charlotte Accuracy: " << trAccCh << endl;
            cout << "Train Ivy Accuracy: " << trAccIv << endl;
            cout << "Train Leah Accuracy: " << trAccLe << endl;

            auto [testLoss, testAccuracy, teLossCh, teLossIv, teLossLe, teAccCh, teAccIv, teAccLe] = diagnose(*X_test, yTestBatch, &testFlattened);
            cout << "Test Loss : " << testLoss << endl;
            cout << "Test Accuracy : " << testAccuracy << endl;
            cout << "Test Charlotte Loss: " << teLossCh << endl;
            cout << "Test Ivy Loss: " << teLossIv << endl;
            cout << "Test Leah Loss: " << teLossLe << endl;
            cout << "Test Charlotte Accuracy: " << teAccCh << endl;
            cout << "Test Ivy Accuracy: " << teAccIv << endl;
            cout << "Test Leah Accuracy: " << teAccLe << endl;
        }
        
        auto [batch, batchY, indexes] = makeBatch(10);
        Tensor cached = flattened.pick(indexes); // m x n
        cached.transpose(); // n x m

        vector<variant<ClassifierOutput, ConvOutput>> outputs = forwardProp(batch, &cached);
        backwardProp(outputs, batch, batchY, &cached, 0);
    }
}

void CNN::writeParameters(string fileName) {
    json j;
    j["features.0.weight"] = convWeights[0].d;
    j["features.0.bias"] = convBias[0];
    j["features.3.weight"] = convWeights[2].d;
    j["features.3.bias"] = convBias[2];
    j["features.6.weight"] = convWeights[4].d;
    j["features.6.bias"] = convBias[4];
    j["features.8.weight"] = convWeights[5].d;
    j["features.8.bias"] = convBias[5];
    j["features.11.weight"] = convWeights[7].d;
    j["features.11.bias"] = convBias[7];
    j["features.13.weight"] = convWeights[8].d;
    j["features.13.bias"] = convBias[8];
    j["features.16.weight"] = convWeights[10].d;
    j["features.16.bias"] = convBias[10];
    j["features.18.weight"] = convWeights[11].d;
    j["features.18.bias"] = convBias[11];

    j["classifier.0.weight"] = classifierWeights[0].d;
    j["classifier.0.bias"] = classifierBias[0];
    j["classifier.1.weight"] = classifierWeights[1].d;
    j["classifier.1.bias"] = classifierBias[1];

    ofstream out(fileName);
    out << j.dump(2);
    cout << "Wrote parameters to " << fileName;
}

void RunHTTPServer(CNN &model) {
    int PORT = 8000;
    cout << "Starting HTTP server on PORT : " << PORT << endl;

    httplib::Server server;
    server.set_payload_max_length(1024 * 1024 * 20);
    server.set_default_headers({
        {"Access-Control-Allow-Origin", "http://localhost:3000"},
        {"Access-Control-Allow-Methods", "GET, POST, OPTIONS"},
        {"Access-Control-Allow-Headers", "Content-Type"}
    });

    const int IMAGE_SIZE = IMAGE_HEIGHT * IMAGE_WIDTH * 3;

    server.Post("/predict", [&model](const httplib::Request& req, httplib::Response& res) {
        auto start = chrono::steady_clock::now();

        auto ct = req.get_header_value("Content-Type");
        if (ct.rfind("application/json", 0) != 0) {
            res.status = 415;
            res.set_content(json{{"error", "invalid content type"}}.dump(), "application/json");
            return;
        }

        json body = json::parse(req.body, nullptr, false);
        if (body.is_discarded()) {
            res.status = 400;
            res.set_content(json{{"error", "malformed JSON body"}}.dump(), "application/json");
            return;
        }

        if (!body.is_object()) {
            res.status = 400;
            res.set_content(json{{"error", "body must be JSON object"}}.dump(), "application/json");
            return;
        }

        if (!body.contains("pixels") || !body["pixels"].is_array()) {
            res.status = 400;
            res.set_content(json{{"error", "'pixels' must be an array"}}.dump(), "application/json");
            return;
        }

        const auto &pixels = body["pixels"].get<vector<double>>();
        if (pixels.size() != IMAGE_SIZE) {
            res.status = 400;
            res.set_content(json{{"error", "expected 3x224x224 image"}}.dump(), "application/json");
            return;
        }

        Tensor img({3, IMAGE_HEIGHT, IMAGE_WIDTH}, pixels);
        double* id = img.d.data();
        for (int ch = 0; ch < img.depth(); ++ch) {
            double* cd = id + ch * img.rows * img.cols;
            for (int p = 0; p < img.rows * img.cols; ++p) {
                double* v = &cd[p];

                if (*v < 0 || *v > 255) {
                    res.status = 400;
                    res.set_content(json{{"error", "expected pixels in [0, 255]"}}.dump(), "application/json");
                    return;
                }

                *v = (*v / 255.0 - model.mus[ch]) / model.stds[ch];
            }
        }

        auto [predictions, scores] = model.predict(img);
        res.status = 201;
        res.set_content(json{{"predictions", predictions}, {"scores", scores}}.dump(), "application/json");

        auto end = chrono::steady_clock::now();
        double sec = chrono::duration<double>(end - start).count();
        println("Inference in {}s with predictions {} and scores {}", sec, predictions, scores);
    });

    server.Options("/predict", [](const httplib::Request&, httplib::Response& res) {
        res.status = 204;
    });

    server.Get("/total-parameters", [&model](const httplib::Request& req, httplib::Response& res) {
        res.status = 200;
        res.set_content(json{{"parameters", model.totalParameters}}.dump(), "application/json");
    });

    bool listening = server.listen("0.0.0.0", PORT);
}

int main(int argc, char* argv[]) {
    cout << "main()" << endl;
    vector<int> convLayers =  {64, -1, 128, -1, 256, 256, -1, 512, 512, -1, 512, 512, -1};
    vector<int> classifierLayers = {256, 3};
    vector<double> mus = {0.485, 0.456, 0.406};
    vector<double> stds = {0.229, 0.224, 0.225};
    CNN model(convLayers, classifierLayers, mus, stds);

    if (argc < 2) {
        cerr << "Mode argument not specified" << endl;
        abort();
    }

    string mode = argv[1];
    cout << "Running mode : " << mode << endl;
    if (mode == "train") {
        model.loadData("train.csv", false);
        model.loadData("test.csv", true);
        model.loadParameters("vgg11weights.json", true);
        model.train(2000);
        model.writeParameters("params.json");
    } if (mode == "inference") {
        model.loadParameters("params.json", false);
        RunHTTPServer(model);
    } else {
        cerr << "Invalid mode : " << mode << endl;
    }

    return 0;
}
