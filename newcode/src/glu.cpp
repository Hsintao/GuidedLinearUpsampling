#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <algorithm>
#include <memory>
#include <cstdint>

using namespace cv;

inline bool nofill(const Mat& m) {
    return m.elemSize() * m.cols == m.step;
}

class GLU
{
    Mat2i  _upsampleIdx;
    Mat1f  _upsampleW;
    Mat1i  _downsampleIdx;

    struct Builder
    {
        Mat2i  idx;
        Mat1f  w;
        const Mat3b& large;
        Mat3b&       small;
        int     sstride, swidth;
        int     idxOffset[9];

    public:
        Builder(const Mat3b* _large, Mat3b* _small)
            : large(*_large), small(*_small)
        {
            idx.create(large.size());
            w.create(large.size());

            CV_Assert(nofill(*_large) && nofill(*_small) && nofill(idx) && nofill(w));

            sstride = small.step, swidth = small.cols;
            const int _idxOffset[] = { -swidth - 1,-swidth,-swidth + 1,-1,0,1,swidth - 1,swidth,swidth + 1 };
            memcpy(idxOffset, _idxOffset, sizeof(idxOffset));
        }

        static int makeBorder(int x, int width)
        {
            return x <= 0 ? 1 : x >= width - 1 ? width - 2 : x;
        }

        static float getInterpError(const uchar* c, const uchar* F, const uchar* B, float w) {
            float err = 0;
            for (int i = 0; i < 3; ++i)
            {
                float d = B[i] + w * (float(F[i]) - B[i]) - c[i];
                err += fabs(d);
            }
            return err;
        }

        Mat1b getCurrentErrorMap() const
        {
            Mat1b err(large.size());

            CV_Assert(small.depth() == CV_8U);
            if (small.step != small.channels() * small.cols)
                small = small.clone();
            CV_Assert(small.step == small.channels() * small.cols);

            const uchar* smallData = small.data;
            const int cn = small.channels();

            for (int y = 0; y < idx.rows; ++y) {
                for (int x = 0; x < idx.cols; ++x) {
                    const int* idxPtr = idx.ptr<int>(y, x);
                    float wVal = w.at<float>(y, x);
                    const uchar* c = large.ptr<uchar>(y, x);
                    const uchar* p = smallData + idxPtr[0] * cn;
                    const uchar* q = smallData + idxPtr[1] * cn;
                    float e = getInterpError(c, p, q, wVal);
                    err.at<uchar>(y, x) = uchar(e / 3.f);
                }
            }
            return err;
        }

        float update_linear_fast(int pi, int qi)
        {
            const uchar* p = large.data + pi * 3;
            const uchar* q = small.data + qi * 3;

            const uchar* qnbr[] = { q - sstride - 3,q - sstride, q - sstride + 3, q - 3, q, q + 3, q + sstride - 3, q + sstride, q + sstride + 3 };
            float vdiff[9];

            float minErr = FLT_MAX, wm = 0;
            int im = -1, jm = -1;

            for (int i = 0; i < 9; ++i)
            {
                const uchar* a = qnbr[i];
                int dv[] = { int(p[0]) - a[0], int(p[1]) - a[1], int(p[2]) - a[2] };
                vdiff[i] = sqrt(float(dv[0] * dv[0] + dv[1] * dv[1] + dv[2] * dv[2]) + 1e-3f);
                if (vdiff[i] < minErr)
                {
                    minErr = vdiff[i];
                    im = i;
                }
            }

            minErr = FLT_MAX;
            for (int j = 0; j < 9; ++j)
            {
                float w = vdiff[j] / (vdiff[im] + vdiff[j]);

                float err = getInterpError(p, qnbr[im], qnbr[j], w);
                if (err < minErr)
                {
                    minErr = err;
                    jm = j;
                    wm = w;
                }
            }
            ((Vec2i*)idx.data)[pi] = Vec2i(qi + idxOffset[im], qi + idxOffset[jm]);
            ((float*)w.data)[pi] = wm;

            return minErr;
        }

        float update_linear_full(int pi, int qi)
        {
            const uchar* p = large.data + pi * 3;
            const uchar* q = small.data + qi * 3;

            const uchar* qnbr[] = { q - sstride - 3,q - sstride, q - sstride + 3, q - 3, q, q + 3, q + sstride - 3, q + sstride, q + sstride + 3 };

            float minErr = FLT_MAX, wm = 0;
            int im = -1, jm = -1;
            for (int i = 0; i < 9; ++i)
                for (int j = i + 1; j < 9; ++j)
                {
                    const uchar* a = qnbr[i], * b = qnbr[j];
                    Vec3f dab(float(a[0]) - b[0], float(a[1]) - b[1], float(a[2]) - b[2]);
                    Vec3f dpb(float(p[0]) - b[0], float(p[1]) - b[1], float(p[2]) - b[2]);
                    float w = dab.dot(dpb) / dab.dot(dab);
                    if (w < 0.f)
                        w = 0.f;
                    else if (w > 1.f)
                        w = 1.f;

                    float err = getInterpError(p, qnbr[i], qnbr[j], w);
                    if (err < minErr)
                    {
                        minErr = err;
                        im = i; jm = j;
                        wm = w;
                    }
                }

            ((Vec2i*)idx.data)[pi] = Vec2i(qi + idxOffset[im], qi + idxOffset[jm]);
            ((float*)w.data)[pi] = wm;

            return minErr;
        }

        float update(int pi, int qi)
        {
            return update_linear_fast(pi, qi);
        }
    };

    static void _initSampleIndex(Size largeSize, double downscale, Mat1i& smallIdx, Mat1i& largeIdx)
    {
        Size dsize(int(largeSize.width * downscale + 0.5) + 1, int(largeSize.height * downscale + 0.5) + 1);

        smallIdx.create(dsize);
        for (int y = 0; y < dsize.height; ++y)
        {
            int py = int((y + 0.5) / downscale + 0.5);
            if (py >= largeSize.height)
                py = largeSize.height - 1;
            for (int x = 0; x < dsize.width; ++x)
            {
                int px = int((x + 0.5) / downscale + 0.5);
                if (px >= largeSize.width)
                    px = largeSize.width - 1;

                smallIdx.at<int>(y, x) = py * largeSize.width + px;
            }
        }

        largeIdx.create(largeSize);
        for (int y = 0; y < largeIdx.rows; ++y)
        {
            const int dy = Builder::makeBorder(int(y * downscale + 0.5f), dsize.height);

            for (int x = 0; x < largeIdx.cols; ++x)
            {
                const int dx = Builder::makeBorder(int(x * downscale + 0.5f), dsize.width);
                largeIdx.at<int>(y, x) = dy * dsize.width + dx;
            }
        }
    }

    static Mat _downsample(const Mat& large, const Mat1i& smallIdx)
    {
        CV_Assert(large.depth() == CV_8U && large.step == large.cols * large.channels());
        const int cn = large.channels();
        const uchar* largeData = large.data;
        Mat small(smallIdx.size(), large.type());
        for (int y = 0; y < small.rows; ++y) {
            for (int x = 0; x < small.cols; ++x) {
                int idx = smallIdx.at<int>(y, x);
                const uchar* src = largeData + idx * cn;
                uchar* dst = small.ptr<uchar>(y, x);
                for (int j = 0; j < cn; ++j)
                    dst[j] = src[j];
            }
        }
        return small;
    }

public:
    Mat upsample(Mat small)
    {
        CV_Assert(small.size() == _downsampleIdx.size());

        Mat large(_upsampleIdx.size(), small.type());
        CV_Assert(small.depth() == CV_8U);
        if (small.step != small.channels() * small.cols)
            small = small.clone();
        CV_Assert(small.step == small.channels() * small.cols);

        const uchar* smallData = small.data;
        const int cn = small.channels();

        for (int y = 0; y < _upsampleIdx.rows; ++y) {
            for (int x = 0; x < _upsampleIdx.cols; ++x) {
                const int* idxPtr = _upsampleIdx.ptr<int>(y, x);
                float wVal = _upsampleW.at<float>(y, x);
                uchar* c = large.ptr<uchar>(y, x);
                const uchar* p = smallData + idxPtr[0] * cn;
                const uchar* q = smallData + idxPtr[1] * cn;
                for (int i = 0; i < cn; ++i)
                    c[i] = uchar(q[i] + wVal * (int(p[i]) - q[i]));
            }
        }
        return large;
    }

    Mat downsample(Mat large)
    {
        CV_Assert(large.size() == _upsampleIdx.size());
        return this->_downsample(large, _downsampleIdx);
    }

    Mat3b build(const Mat3b& large, double downscale, bool optimizeDownsample = true, int errT = 30, int regionSizeT = 5, int maxItr = 2)
    {
        Mat1i smallIdx, largeIdx;
        this->_initSampleIndex(large.size(), downscale, smallIdx, largeIdx);
        Mat3b small = _downsample(large, smallIdx);

        Builder builder(&large, &small);

        {
            CV_Assert(nofill(largeIdx));
            const int* largeIdxData = (const int*)largeIdx.data;

            parallel_for_(cv::Range(0, large.rows * large.cols), [&builder, largeIdxData](const cv::Range& r) {
                for (int p = r.start; p < r.end; ++p)
                {
                    builder.update(p, largeIdxData[p]);
                }
            });
        }

        if (optimizeDownsample)
        {
            const int W = large.cols, smallW = small.cols;

            Mat2i invIdx(small.size());

            {
                invIdx.setTo(Scalar(-1, -1));
                int* dq = (int*)invIdx.data;

                for (int y = 0; y < largeIdx.rows; ++y) {
                    for (int x = 0; x < largeIdx.cols; ++x) {
                        int q = largeIdx.at<int>(y, x);
                        int p = y * W + x;

                        int* v = dq + q * 2;

                        if (v[0] < 0)
                            v[0] = p;
                        v[1] = p;
                    }
                }
            }

            struct CC
            {
                int size = 0;
                int err = 0;
                std::vector<int>  largePixels;
            };

            Mat1b err = builder.getCurrentErrorMap();

            for (int itr = 0; itr < maxItr; ++itr)
            {
                Mat1b mask(err.size());
                CV_Assert(mask.step == mask.cols);

                cv::threshold(err, mask, errT, 0, THRESH_BINARY);

                Mat1i cc;
                int ncc = connectedComponents(mask, cc);
                std::unique_ptr<CC[]> _vcc(new CC[ncc]);
                CC* vcc = _vcc.get();

                for (int y = 0; y < cc.rows; ++y) {
                    for (int x = 0; x < cc.cols; ++x) {
                        int i = cc.at<int>(y, x);
                        uchar m = mask.at<uchar>(y, x);
                        uchar e = err.at<uchar>(y, x);
                        if (m)
                        {
                            vcc[i].size++;
                            vcc[i].err += e;
                        }
                    }
                }

                std::vector<CC*> selCC;
                selCC.reserve(ncc);
                for (int i = 0; i < ncc; ++i)
                    if (vcc[i].size > regionSizeT)
                        selCC.push_back(vcc + i);
                    else
                        vcc[i].size = 0;

                std::sort(selCC.begin(), selCC.end(), [](const CC* a, const CC* b) {
                    return a->err > b->err;
                });

                int maxRegionSize = 0;
                for (int i = 0; i < ncc; ++i) {
                    if (vcc[i].size > maxRegionSize) maxRegionSize = vcc[i].size;
                }

                for (auto* cc : selCC)
                    cc->largePixels.reserve(maxRegionSize);

                for (int y = 0; y < mask.rows; ++y) {
                    for (int x = 0; x < mask.cols; ++x) {
                        uchar m = mask.at<uchar>(y, x);
                        int ci = cc.at<int>(y, x);
                        if (m && vcc[ci].size > 0)
                            vcc[ci].largePixels.push_back(y * W + x);
                    }
                }

                const int* largeIdxData = (const int*)largeIdx.data;
                int* smallIdxData = (int*)smallIdx.data;
                const Vec2i* invIdxData = (const Vec2i*)invIdx.data;

                CV_Assert(selCC.size() < USHRT_MAX);
                Mat1s _smallLabel = Mat1s::zeros(small.size());
                ushort* smallLabel = (ushort*)_smallLabel.data;
                ushort label = 0;

                const int qNbrs[] = { -smallW - 1,-smallW,-smallW + 1,-1,1,smallW - 1,smallW,smallW + 1 };
                const uint qSize = uint(small.rows * small.cols);

                std::unique_ptr<int[]> _qbuf(new int[maxRegionSize]);
                int* qbuf = _qbuf.get();
                struct PD
                {
                    Vec2i idx;
                    float w;
                    uchar err;
                };

                int maxDilatedRegionSize = int((maxRegionSize * downscale + 1) / (downscale * downscale) * 9);
                if (maxDilatedRegionSize > large.cols * large.rows)
                    maxDilatedRegionSize = large.cols * large.rows;

                std::unique_ptr<PD[]> _pbuf(new PD[maxDilatedRegionSize]);
                PD* pbuf = _pbuf.get();

                const int maxSmallPixels = maxRegionSize * 9;
                std::unique_ptr<int[]> _ibuf(new int[maxDilatedRegionSize + maxSmallPixels]);
                int* largePixels = _ibuf.get(), * smallPixels = largePixels + maxDilatedRegionSize;

                Vec2i* idxData = (Vec2i*)builder.idx.data;
                float* wData = (float*)builder.w.data;
                CV_Assert(nofill(err) && nofill(largeIdx) && nofill(builder.idx) && nofill(builder.w));

                Vec3b* smallDataPtr = (Vec3b*)small.data;
                const Vec3b* largeDataPtr = (const Vec3b*)large.data;
                Mat1b _smallBuf = Mat1b::zeros(small.size());
                uchar* smallBuf = (uchar*)_smallBuf.data;
                CV_Assert(nofill(small) && nofill(large) && nofill(_smallBuf));

                for (auto* cc : selCC)
                {
                    ++label;
                    int nSmallPixels = 0;
                    for (auto p : cc->largePixels)
                    {
                        int q = largeIdxData[p];
                        if (smallLabel[q] != label)
                        {
                            smallPixels[nSmallPixels++] = q;
                            smallLabel[q] = label;
                            smallBuf[q] = 0;
                        }
                    }
                    for (int i = 0; i < nSmallPixels; ++i)
                        qbuf[i] = smallIdxData[smallPixels[i]];

                    uchar* errData = (uchar*)err.data;

                    for (auto p : cc->largePixels)
                    {
                        const int q = largeIdxData[p];

                        if (errData[p] > smallBuf[q])
                        {
                            smallIdxData[q] = p;
                            smallDataPtr[q] = largeDataPtr[p];
                            smallBuf[q] = errData[p];
                        }
                    }

                    int nDilatedSmallPixels = nSmallPixels;
                    for (int i = 0; i < nSmallPixels; ++i)
                    {
                        int q = smallPixels[i];
                        for (int j = 0; j < 8; ++j)
                        {
                            int nbr = q + qNbrs[j];
                            if (uint(nbr) < qSize && smallLabel[nbr] != label && invIdxData[nbr][0] >= 0)
                            {
                                smallPixels[nDilatedSmallPixels++] = nbr;
                                smallLabel[nbr] = label;
                            }
                        }
                    }
                    CV_Assert(nDilatedSmallPixels <= maxSmallPixels);

                    int nLargePixels = 0;
                    for (int i = 0; i < nDilatedSmallPixels; ++i)
                    {
                        int q = smallPixels[i];

                        Vec2i mm = invIdxData[q];
                        const int width = mm[1] % W - mm[0] % W + 1;
                        for (int pr = mm[0]; pr < mm[1]; pr += W)
                        {
                            for (int p = pr; p < pr + width; ++p)
                            {
                                largePixels[nLargePixels++] = p;
                            }
                        }
                    }
                    CV_Assert(nLargePixels <= maxDilatedRegionSize);

                    int curErr = 0, updatedErr = 0;
                    for (int i = 0; i < nLargePixels; ++i)
                    {
                        const int p = largePixels[i];

                        pbuf[i].idx = idxData[p];
                        pbuf[i].w = wData[p];
                        pbuf[i].err = errData[p];

                        curErr += errData[p];

                        uchar e = uchar(builder.update(p, largeIdxData[p]) / 3.f);
                        errData[p] = e;
                        updatedErr += e;
                    }

                    if (updatedErr > curErr)
                    {
                        for (int i = 0; i < nLargePixels; ++i)
                        {
                            const int p = largePixels[i];

                            idxData[p] = pbuf[i].idx;
                            wData[p] = pbuf[i].w;
                            errData[p] = pbuf[i].err;
                        }
                        for (int i = 0; i < nSmallPixels; ++i)
                        {
                            const int q = smallPixels[i], p = qbuf[i];

                            smallIdxData[q] = p;
                            smallDataPtr[q] = largeDataPtr[p];
                        }
                    }
                }
            }
        }

        _upsampleIdx = builder.idx;
        _upsampleW = builder.w;
        _downsampleIdx = smallIdx;

        return small;
    }
};

void glu_self_upsampling()
{
    std::string file = "images/img01.png";
    Mat3b large = imread(file);

    if (large.empty()) {
        std::cerr << "Failed to load image: " << file << std::endl;
        return;
    }

    double ratio = 1.0 / 4;

    GLU glu;
    Mat3b small = glu.build(large, ratio);

    Mat3b upsampled = glu.upsample(small);

    Mat3b smallScaled;
    cv::resize(small, smallScaled, large.size(), 0, 0, INTER_NEAREST);

    imshow("input", large);
    imshow("downsampled", smallScaled);
    imshow("upsampled", upsampled);
    cv::imwrite("images/input.png", large);
    cv::imwrite("images/downsampled.png", smallScaled);
    cv::imwrite("images/upsampled.png", upsampled);
    cv::waitKey();
}

void glu_guided_upsampling()
{
    Mat3b source = imread("images/img01.png");
    Mat   target = imread("images/alpha01.png", IMREAD_GRAYSCALE);

    if (source.empty() || target.empty()) {
        std::cerr << "Failed to load images" << std::endl;
        return;
    }

    double ratio = 1.0 / 4;

    GLU glu;

    Mat3b smallSource = glu.build(source, ratio);
    Mat smallTarget;

    smallTarget = glu.downsample(target);

    Mat1b linearUpsampled;
    cv::resize(smallTarget, linearUpsampled, source.size(), 0, 0, INTER_LINEAR);

    imshow("linearUpsampled", linearUpsampled);
    cv::imwrite("images/linearUpsampled.png", linearUpsampled);
    cv::waitKey();
    imshow("reference", target);
    cv::imwrite("images/reference.png", target);
    cv::waitKey();

    Mat1b upsampledTarget = glu.upsample(smallTarget);
    imshow("upsampledTarget", upsampledTarget);
    cv::imwrite("images/upsampledTarget.png", upsampledTarget);

    cv::waitKey();
}