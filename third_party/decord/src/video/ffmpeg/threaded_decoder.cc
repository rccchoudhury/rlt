/*!
 *  Copyright (c) 2019 by Contributors if not otherwise specified
 * \file threaded_decoder.cc
 * \brief FFmpeg threaded decoder Impl
 */

#include "threaded_decoder.h"

#include <random>
#include <math.h>
#include <dmlc/logging.h>

namespace decord {
namespace ffmpeg {

FFMPEGThreadedDecoder::FFMPEGThreadedDecoder() : frame_count_(0), draining_(false), run_(false), error_status_(false), error_message_() {
}

void FFMPEGThreadedDecoder::SetCodecContext(AVCodecContext *dec_ctx, int width, int height, int rotation,
                                            bool use_rrc, int orig_width, int orig_height, double scale_min, double scale_max, double ratio_min, double ratio_max,
                                            bool use_msc,
                                            bool use_rcc,
                                            bool use_centercrop,
                                            bool use_fixedcrop, int crop_x, int crop_y,
                                            double hflip_prob, double vflip_prob) {
    bool running = run_.load();
    Clear();
    dec_ctx_.reset(dec_ctx);
    // LOG(INFO) << dec_ctx->width << " x " << dec_ctx->height << " : " << dec_ctx->time_base.num << " , " << dec_ctx->time_base.den;
    // std::string descr = "scale=320:240";
    char descr[128];
    int cx;
    // determine if we need to do hflip (vflip)
    std::random_device rd;
    std::mt19937_64 generator(rd());
    std::uniform_real_distribution<double> hflip_sampler(0, 1);
    std::uniform_real_distribution<double> vflip_sampler(0, 1);
    bool hflip = false;
    bool vflip = false;
    // double t1 = hflip_sampler(generator);
    // double t2 = vflip_sampler(generator);
    // LOG(INFO) << "t1: " << t1 << "; t2: " << t2;
    if ((hflip_prob > 0.) && (hflip_sampler(generator) < hflip_prob)) {
        hflip = true;
    }
    if ((vflip_prob > 0.) && (vflip_sampler(generator) < vflip_prob)) {
        vflip = true;
    }
    if (use_rrc) {
        int area = orig_width * orig_height;
        double log_ratio_min = log(ratio_min);
        double log_ratio_max = log(ratio_max);
        int out_w = -1;
        int out_h = -1;
        int x = -1;
        int y = -1;
        std::uniform_real_distribution<double> scale_sampler(scale_min, scale_max);
        std::uniform_real_distribution<double> ratio_sampler(log_ratio_min, log_ratio_max);
        bool success = false;
        for (int i = 0; (i < 10) && (!success); ++i) {
            double target_area = area * scale_sampler(generator);
            double aspect_ratio = exp(ratio_sampler(generator));
            out_w = int(round(sqrt(target_area * aspect_ratio)));
            out_h = int(round(sqrt(target_area / aspect_ratio)));
            if ((out_w > 0) && (out_w <= orig_width) && (out_h > 0) && (out_h <= orig_height)) {
                std::uniform_int_distribution<> y_sampler(0, orig_height - out_h + 1);
                std::uniform_int_distribution<> x_sampler(0, orig_width - out_w + 1);
                y = y_sampler(generator);
                x = x_sampler(generator);
                success = true;
            }
        }
        if (!success) {
            // Fallback to central crop
            double in_ratio = double(orig_width) / double(orig_height);
            if (in_ratio < ratio_min) {
                out_w = orig_width;
                out_h = int(round(out_w / ratio_min));
            } else if (in_ratio > ratio_max) {
                out_h = orig_height;
                out_w = int(round(out_h * ratio_max));
            } else {
                out_h = orig_height;
                out_w = orig_width;
            }
            y = (orig_height - out_h) / 2;
            x = (orig_width - out_w) / 2;
        }
        if (rotation != 0) {
            LOG(FATAL) << "rotation != 0 is not supported for RRC!";
        } else {
            cx = std::snprintf(descr, sizeof(descr), "crop=%d:%d:%d:%d,%s%sscale=%d:%d", out_w, out_h, x, y, hflip ? "hflip," : "", vflip ? "vflip," : "", width, height);
        }
    } else if (use_msc) {
        // find a crop size
        int base_size = (orig_height > orig_width) ? orig_width : orig_height;
        int max_distort = 1;
        std::vector<float> scales = {1., 0.875, 0.75, 0.66};
        std::vector<std::tuple<int, int>> crop_pairs;
        for (int i = 0; i < scales.size(); i++) {
            for (int j = 0; j < scales.size(); j++) {
                if (abs(i - j) <= max_distort) {
                    int crop_h = int(base_size * scales[i]);
                    if (abs(crop_h - height) < 3) crop_h = height;
                    int crop_w = int(base_size * scales[j]);
                    if (abs(crop_w - width) < 3) crop_w = width;
                    crop_pairs.push_back(std::tuple<int, int>(crop_w, crop_h));
                }
            }
        }
        std::uniform_int_distribution<> crop_pair_sampler(0, crop_pairs.size() - 1);
        std::tuple<int, int> crop_pair = crop_pairs[crop_pair_sampler(generator)];
        int out_w = std::get<0>(crop_pair);
        int out_h = std::get<1>(crop_pair);
        // fill_fix_offset
        int w_step = (orig_width - out_w) / 4;
        int h_step = (orig_height - out_h) / 4;
        std::vector<std::tuple<int, int>> offsets;
        offsets.push_back(std::make_tuple(0, 0));                    // upper left
        offsets.push_back(std::make_tuple(4 * w_step, 0));           // upper right
        offsets.push_back(std::make_tuple(0, 4 * h_step));           // lower left
        offsets.push_back(std::make_tuple(4 * w_step, 4 * h_step));  // lower right
        offsets.push_back(std::make_tuple(2 * w_step, 2 * h_step));  // center
        
        offsets.push_back(std::make_tuple(2 * w_step, 0));           // upper center
        offsets.push_back(std::make_tuple(0, 2 * h_step));           // center left
        offsets.push_back(std::make_tuple(2 * w_step, 4 * h_step));  // lower center
        offsets.push_back(std::make_tuple(4 * w_step, 2 * h_step));  // center right

        offsets.push_back(std::make_tuple(1 * w_step, 1 * h_step));  // upper left quarter
        offsets.push_back(std::make_tuple(3 * w_step, 1 * h_step));  // upper right quater
        offsets.push_back(std::make_tuple(1 * w_step, 3 * h_step));  // lower left quarter
        offsets.push_back(std::make_tuple(3 * w_step, 3 * h_step));  // lower right quarter
        std::uniform_int_distribution<> offset_sampler(0, offsets.size() - 1);
        std::tuple<int, int> offset = offsets[offset_sampler(generator)];
        int x = std::get<0>(offset);
        int y = std::get<1>(offset);
    
        // LOG(INFO) << "crop_w: " << out_w << "; crop_h: " << out_h << "; x: " << x << "; y: " << y;
        if (rotation != 0) {
            LOG(FATAL) << "rotation != 0 is not supported for RRC!";
        } else {
            cx = std::snprintf(descr, sizeof(descr), "crop=%d:%d:%d:%d,%s%sscale=%d:%d", out_w, out_h, x, y, hflip ? "hflip," : "", vflip ? "vflip," : "", width, height);
        }
    } else if (use_rcc) {
        int crop_size = (orig_height > orig_width) ? orig_width : orig_height;
        int y = (orig_height - crop_size) / 2;
        int x = (orig_width - crop_size) / 2;
        if (hflip || vflip) {
            LOG(FATAL) << "hflip_prob > 0 or vflip_prob > 0 is not supported for CenterCrop!";
        }
        if (rotation != 0) {
            LOG(FATAL) << "rotation != 0 is not supported for CenterCrop!";
        } else {
            cx = std::snprintf(descr, sizeof(descr), "crop=%d:%d:%d:%d,scale=%d:%d", crop_size, crop_size, x, y, width, height);
        }
    } else if (use_fixedcrop) {
        if (hflip || vflip) {
            LOG(FATAL) << "hflip_prob > 0 or vflip_prob > 0 is not supported for FixedCrop!";
        }
        if (rotation != 0) {
            LOG(FATAL) << "rotation != 0 is not supported for FixedCrop!";
        } else {
            cx = std::snprintf(descr, sizeof(descr), "crop=%d:%d:%d:%d,%s%sscale=%d:%d", width, height, crop_x, crop_y, hflip ? "hflip," : "", vflip ? "vflip," : "", width, height);
        }        
    } else if (use_centercrop) {
        if ((width > orig_width) || (height > orig_height)) {
            LOG(FATAL) << "Width or height in too short in CenterCrop";
        }
        int y = (orig_height - height) / 2;
        int x = (orig_width - width) / 2;
        if (hflip || vflip) {
            LOG(FATAL) << "hflip_prob > 0 or vflip_prob > 0 is not supported for CenterCrop!";
        }
        if (rotation != 0) {
            LOG(FATAL) << "rotation != 0 is not supported for CenterCrop!";
        } else {
            cx = std::snprintf(descr, sizeof(descr), "crop=%d:%d:%d:%d,scale=%d:%d", width, height, x, y, width, height);
        }
    } else if (use_fixedcrop) {
        if (hflip || vflip) {
            LOG(FATAL) << "hflip_prob > 0 or vflip_prob > 0 is not supported for FixedCrop!";
        }
        if (rotation != 0) {
            LOG(FATAL) << "rotation != 0 is not supported for FixedCrop!";
        } else {
            cx = std::snprintf(descr, sizeof(descr), "crop=%d:%d:%d:%d,scale=%d:%d", width, height, crop_x, crop_y, width, height);
        }
    } else {
        switch(rotation) {
            case 90:
                cx = std::snprintf(descr, sizeof(descr), "transpose=1,scale=%d:%d", width, height);
                break;
            case 180:
                cx = std::snprintf(descr, sizeof(descr), "transpose=1,transpose=1,scale=%d:%d", width, height);
                break;
            case 270:
                cx = std::snprintf(descr, sizeof(descr), "transpose=2,scale=%d:%d", width, height);
                break;
            case 0:
            default:
                cx = std::snprintf(descr, sizeof(descr), "scale=%d:%d", width, height);
        }
    }
    filter_graph_ = FFMPEGFilterGraphPtr(new FFMPEGFilterGraph(descr, dec_ctx_.get()));
    if (running) {
        Start();
    }
}

void FFMPEGThreadedDecoder::Start() {
    CheckErrorStatus();
    if (!run_.load()) {
        pkt_queue_.reset(new PacketQueue());
        frame_queue_.reset(new FrameQueue());
        buffer_queue_.reset(new BufferQueue());
        run_.store(true);
        auto t = std::thread(&FFMPEGThreadedDecoder::WorkerThread, this);
        std::swap(t_, t);
    }
}

void FFMPEGThreadedDecoder::Stop() {
    if (run_.load()) {
        if (pkt_queue_) {
            pkt_queue_->SignalForKill();
        }
        if (buffer_queue_) {
            buffer_queue_->SignalForKill();
        }
        run_.store(false);
        if (frame_queue_) {
            frame_queue_->SignalForKill();
        }
    }
    if (t_.joinable()) {
        // LOG(INFO) << "joining";
        t_.join();
    }
}

void FFMPEGThreadedDecoder::Clear() {
    Stop();
    if (dec_ctx_.get()) {
        avcodec_flush_buffers(dec_ctx_.get());
    }
    frame_count_.store(0);
    draining_.store(false);
    {
      std::lock_guard<std::mutex> lock(pts_mutex_);
      discard_pts_.clear();
    }
    error_status_.store(false);
    {
        std::lock_guard<std::mutex> lock(error_mutex_);
        error_message_.clear();
    }
}

void FFMPEGThreadedDecoder::SuggestDiscardPTS(std::vector<int64_t> dts) {
    std::lock_guard<std::mutex> lock(pts_mutex_);
    discard_pts_.insert(dts.begin(), dts.end());
}

void FFMPEGThreadedDecoder::ClearDiscardPTS() {
    std::lock_guard<std::mutex> lock(pts_mutex_);
    discard_pts_.clear();
}

void FFMPEGThreadedDecoder::Push(AVPacketPtr pkt, runtime::NDArray buf) {
    CHECK(run_.load());
    if (!pkt) {
        CHECK(!draining_.load()) << "Start draining twice...";
        draining_.store(true);
    }

    pkt_queue_->Push(pkt);
    buffer_queue_->Push(buf);

    // LOG(INFO)<< "frame push: " << frame_count_;
    // LOG(INFO) << "Pushed pkt to pkt_queue";
}

bool FFMPEGThreadedDecoder::Pop(runtime::NDArray *frame) {
    // Pop is blocking operation
    // unblock and return false if queue has been destroyed.

    CheckErrorStatus();
    if (!frame_count_.load() && !draining_.load()) {
        return false;
    }
    // LOG(INFO) << "Waiting for pop";
    bool ret = frame_queue_->Pop(frame);
    // LOG(INFO) << "Poped";
    CheckErrorStatus();

    if (ret) {
        --frame_count_;
    }
    return (ret && frame->data_);
}

FFMPEGThreadedDecoder::~FFMPEGThreadedDecoder() {
    Stop();
}

void FFMPEGThreadedDecoder::ProcessFrame(AVFramePtr frame, NDArray out_buf) {
    frame->pts = frame->best_effort_timestamp;
    bool skip = false;
    {
      std::lock_guard<std::mutex> lock(pts_mutex_);
      skip = discard_pts_.find(frame->pts) != discard_pts_.end();
    }
    if (skip) {
        // skip resize/filtering
        NDArray empty = NDArray::Empty({1}, kUInt8, kCPU);
        empty.pts = frame->pts;
        frame_queue_->Push(empty);
        ++frame_count_;
        return;
    }
    // filter image frame (format conversion, scaling...)
    filter_graph_->Push(frame.get());
    AVFramePtr out_frame = AVFramePool::Get()->Acquire();
    AVFrame *out_frame_p = out_frame.get();
    CHECK(filter_graph_->Pop(&out_frame_p)) << "Error fetch filtered frame.";

    auto tmp = AsNDArray(out_frame);
    if (out_buf.defined()) {
        CHECK(out_buf.Size() == tmp.Size());
        out_buf.CopyFrom(tmp);
        frame_queue_->Push(out_buf);
        ++frame_count_;
    } else {
        frame_queue_->Push(tmp);
        ++frame_count_;
    }
}

void FFMPEGThreadedDecoder::WorkerThread() {
    try {
        WorkerThreadImpl();
    } catch (dmlc::Error error) {
        RecordInternalError(error.what());
        run_.store(false);
        frame_queue_->SignalForKill(); // Unblock all consumers
    }
}

void FFMPEGThreadedDecoder::WorkerThreadImpl() {
    while (run_.load()) {
        // CHECK(filter_graph_) << "FilterGraph not initialized.";
        if (!filter_graph_) return;
        AVPacketPtr pkt;

        int got_picture;
        bool ret = pkt_queue_->Pop(&pkt);
        if (!ret) {
            return;
        }
        AVFramePtr frame = AVFramePool::Get()->Acquire();
        if (!pkt) {
            // LOG(INFO) << "Draining mode start...";
            // draining mode, pulling buffered frames out
            CHECK_GE(avcodec_send_packet(dec_ctx_.get(), NULL), 0) << "Thread worker: Error entering draining mode.";
            while (true) {
                got_picture = avcodec_receive_frame(dec_ctx_.get(), frame.get());
                if (got_picture == AVERROR_EOF) {
                    // LOG(INFO) << "stop draining";
                    for (int cnt = 0; cnt < 128; ++cnt) {
                        // special signal
                        frame_queue_->Push(NDArray::Empty({1}, kInt64, kCPU));
                        ++frame_count_;
                    }
                    draining_.store(false);
                    break;
                }
                NDArray out_buf;
                bool get_buf = buffer_queue_->Pop(&out_buf);
                if (!get_buf) return;
                ProcessFrame(frame, out_buf);
            }
        } else {
            // normal mode, push in valid packets and retrieve frames
            CHECK_GE(avcodec_send_packet(dec_ctx_.get(), pkt.get()), 0) << "Thread worker: Error sending packet.";
            got_picture = avcodec_receive_frame(dec_ctx_.get(), frame.get());
            if (got_picture == 0) {
                NDArray out_buf;
                bool get_buf = buffer_queue_->Pop(&out_buf);
                if (!get_buf) return;
                ProcessFrame(frame, out_buf);
            } else if (AVERROR(EAGAIN) == got_picture || AVERROR_EOF == got_picture) {
                frame_queue_->Push(NDArray());
                ++frame_count_;
            } else {
                LOG(FATAL) << "Thread worker: Error decoding frame: " << got_picture;
            }
        }
        // free raw memories allocated with ffmpeg
        // av_packet_unref(pkt);
    }
}

NDArray FFMPEGThreadedDecoder::CopyToNDArray(AVFramePtr p) {
    CHECK(p) << "Error: converting empty AVFrame to DLTensor";
    // int channel = p->linesize[0] / p->width;
    CHECK(AVPixelFormat(p->format) == AV_PIX_FMT_RGB24 || AVPixelFormat(p->format) == AV_PIX_FMT_GRAY8)
        << "Only support RGB24/GRAY8 image to NDArray conversion, given: "
        << AVPixelFormat(p->format);
    int channel = AVPixelFormat(p->format) == AV_PIX_FMT_RGB24 ? 3 : 1;
    // CHECK(p->linesize[0] % p->width == 0)
    //     << "AVFrame data is not a compact array. linesize: " << p->linesize[0]
    //     << " width: " << p->width;

    DLContext ctx;
    CHECK(!p->hw_frames_ctx) << "Not supported hw_frames_ctx";
    ctx = kCPU;
    NDArray arr = NDArray::Empty({p->height, p->width, channel}, kUInt8, ctx);
    auto device_api = runtime::DeviceAPI::Get(ctx);
    void *to_ptr = arr.data_->dl_tensor.data;
    void *from_ptr = p->data[0];
    int linesize = p->width * channel;

    // arr.CopyFrom(&dlt);
    for (int i = 0; i < p->height; ++i) {
        // copy line by line
        device_api->CopyDataFromTo(
            from_ptr, i * p->linesize[0],
            to_ptr, i * linesize,
            linesize, ctx, ctx, kUInt8, nullptr);
    }
    arr.pts = p->pts;
    return arr;
}

static void AVFrameManagerDeleter(DLManagedTensor *manager) {
	delete static_cast<AVFrameManager*>(manager->manager_ctx);
	delete manager;
}

NDArray FFMPEGThreadedDecoder::AsNDArray(AVFramePtr p) {
    if (p->linesize[0] % p->width != 0) {
        // Fallback to copy since original AVFrame is not compact
        return CopyToNDArray(p);
    }
	DLManagedTensor* manager = new DLManagedTensor();
    auto av_manager = new AVFrameManager(p);
	manager->manager_ctx = av_manager;
	ToDLTensor(p, manager->dl_tensor, av_manager->shape);
	manager->deleter = AVFrameManagerDeleter;
	NDArray arr = NDArray::FromDLPack(manager);
    arr.pts = p->pts;
	return arr;
}

void FFMPEGThreadedDecoder::CheckErrorStatus() {
    if (error_status_.load()) {
        std::lock_guard<std::mutex> lock(error_mutex_);
        LOG(FATAL) << error_message_;
    }
}

void FFMPEGThreadedDecoder::RecordInternalError(std::string message) {
    {
        std::lock_guard<std::mutex> lock(error_mutex_);
        error_message_ = message;
    }
    error_status_.store(true);
}

}  // namespace ffmpeg
}  // namespace decord
