/*!
 *  Copyright (c) 2019 by Contributors if not otherwise specified
 * \file video_decoder_interface.h
 * \brief Video Decoder Interface
 */

#ifndef DECORD_VIDEO_THREADED_DECODER_INTERFACE_H_
#define DECORD_VIDEO_THREADED_DECODER_INTERFACE_H_

#include "ffmpeg/ffmpeg_common.h"
#include <vector>
#include <decord/runtime/ndarray.h>

namespace decord {
typedef enum {
    DECORD_SKIP_FRAME   = 0x01,   /**< Set when the frame is not wanted, we can skip image processing  */
} ThreadedDecoderFlags;

class ThreadedDecoderInterface {
    public:
        virtual void SetCodecContext(AVCodecContext *dec_ctx, int width = -1, int height = -1,
                                     int rotation = 0, bool use_rrc = false,
                                     int orig_width = 224, int orig_height = 224,
                                     double scale_min = 0.08, double scale_max = 1.,
                                     double ratio_min = 0.75, double ratio_max = 4./3,
                                     bool use_msc = false,
                                     bool use_rcc = false,
                                     bool use_centercrop = false,
                                     bool use_fixedcrop = false, int crop_x = 0, int crop_y = 0,
                                     double hflip_prob = 0., double vflip_prob = 0.) = 0;
        virtual void Start() = 0;
        virtual void Stop() = 0;
        virtual void Clear() = 0;
        virtual void Push(ffmpeg::AVPacketPtr pkt, runtime::NDArray buf) = 0;
        virtual bool Pop(runtime::NDArray *frame) = 0;
        virtual void SuggestDiscardPTS(std::vector<int64_t> dts) = 0;
        virtual void ClearDiscardPTS() = 0;
        virtual ~ThreadedDecoderInterface() = default;
};  // class ThreadedDecoderInterface

}  // namespace decord
#endif  // DECORD_VIDEO_THREADED_DECODER_INTERFACE_H_
