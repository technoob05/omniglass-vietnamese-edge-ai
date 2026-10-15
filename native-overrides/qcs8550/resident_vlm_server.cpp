// Minimal resident HTTP service for MiniCPM-V on the QCS8550 lane.
// It deliberately uses the same mtmd/llama APIs as llama-mtmd-cli, but keeps
// the model, llama context, sampler and HTP vision context alive between turns.

#include "arg.h"
#include "common.h"
#include "ggml.h"
#include "llama.h"
#include "log.h"
#include "mtmd.h"
#include "mtmd-helper.h"
#include "sampling.h"
#include "httplib.h"
#include <nlohmann/json.hpp>

#include <cstdlib>
#include <mutex>
#include <string>
#include <vector>

using json = nlohmann::json;

struct resident_vlm {
    common_init_result_ptr init;
    mtmd::context_ptr vision;
    llama_model * model = nullptr;
    llama_context * lctx = nullptr;
    const llama_vocab * vocab = nullptr;
    common_sampler * sampler = nullptr;
    llama_batch batch{};
    int n_batch = 0;
    int n_predict = 32;
    std::mutex mutex;

    explicit resident_vlm(common_params & params) : init(common_init_from_params(params)) {
        model = init->model();
        lctx = init->context();
        if (!model || !lctx) {
            throw std::runtime_error("unable to load MiniCPM-V model/context");
        }
        vocab = llama_model_get_vocab(model);
        sampler = common_sampler_init(model, params.sampling);
        batch = llama_batch_init(1, 0, 1);
        n_batch = params.n_batch;
        n_predict = params.n_predict > 0 ? params.n_predict : 32;

        mtmd_context_params mp = mtmd_context_params_default();
        mp.use_gpu = params.mmproj_use_gpu;
        mp.print_timings = true;
        mp.n_threads = params.cpuparams.n_threads;
        mp.flash_attn_type = params.flash_attn_type;
        mp.warmup = params.warmup;
        mp.image_min_tokens = params.image_min_tokens;
        mp.image_max_tokens = params.image_max_tokens;
        vision.reset(mtmd_init_from_file(params.mmproj.path.c_str(), model, mp));
        if (!vision) {
            throw std::runtime_error("unable to load mmproj / HTP vision context");
        }
    }

    ~resident_vlm() {
        if (sampler) common_sampler_free(sampler);
        llama_batch_free(batch);
    }

    std::string infer(const std::string & image_path, const std::string & prompt) {
        std::lock_guard<std::mutex> guard(mutex);
        llama_memory_clear(llama_get_memory(lctx), true);
        common_sampler_reset(sampler);

        mtmd::bitmap image(mtmd_helper_bitmap_init_from_file(vision.get(), image_path.c_str()));
        if (!image.ptr) throw std::runtime_error("cannot read image on box");
        const mtmd_bitmap * images[] = {image.ptr.get()};

        const std::string text_prompt = std::string(mtmd_default_marker()) + prompt;
        mtmd_input_text text{ text_prompt.c_str(), true, true };
        mtmd::input_chunks chunks(mtmd_input_chunks_init());
        if (mtmd_tokenize(vision.get(), chunks.ptr.get(), &text, images, 1) != 0) {
            throw std::runtime_error("mtmd_tokenize failed");
        }
        llama_pos n_past = 0;
        if (mtmd_helper_eval_chunks(vision.get(), lctx, chunks.ptr.get(), 0, 0, n_batch, true, &n_past)) {
            throw std::runtime_error("mtmd prompt evaluation failed");
        }

        llama_tokens output;
        for (int i = 0; i < n_predict; ++i) {
            const llama_token token = common_sampler_sample(sampler, lctx, -1);
            common_sampler_accept(sampler, token, true);
            if (llama_vocab_is_eog(vocab, token)) break;
            output.push_back(token);
            common_batch_clear(batch);
            common_batch_add(batch, token, n_past++, {0}, true);
            if (llama_decode(lctx, batch)) throw std::runtime_error("llama_decode failed");
        }
        return common_detokenize(lctx, output);
    }
};

static int port_from_env() {
    const char * value = std::getenv("RESIDENT_VLM_PORT");
    return value ? std::max(1, std::atoi(value)) : 18191;
}

int main(int argc, char ** argv) {
    common_params params;
    common_init();
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_MTMD)) return 1;
    llama_backend_init();
    try {
        resident_vlm runner(params);
        httplib::Server server;
        server.Get("/health", [&](const httplib::Request &, httplib::Response & res) {
            res.set_content(json{{"status", "ok"}, {"engine", "resident-mtmd"}}.dump(), "application/json");
        });
        server.Post("/v1/vision", [&](const httplib::Request & req, httplib::Response & res) {
            try {
                const auto in = json::parse(req.body);
                const auto image = in.value("image_path", "");
                const auto prompt = in.value("prompt", "Describe this image concisely.");
                if (image.empty()) throw std::runtime_error("image_path is required");
                res.set_content(json{{"answer", runner.infer(image, prompt)}}.dump(), "application/json");
            } catch (const std::exception & e) {
                res.status = 400;
                res.set_content(json{{"error", e.what()}}.dump(), "application/json");
            }
        });
        LOG_INF("resident MiniCPM-V service listening on 127.0.0.1:%d\n", port_from_env());
        server.listen("127.0.0.1", port_from_env());
    } catch (const std::exception & e) {
        LOG_ERR("resident VLM startup failed: %s\n", e.what());
        llama_backend_free();
        return 1;
    }
    llama_backend_free();
    return 0;
}
