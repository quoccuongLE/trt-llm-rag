import argparse
import json
import os
import sys
from pathlib import Path

import requests

# isort: off
import torch
import tensorrt as trt
# isort: on

from huggingface_hub import hf_hub_download
from PIL import Image
from transformers import (AutoConfig, AutoProcessor, AutoTokenizer,
                          Blip2Processor, NougatProcessor, NougatTokenizerFast)

import tensorrt_llm
import tensorrt_llm.profiler as profiler
from tensorrt_llm import logger
from tensorrt_llm.runtime import ModelRunner, Session, TensorInfo

sys.path.append(str(Path(__file__).parent.parent))
from enc_dec.run import TRTLLMEncDecModel


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_new_tokens', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--log_level', type=str, default='info')
    parser.add_argument('--visual_engine_dir',
                        type=str,
                        default=None,
                        help='Directory containing visual TRT engines')
    parser.add_argument('--llm_engine_dir',
                        type=str,
                        default=None,
                        help='Directory containing TRT-LLM engines')
    parser.add_argument('--hf_model_dir',
                        type=str,
                        default=None,
                        help="Directory containing tokenizer")
    parser.add_argument('--input_text',
                        type=str,
                        default=None,
                        help='Text prompt to LLM')
    parser.add_argument('--num_beams',
                        type=int,
                        help="Use beam search if num_beams >1",
                        default=1)
    parser.add_argument('--top_k', type=int, default=1)
    parser.add_argument('--run_profiling',
                        action='store_true',
                        help='Profile runtime over several iterations')
    parser.add_argument('--check_accuracy',
                        action='store_true',
                        help='Check correctness of text output')

    return parser.parse_args()


def trt_dtype_to_torch(dtype):
    if dtype == trt.float16:
        return torch.float16
    elif dtype == trt.float32:
        return torch.float32
    elif dtype == trt.int32:
        return torch.int32
    else:
        raise TypeError("%s is not supported" % dtype)


class MultimodalModelRunner:

    def __init__(self, args):
        self.args = args

        self.runtime_rank = tensorrt_llm.mpi_rank()
        device_id = self.runtime_rank % torch.cuda.device_count()
        torch.cuda.set_device(device_id)
        self.device = "cuda:%d" % (device_id)

        self.stream = torch.cuda.Stream(torch.cuda.current_device())
        torch.cuda.set_stream(self.stream)

        # parse model type from visual engine config
        with open(os.path.join(self.args.visual_engine_dir, "config.json"),
                  "r") as f:
            config = json.load(f)
        self.model_type = config['builder_config']['model_type']
        self.decoder_llm = not (
            't5' in self.model_type or 'nougat' in self.model_type
        )  # BLIP2-T5 and Nougat are using encoder-decoder models as LLMs

        self.profiling_iterations = 20

        self.init_image_encoder()
        self.init_tokenizer()
        self.init_llm()

    def init_tokenizer(self):
        if self.model_type == 'nougat':
            self.tokenizer = NougatTokenizerFast.from_pretrained(
                self.args.hf_model_dir)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.args.hf_model_dir, use_fast=False, use_legacy=False)

        self.tokenizer.padding_side = "right"

    def init_image_encoder(self):
        vision_encoder_path = os.path.join(self.args.visual_engine_dir,
                                           'visual_encoder.engine')
        logger.info(f'Loading engine from {vision_encoder_path}')
        with open(vision_encoder_path, 'rb') as f:
            engine_buffer = f.read()
        logger.info(f'Creating session from engine {vision_encoder_path}')
        self.visual_encoder_session = Session.from_serialized_engine(
            engine_buffer)

    def init_llm(self):
        if self.decoder_llm:
            self.model = ModelRunner.from_dir(self.args.llm_engine_dir,
                                              rank=tensorrt_llm.mpi_rank(),
                                              debug_mode=False,
                                              stream=self.stream)
            self.model_config = self.model.session._model_config
            self.runtime_mapping = self.model.session.mapping
        else:
            self.model = TRTLLMEncDecModel.from_engine(
                os.path.basename(self.args.hf_model_dir),
                self.args.llm_engine_dir,
                skip_encoder=(self.model_type == 'nougat'),
                debug_mode=False,
                stream=self.stream)

            if self.model_type == 'nougat':
                self.model_config = self.model.decoder_model_config
                self.runtime_mapping = self.model.decoder_runtime_mapping
            else:
                self.model_config = self.model.encoder_model_config
                self.runtime_mapping = self.model.encoder_runtime_mapping

    def generate(self, pre_prompt, post_prompt, image, decoder_input_ids,
                 max_new_tokens, warmup):
        if not warmup:
            profiler.start("Generate")
            profiler.start("Vision")
        visual_features, visual_atts = self.get_visual_features(image)
        if not warmup:
            profiler.stop("Vision")

        pre_input_ids = self.tokenizer(pre_prompt,
                                       return_tensors="pt",
                                       padding=True).input_ids
        if post_prompt[0] is not None:
            post_input_ids = self.tokenizer(post_prompt,
                                            return_tensors="pt",
                                            padding=True).input_ids
            length = pre_input_ids.shape[1] + post_input_ids.shape[
                1] + visual_atts.shape[1]
        else:
            post_input_ids = None
            length = pre_input_ids.shape[1] + visual_atts.shape[1]

        input_lengths = torch.IntTensor([length] * args.batch_size).to(
            torch.int32)
        input_ids, ptuning_args = self.setup_fake_prompts(
            visual_features, pre_input_ids, post_input_ids, input_lengths)

        if warmup: return None

        profiler.start("LLM")
        if self.decoder_llm:
            end_id = self.tokenizer.eos_token_id
            if 'opt' in self.model_type and 'blip2' in self.model_type:
                # For BLIP2-OPT, model outputs a "\n" at the end.
                # we avoid it by using newline as the end token
                end_id = self.tokenizer.encode("\n",
                                               add_special_tokens=False)[0]

            ptuning_args[0] = torch.stack([ptuning_args[0]])
            output_ids = self.model.generate(input_ids,
                                             sampling_config=None,
                                             prompt_table=ptuning_args[0],
                                             max_new_tokens=max_new_tokens,
                                             end_id=end_id,
                                             pad_id=self.tokenizer.pad_token_id,
                                             top_k=self.args.top_k,
                                             num_beams=self.args.num_beams,
                                             output_sequence_lengths=False,
                                             return_dict=False)
        else:
            if self.model_type == 'nougat':
                # Trim encoder input_ids to match visual features shape
                ids_shape = (self.args.batch_size, visual_features.shape[1])
                input_ids = torch.zeros(ids_shape, dtype=torch.int32)

            output_ids = self.model.generate(
                input_ids,
                decoder_input_ids,
                max_new_tokens,
                num_beams=self.args.num_beams,
                bos_token_id=self.tokenizer.bos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                debug_mode=False,
                prompt_embedding_table=ptuning_args[0],
                prompt_tasks=ptuning_args[1],
                prompt_vocab_size=ptuning_args[2])

            # Reset input_lengths to match decoder_input_ids
            input_lengths = torch.ones(input_lengths.shape,
                                       dtype=input_lengths.dtype)
        profiler.stop("LLM")

        if tensorrt_llm.mpi_rank() == 0:
            # Extract a list of tensors of shape beam_width x output_ids.
            output_beams_list = [
                self.tokenizer.batch_decode(
                    output_ids[batch_idx, :, input_lengths[batch_idx]:],
                    skip_special_tokens=True)
                for batch_idx in range(self.args.batch_size)
            ]

            stripped_text = [[
                output_beams_list[batch_idx][beam_idx].strip()
                for beam_idx in range(self.args.num_beams)
            ] for batch_idx in range(self.args.batch_size)]
            profiler.stop("Generate")
            return stripped_text
        else:
            profiler.stop("Generate")
            return None

    def get_visual_features(self, image):
        visual_features = {'input': image.half()}
        visual_output_info = self.visual_encoder_session.infer_shapes(
            [TensorInfo('input', trt.DataType.HALF, image.shape)])
        visual_outputs = {
            t.name: torch.empty(tuple(t.shape),
                                dtype=trt_dtype_to_torch(t.dtype),
                                device=image.device)
            for t in visual_output_info
        }

        ok = self.visual_encoder_session.run(visual_features, visual_outputs,
                                             self.stream.cuda_stream)
        assert ok, "Runtime execution failed for vision encoder session"
        self.stream.synchronize()

        image_embeds = visual_outputs['output']
        image_atts = torch.ones(image_embeds.size()[:-1],
                                dtype=torch.long).to(image.device)

        return image_embeds, image_atts

    def setup_fake_prompts(self, visual_features, pre_input_ids, post_input_ids,
                           input_lengths):
        # Assemble fake prompts which points to image embedding actually
        fake_prompt_id = torch.arange(
            self.model_config.vocab_size, self.model_config.vocab_size +
            visual_features.shape[0] * visual_features.shape[1])
        fake_prompt_id = fake_prompt_id.reshape(visual_features.shape[0],
                                                visual_features.shape[1])

        if post_input_ids is not None:
            input_ids = [pre_input_ids, fake_prompt_id, post_input_ids]
        else:
            input_ids = [fake_prompt_id, pre_input_ids]
        input_ids = torch.cat(input_ids, dim=1).contiguous().to(torch.int32)

        if self.decoder_llm or self.runtime_mapping.is_first_pp_rank():
            ptuning_args = self.ptuning_setup(visual_features, input_ids,
                                              input_lengths)
        else:
            ptuning_args = [None, None, None]

        return input_ids, ptuning_args

    def ptuning_setup(self, prompt_table, input_ids, input_lengths):
        hidden_size = self.model_config.hidden_size * self.runtime_mapping.tp_size
        if prompt_table is not None:
            task_vocab_size = torch.tensor(
                [prompt_table.shape[1]],
                dtype=torch.int32,
            ).cuda()
            prompt_table = prompt_table.view(
                (prompt_table.shape[0] * prompt_table.shape[1],
                 prompt_table.shape[2]))

            assert prompt_table.shape[
                1] == hidden_size, "Prompt table dimensions do not match hidden size"

            prompt_table = prompt_table.cuda().to(
                dtype=tensorrt_llm._utils.str_dtype_to_torch(
                    self.model_config.dtype))
        else:
            prompt_table = torch.empty([1, hidden_size]).cuda()
            task_vocab_size = torch.zeros([1]).cuda()

        if self.model_config.remove_input_padding:
            tasks = torch.zeros([torch.sum(input_lengths)],
                                dtype=torch.int32).cuda()
            if self.decoder_llm: tasks = tasks.unsqueeze(0)
        else:
            tasks = torch.zeros(input_ids.shape, dtype=torch.int32).cuda()

        return [prompt_table, tasks, task_vocab_size]

    def load_test_image(self):
        if "vila" in self.model_type:
            img_url = 'https://github.com/Efficient-Large-Model/VILA/raw/main/demo_images/av.png'
            image = Image.open(requests.get(img_url,
                                            stream=True).raw).convert('RGB')
        elif "nougat" in self.model_type:
            filepath = hf_hub_download(
                repo_id="hf-internal-testing/fixtures_docvqa",
                filename="nougat_paper.png",
                repo_type="dataset")
            image = Image.open(filepath)
        else:
            img_url = 'https://storage.googleapis.com/sfr-vision-language-research/LAVIS/assets/merlion.png'
            image = Image.open(requests.get(img_url,
                                            stream=True).raw).convert('RGB')

        return image

    def setup_inputs(self, input_text, raw_image):
        if 'blip2' in self.model_type:
            processor = Blip2Processor.from_pretrained(self.model_type)
            image = processor(raw_image, input_text,
                              return_tensors="pt")['pixel_values']

            if input_text is None:
                input_text = "Question: which city is this? Answer:"

            pre_prompt = input_text
            post_prompt = None
        elif 'nougat' in self.model_type:
            processor = NougatProcessor.from_pretrained(self.args.hf_model_dir)
            image = processor(raw_image, return_tensors="pt")['pixel_values']

            # Nougat doesn't need text prompt (mBART use single token to start generation), just leave a dummy one here
            if input_text is None:
                input_text = "Question: which city is this? Answer:"

            pre_prompt = input_text
            post_prompt = None
        elif 'llava' in self.model_type or 'vila' in self.model_type:
            # LLaVA and VILA
            if self.model_type == "llava":
                pre_prompt = "USER:\n"
                if input_text is None:
                    input_text = "Question: which city is this? Answer:"
            elif self.model_type == "vila":
                pre_prompt = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: "
                if input_text is None:
                    input_text = "Please describe the traffic condition."
            post_prompt = input_text + " ASSISTANT:"

            if self.model_type == "vila":
                sys.path.append(self.args.hf_model_dir + "/../VILA")
                from llava.model import LlavaLlamaForCausalLM
                model = LlavaLlamaForCausalLM.from_pretrained(
                    self.args.hf_model_dir, torch_dtype=torch.float16)
                vision_tower = model.get_vision_tower()
                image_processor = vision_tower.image_processor
                image = image_processor(images=raw_image,
                                        return_tensors="pt")['pixel_values']
            else:
                processor = AutoProcessor.from_pretrained(
                    self.args.hf_model_dir)
                image = processor(text=input_text,
                                  images=raw_image,
                                  return_tensors="pt")['pixel_values']

        # Repeat inputs to match batch size
        pre_prompt = [pre_prompt] * self.args.batch_size
        post_prompt = [post_prompt] * self.args.batch_size
        image = image.expand(self.args.batch_size, -1, -1, -1).contiguous()
        image = image.to(self.device)

        # Generate decoder_input_ids for enc-dec models
        # Custom prompts can be added as:
        # decoder_input_ids = model.tokenizer(decoder_prompt).input_ids
        if self.decoder_llm:
            decoder_input_ids = None
        else:
            config = AutoConfig.from_pretrained(args.hf_model_dir)
            decoder_start_id = config.decoder_start_token_id  # T5
            if decoder_start_id is None:
                decoder_start_id = config.decoder.bos_token_id  # Nougat

            decoder_input_ids = torch.IntTensor([[decoder_start_id]])
            decoder_input_ids = decoder_input_ids.repeat((args.batch_size, 1))

        return input_text, pre_prompt, post_prompt, image, decoder_input_ids

    def run(self, input_text, input_image, max_new_tokens):
        input_text, pre_prompt, post_prompt, processed_image, decoder_input_ids = model.setup_inputs(
            input_text, raw_image)

        model.generate(pre_prompt,
                       post_prompt,
                       processed_image,
                       decoder_input_ids,
                       max_new_tokens,
                       warmup=True)
        num_iters = self.profiling_iterations if self.args.run_profiling else 1
        for _ in range(num_iters):
            output_text = model.generate(pre_prompt,
                                         post_prompt,
                                         processed_image,
                                         decoder_input_ids,
                                         max_new_tokens,
                                         warmup=False)
        if self.runtime_rank == 0:
            self.print_result(input_text, output_text)
        return output_text

    def print_result(self, input_text, output_text):
        logger.info("---------------------------------------------------------")
        if self.model_type != 'nougat':
            logger.info(f"\n[Q] {input_text}")
        logger.info(f"\n[A] {output_text[0]}")

        if args.num_beams == 1:
            output_ids = self.tokenizer(output_text[0][0],
                                        add_special_tokens=False)['input_ids']
            logger.info(f"Generated {len(output_ids)} tokens")

        if self.args.check_accuracy:
            for i in range(self.args.batch_size - 1):
                if not (output_text[i] == output_text[i + 1]):
                    logger.info(f"Output {i} and {i + 1} do not match")
                    assert False
            if self.model_type != 'nougat':
                if self.model_type == "vila":
                    assert output_text[0][0].lower(
                    ) == 'the traffic condition in the image is quite busy, with multiple cars and bicycles sharing the road. there are also pedestrians walking on'
                else:
                    assert output_text[0][0].lower() == 'singapore'

        if self.args.run_profiling:
            msec_per_batch = lambda name: 1000 * profiler.elapsed_time_in_sec(
                name) / self.profiling_iterations
            logger.info('Latencies per batch (msec)')
            logger.info('TRT vision encoder: %.1f' % (msec_per_batch('Vision')))
            logger.info('TRTLLM LLM generate: %.1f' % (msec_per_batch('LLM')))
            logger.info('Multimodal generate: %.1f' %
                        (msec_per_batch('Generate')))

        logger.info("---------------------------------------------------------")


if __name__ == '__main__':
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    args = parse_arguments()
    tensorrt_llm.logger.set_level(args.log_level)

    model = MultimodalModelRunner(args)

    raw_image = model.load_test_image()
    text_output = model.run(args.input_text, raw_image, args.max_new_tokens)
