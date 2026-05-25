"""ChatGLM adapter for optical-network QA."""

import torch
from torch.nn.utils.rnn import pad_sequence
from common.registry import registry
from models.translator_models.translator_chatglm_arxiv import TranslatorCHATGLMArxiv

IMAGE_TOKEN_ID = 101
MAX_CONTEXT_CHARS = 420
QA_INSTRUCTION = (
    "你是光网络拓扑问答助手。请依据图表示和给定事实回答问题，"
    "只输出答案，不要编造无关内容。\n"
    "事实：{context}\n"
    "问题：{question}\n"
    "答案："
)


@registry.register_model("translator_optical_chatglm")
class TranslatorCHATGLMOptical(TranslatorCHATGLMArxiv):
    PRETRAINED_MODEL_CONFIG_DICT = {
        "pretrain_optical": "train/pretrain_optical_stage2.yaml",
        "pretrain_optical_qa_1k": "train/pretrain_optical_stage2_qa_1k.yaml",
        "generate_optical": "train/generate_optical.yaml",
        "generate_optical_qa_32": "train/generate_optical_qa_32.yaml",
    }

    def _build_qa_prompts(self, producer_texts, questions):
        prompts = []
        for producer_text, question in zip(producer_texts, questions):
            context = str(producer_text).replace("\n", " ")[:MAX_CONTEXT_CHARS]
            prompts.append(QA_INSTRUCTION.format(context=context, question=str(question)))
        return prompts

    def prepare_lm_input(self, vtokens, text_input, answer=None):
        bsz, nvtoken, _ = vtokens.size()
        tokenizer = self.chatglm2_tokenizer
        device = self.device

        sequences = []
        labels = []
        for idx, text in enumerate(text_input):
            prompt_ids = [IMAGE_TOKEN_ID] * nvtoken + tokenizer.encode(str(text), add_special_tokens=True)
            answer_ids = []
            if answer is not None:
                answer_ids = tokenizer.encode(str(answer[idx]), add_special_tokens=False)
                answer_ids = answer_ids[: max(0, self.max_txt_len - len(prompt_ids))]

            input_ids = torch.as_tensor(prompt_ids + answer_ids, dtype=torch.long)
            label = input_ids.detach().clone()
            label[: len(prompt_ids)] = -100
            sequences.append(input_ids)
            labels.append(label)

        input_ids = pad_sequence(sequences, batch_first=True, padding_value=tokenizer.pad_token_id).to(device)
        labels = pad_sequence(labels, batch_first=True, padding_value=-100).to(device)
        inputs_embeds = self.chatglm2_model.transformer.embedding.word_embeddings(input_ids)
        inputs_embeds[:, :nvtoken] = vtokens
        inputs_embeds = inputs_embeds.transpose(0, 1).contiguous()
        return input_ids, labels, inputs_embeds

    def forward(self, samples):
        multimodal_embeds = samples[1].unsqueeze(dim=1).to(self.device)
        producer_texts = samples[2]
        questions = samples[3]
        answers = samples[4]
        instructions = self._build_qa_prompts(producer_texts, questions)
        device = self.Qformer.bert.device

        multimodal_atts = torch.ones(multimodal_embeds.size()[:-1], dtype=torch.long).to(device)
        query_tokens = self.query_tokens.expand(multimodal_embeds.shape[0], -1, -1).to(device)

        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=multimodal_embeds,
            encoder_attention_mask=multimodal_atts,
            return_dict=True,
        )
        vtokens = self.chatglm2_proj(query_output.last_hidden_state[:, : query_tokens.size(1), :])

        input_ids, labels, inputs_embeds = self.prepare_lm_input(
            vtokens=vtokens,
            text_input=instructions,
            answer=list(answers),
        )

        outputs = self.chatglm2_model(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            return_dict=True,
            labels=labels,
        )

        return {"loss": outputs.loss, "vtokens": vtokens, "logits": outputs.logits}

    @torch.no_grad()
    def generate(
        self,
        samples,
        prompts=None,
        max_length=1024,
        **kwargs,
    ):
        device = self.Qformer.bert.device
        multimodal_embeds = samples[1].unsqueeze(dim=1).to(device)
        producer_texts = samples[2]
        questions = samples[3]
        instructions = self._build_qa_prompts(producer_texts, questions)

        with self.maybe_autocast():
            multimodal_atts = torch.ones(multimodal_embeds.size()[:-1], dtype=torch.long).to(device)
            query_tokens = self.query_tokens.expand(multimodal_embeds.shape[0], -1, -1).to(device)
            query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=multimodal_embeds,
                encoder_attention_mask=multimodal_atts,
                return_dict=True,
            )
            vtokens = self.chatglm2_proj(query_output.last_hidden_state[:, : query_tokens.size(1), :])

            input_ids, _, inputs_embeds = self.prepare_lm_input(
                vtokens=vtokens,
                text_input=instructions,
                answer=None,
            )

            generation_max_length = min(max_length, input_ids.shape[1] + 96)
            outputs = self.chatglm2_model.generate(
                input_ids,
                inputs_embeds=inputs_embeds,
                max_length=generation_max_length,
                do_sample=False,
            )

        response_output = []
        for i in range(multimodal_embeds.shape[0]):
            generated_ids = outputs.tolist()[i]
            if len(generated_ids) > len(input_ids[i]):
                outputs_i = generated_ids[len(input_ids[i]) :]
            else:
                outputs_i = generated_ids
            response = self.chatglm2_tokenizer.decode(outputs_i)
            response_output.append(self.chatglm2_model.process_response(response))
        return response_output
