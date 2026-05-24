"""ChatGLM adapter for optical-network QA."""

import torch
from common.registry import registry
from models.translator_models.translator_chatglm_arxiv import TranslatorCHATGLMArxiv

IMAGE_TOKEN_ID = 101


@registry.register_model("translator_optical_chatglm")
class TranslatorCHATGLMOptical(TranslatorCHATGLMArxiv):
    PRETRAINED_MODEL_CONFIG_DICT = {
        "pretrain_optical": "train/pretrain_optical_stage2.yaml",
        "generate_optical": "train/generate_optical.yaml",
    }

    def forward(self, samples):
        multimodal_embeds = samples[1].unsqueeze(dim=1).to(self.device)
        answers = samples[2]
        questions = samples[3]
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
            text_input=list(questions),
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
        max_length=256,
        **kwargs,
    ):
        device = self.Qformer.bert.device
        multimodal_embeds = samples[1].unsqueeze(dim=1).to(device)
        questions = list(samples[3])

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
                text_input=questions,
                answer=None,
            )

            outputs = self.chatglm2_model.generate(
                input_ids,
                inputs_embeds=inputs_embeds,
                max_length=max_length,
            )

        response_output = []
        for i in range(multimodal_embeds.shape[0]):
            outputs_i = outputs.tolist()[i][len(input_ids[i]) :]
            response = self.chatglm2_tokenizer.decode(outputs_i)
            response_output.append(self.chatglm2_model.process_response(response))
        return response_output
