from transformers import LlamaTokenizerFast, RobertaTokenizerFast, BartTokenizerFast
import json
import os


def fn_load_tokenizer_chemllama(
    max_seq_length,
    dir_tokenizer: str = "./models_mtr/tokenizer.json",
    # dir_tokenizer:str = os.path.abspath(os.path.join(os.getcwd(), '..', "models_mtr/tokenizer.json")), # for JUP
    add_eos_token:bool = True,
):

    tokenizer = LlamaTokenizerFast(
        tokenizer_file=dir_tokenizer,
        model_max_length=max_seq_length,
        padding_side="right",
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
        add_eos_token=add_eos_token,
    )
    tokenizer.add_special_tokens({"pad_token": "<pad>", "sep_token": "</s>", "cls_token": "<s>", "mask_token":"<mask>"})
    # tokenizer.add_special_tokens({"pad_token": "<pad>"})

    return tokenizer
# chemllama size 591

def fn_load_tokenizer_chembart(
    max_seq_length,
    dir_tokenizer: str = "./models_mtr/tokenizer.json",
    # dir_tokenizer:str = os.path.abspath(os.path.join(os.getcwd(), '..', "models_mtr/tokenizer.json")),
):

    tokenizer = BartTokenizerFast(
        tokenizer_file=dir_tokenizer,
        model_max_length=max_seq_length,
        padding_side="right",
        bos_token="<s>",
        eos_token="</s>",
        sep_token="</s>",
        cls_token="<s>",
        unk_token="<unk>",
        pad_token="<pad>",
        mask_token="<mask>",
    )
    # tokenizer.add_special_tokens({"pad_token": "[PAD]", "mask_token": "[MASK]"})

    return tokenizer
# chembart size 591

# But somehow, chemberta not working..
# ./aten/src/ATen/native/cuda/Indexing.cu:1146: indexSelectLargeIndex: block: [240,0,0], thread: [66,0,0] Assertion `srcIndex < srcSelectDimSize` failed.
# Embedding index error, but it does not makes sense
# Using Llama Tokenizer works fine.
# def fn_load_tokenizer_chemberta(
#     max_seq_length,
#     dir_tokenizer: str = "./models_mtr/tokenizer.json",
# ):

#     tokenizer = RobertaTokenizerFast(
#         tokenizer_file=dir_tokenizer,
#         model_max_length=max_seq_length,
#         padding_side="right",
#         bos_token="<s>",
#         eos_token="</s>",
#         sep_token="</s>",
#         cls_token="<s>",
#         unk_token="<unk>",
#         pad_token="<pad>",
#         mask_token="<mask>",
#     )
#     # tokenizer.add_special_tokens({"pad_token": "[PAD]", "mask_token": "[MASK]"})

#     return tokenizer

def fn_load_tokenizer_chemberta(
    max_seq_length,
    dir_tokenizer: str = "./models_mtr/tokenizer.json",
    add_eos_token: bool = True,
):

    tokenizer = RobertaTokenizerFast(
        tokenizer_file=dir_tokenizer,
        model_max_length=max_seq_length,
        padding_side="right",
        add_eos_token=add_eos_token,
        bos_token="<s>",
        eos_token="</s>",
        sep_token="</s>",
        cls_token="<s>",
        unk_token="<unk>",
        pad_token="<pad>",
        mask_token="<mask>",
    )
    # tokenizer.add_special_tokens({"pad_token": "[PAD]", "mask_token": "[MASK]"})
    # tokenizer.add_special_tokens({"pad_token": "<pad>"})
    # tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    return tokenizer


def fn_load_descriptor_list(
    key_descriptor_list,
    dir_descriptor_list,
):

    with open(dir_descriptor_list, "r") as js:
        list_descriptor = json.load(js)[key_descriptor_list]

    return list_descriptor
