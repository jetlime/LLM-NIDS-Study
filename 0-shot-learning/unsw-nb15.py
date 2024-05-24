from datasets import load_dataset, Dataset
Dataset.cleanup_cache_files
from tqdm import tqdm
import transformers
import torch
torch.cuda.empty_cache()

dataset = load_dataset("Jetlime/NF-UNSW-NB15-v2", streaming=True, split="test")

classes = dataset.features["output"].names



# We choose the instruction version of Llama 3 as the foundational
# model showed difficulties to answer in the required format.
# This is an expected behavior as these models were not trained to
# understand instructions but simply to predict the sequence of words.
model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device="cuda",
    pad_token_id = 50256
)

def classification_pipeline(netflow):

    messages = [
        {"role": "instruction", "content": "You are a cybersecurity expert tasked with classifying network flows as either malicious or benign. If you determine the network flow is benign, respond with '0'. If you determine the network flow is malicious, respond with '1'. For example, if given the following network flow: 'IPV4_SRC_ADDR: 59.166.0.7, L4_SRC_PORT: 53030, IPV4_DST_ADDR: 149.171.126.7, L4_DST_PORT: 44287, PROTOCOL: 6, L7_PROTO: 0.0, IN_YTES: 8928, IN_TS: 14, OUT_TES: 320, OUT_S: 6, TCP_AGS: 27, CLIENT_CP_AGS: 27, SERVER_CP_LAGS: 19, FLOW_URATION_ILLISECONDS: 0, DURATION_N: 0, DURATION_UT: 0, MIN_L: 31, MAX_L: 32, LONGEST_OW_T: 1500, SHORTEST_OW_: 52, MIN__PKT_N: 52, MAX__T_: 1500, SRC__T_ECOND_YTES: 8928.0, DST_O_C_COND_TES: 320.0, RETRANSMITTED__TES: 4252, RETRANSMITTED__TS: 3, RETRANSMITTED_T_TES: 0, RETRANSMITTED_T_TS: 0, SRC__DST_VG_ROUGHPUT: 71424000, DST__RC_G_ROUGHPUT: 2560000, NUM_TS___28_TES: 14, NUM_TS_8__6_YTES: 0, NUM_KTS_6__2_TES: 0, NUM_TS_2__24_TES: 0, NUM_TS_24__14_TES: 6, TCP_WIN_MAX_N: 5792, TCP_N_X_T: 10136, ICMP_PE: 39936, ICMP_PV4_TYPE: 156, DNS_QUERY_ID: 0, DNS_QUERY_TYPE: 0, DNS_TTL_ANSWER: 0, FTP_COMMAND_RET_CODE: 0.0' and you assess it as benign, you would respond with '0'. If you assess it as malicious, you would respond with '1'. You are only allowed to respond with '0' or '1'. If requested, provide an explanation for your classification, detailing the reasoning and which feature values influenced your decision."},
        {"role": "input", "content": netflow},
    ]

    prompt = pipeline.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
    )

    terminators = [
        pipeline.tokenizer.eos_token_id,
        pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    outputs = pipeline(
        prompt,
        max_new_tokens=100,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.01,
        top_p=0.9,
        
    )

    return outputs[0]["generated_text"][len(prompt):]

classification_pipeline("IPV4_SRC_ADDR: 149.171.126.0, L4_SRC_PORT: 62073, IPV4_DST_ADDR: 59.166.0.5, L4_DST_PORT: 56082, PROTOCOL: 6, L7_PROTO: 0.0, IN_BYTES: 9672, OUT_BYTES: 416, IN_PKTS: 11, OUT_PKTS: 8, TCP_FLAGS: 25, FLOW_DURATION_MILLISECONDS: 15")

prediction_labels = []

for i in tqdm(dataset, total=120000):
    prediction = classification_pipeline(i['input'])
    prediction_labels.append(prediction)

true_labels = []
for i in tqdm(dataset, total=120000):
    true_labels.append(i["output"])

from sklearn.metrics import classification_report

prediction_labels = [int(item) for item in prediction_labels]

target_names = ['benign', 'malicious']

report = classification_report(true_labels, prediction_labels, digits=4, target_names=target_names)

print(report)

with open('classification_report.txt', 'w') as file:
    file.write(report)
