from statistics import mean
from torch.utils.data import Dataset
import openai
import multiprocessing
import json
import numpy as np
import random
import torch
import re
import random
import datetime
import time
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import os

# generate responses for the given prompt
def predict(cot_trigger, args, decoder, dataloader, temperature=0.5, dataset_size=10, generate_times=10):
    # print the hyperparameters
    print("the cot trigger is {}".format(cot_trigger))
    print("the LLM temperature is {}".format(temperature))
    print("the dataset size is {}".format(min(len(dataloader), dataset_size)))
    print('*************************')

    SC = 0
    accuracy = 0
    SC_list = []
    answer_list = []

    for i, data in enumerate(dataloader):
        print("{}st data".format(i+1))
        print('*************************')

        # Prepare question template ...
        x, y = data
        x = "Q: " + x[0] + "\n" + "A:"
        y = y[0].strip()
        preds = []

        # Answer prediction by generating text ...
        max_length = args.max_length_cot

        cot_prompt = x + " " + cot_trigger

        print("The question is: ")
        print(x)
        print('*************************')

        for iter in range(generate_times):
            cot = decoder.decode(
                args, cot_prompt, max_length, temperature)

            # print the cot
            print("cot : {}".format(cot))
            print('*************************')

            # Answer extraction for zero-shot-cot
            ans_prompt = x + cot + " " + args.direct_answer_trigger_for_zeroshot_cot
            max_length = args.max_length_direct
            pred = decoder.decode(
                args, ans_prompt, max_length, 0)

            # Cleansing of predicted answer ...
            pred = answer_cleansing(args, pred)

            # Choose the most frequent answer from the list ...
            print("pred : {}".format(pred))
            print("GT : " + y)
            print('*************************')

            # add the pred to the list
            preds.append(pred)

        # calculate the frequency of the most frequent answer
        print("preds is {}".format(preds))
        print("the most frequent answer is {}".format(
            max(set(preds), key=preds.count)))
        new_answer = max(set(preds), key=preds.count)
        new_SC = preds.count(new_answer) / generate_times * 100
        SC += new_SC
        SC_list.append(float(new_SC))
        answer_list.append(new_answer)

        # calculate the accuracy
        if y == max(set(preds), key=preds.count):
            accuracy += 100

        if ((i+1) >= dataset_size):
            break

    SC = SC / dataset_size
    accuracy = accuracy / dataset_size

    # print the results
    print("cot trigger is {}".format(cot_trigger))
    print("SC : {}".format(SC))
    print("accuracy : {}".format(accuracy))
    print("SC list : {}".format(SC_list))
    print("answer list : {}".format(answer_list))
    print('*************************')

    return SC, accuracy, SC_list, answer_list

def compute_loss(args, Answer, SC, Score, alpha):
    prompt_num, question_num = SC.shape

    self_loss = torch.sum((Score - SC) ** 2) / 2

    # compute the mask
    # mask[i][j][k] indicates whether the answer of prompt i and prompt j on question k is the same
    mask = torch.zeros(prompt_num, prompt_num, question_num).to(args.device)
    for k in range(question_num):
        for i in range(prompt_num-1):
            for j in range(i + 1, prompt_num):
                if Answer[i][k] == Answer[j][k]:
                    mask[i][j][k] = 1.0

    diff = (Score.permute(1, 0)[:, :, None] - Score.permute(1, 0)[:, None, :]).permute(1, 2, 0)
    refine_loss = torch.sum(mask * (diff ** 2)) 

    total_loss = alpha * self_loss + (1 - alpha) * refine_loss

    return total_loss


def optimize(args, Answer, SC, alpha, max_iter=1000000, lr=0.05):
    SC = torch.tensor(SC, dtype=torch.float32).to(args.device)

    prompt_num, question_num = SC.shape

    Score = torch.tensor(np.random.rand(prompt_num, question_num) * 100, dtype=torch.float32).to(args.device)
    Score.requires_grad = True

    optimizer = torch.optim.Adam([Score], lr=lr)
    last_loss = 0.0

    for iter in range(max_iter):
        loss = compute_loss(args, Answer, SC, Score, alpha)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if abs(last_loss - loss) < 1e-5:
            break

        last_loss = loss

    print('*************************')

    return Score


def shuffleDict(d):
    keys = list(d.keys())
    random.shuffle(keys)
    [(key, d[key]) for key in keys]
    random.shuffle(keys)
    [(key, d[key]) for key in keys]
    random.shuffle(keys)
    keys = [(key, d[key]) for key in keys]
    # keys = d(keys)
    return dict(keys)


def fix_seed(seed):
    # random
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def print_now(return_flag=0):
    t_delta = datetime.timedelta(hours=9)
    JST = datetime.timezone(t_delta, 'JST')
    now = datetime.datetime.now(JST)
    now = now.strftime('%Y/%m/%d %H:%M:%S')
    if return_flag == 0:
        print(now)
    elif return_flag == 1:
        return now
    else:
        pass

# Sentence Generator (Decoder) for GPT-3 ...


def decoder_for_gpt3(args, input, max_length, temperature):
    
    if args.model == "gpt-3.5-turbo":
        engine = "gpt-3.5-turbo"
    else:
        raise ValueError("model is not properly defined ...")
    # openai.api_key = "[Your OpenAI API Key]"

    messages = [{"role": "user", "content": input}]

    while True:
        try:
            response = openai.ChatCompletion.create(
                model=engine,
                messages=messages,
                temperature=temperature,
                max_tokens=max_length,
                top_p=1,  # Not recommended to change with temperature
                frequency_penalty=0,
                presence_penalty=0,
                stop=None
            )

            return response["choices"][0]["message"]["content"]
        except Exception as e:
            print('*************************')
            print("Exception is {}".format(e), flush=True)
            time.sleep(1)


class Decoder():
    def __init__(self, args):
        print_now()

    def decode(self, args, input, max_length, temperature):
        response = decoder_for_gpt3(
            args, input, max_length, temperature)
        return response


def data_loader(args, dataset_source):

    questions = []
    answers = []
    decoder = json.JSONDecoder()

    if dataset_source == "train":
        f = open(args.train_dataset_path, "r")
    elif dataset_source == "test":
        f = open(args.test_dataset_path, "r")

    if args.dataset == "aqua":
        lines = f.readlines()
        for line in lines:
            json_res = decoder.raw_decode(line)[0]
            choice = "(" + "(".join(json_res["options"])
            choice = choice.replace("(", " (").replace(")", ") ")
            choice = "Answer Choices:" + choice
            questions.append(json_res["question"].strip() + " " + choice)
            answers.append(json_res["correct"])

    elif args.dataset == "gsm8k":
        lines = f.readlines()
        for line in lines:
            json_res = decoder.raw_decode(line)[0]
            questions.append(json_res["question"].strip())
            answers.append(json_res["answer"].split("#### ")[-1])

    elif args.dataset in ("addsub", "multiarith"):
        json_data = json.load(f)
        for line in json_data:
            q = line["sQuestion"].strip()
            a = str(line["lSolutions"][0])
            if a[-2:] == ".0":
                a = a[:-2]
            questions.append(q)
            answers.append(a)

    elif args.dataset == "svamp":
        json_data = json.load(f)
        for line in json_data:
            q = line["Body"].strip() + " " + line["Question"].strip()
            a = str(line["Answer"])
            if a[-2:] == ".0":
                a = a[:-2]
            questions.append(q)
            answers.append(a)

    elif args.dataset == "bigbench_date":
        json_data = json.load(f)
        json_data = json_data["examples"]
        choice_index = ['A', 'B', 'C', 'D', 'E', 'F']
        for line in json_data:
            q = line["input"].strip()
            if args.dataset == "bigbench_date":
                choice = "Answer Choices:"
                # Randomly shuffle the answer choice dictionary because the original answer is always A ...
                choice_dic = shuffleDict(line["target_scores"])
            elif args.dataset == "object_tracking":
                choice = "\nWhich choice is true ? Answer Choices:"
                choice_dic = line["target_scores"]
            else:
                raise ValueError("dataset is not properly defined ...")
            for i, key_value in enumerate(choice_dic.items()):
                key, value = key_value
                choice += " ("
                choice += choice_index[i]
                choice += ") "
                choice += key
                if value == 1:
                    a = choice_index[i]
                    # a = key
            q = q + " " + choice
            questions.append(q)
            answers.append(a)

    else:
        raise ValueError("dataset is not properly defined ...")

    return questions, answers

# Create dataset object before dataloader ...


class MyDataset(Dataset):
    def __init__(self, args, dataset_source):
        super().__init__()
        self.questions, self.answers = data_loader(args, dataset_source)
        self.len = len(self.questions)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        input = self.questions[index]
        output = self.answers[index]
        return input, output

def setup_data_loader(args):

    # fix randomness of dataloader to ensure reproducibility
    # https://pytorch.org/docs/stable/notes/randomness.html
    fix_seed(args.random_seed)
    worker_seed = torch.initial_seed() % 2**32
    print("worker_seed : {}".format(worker_seed))

    def seed_worker(worker_id):
        np.random.seed(worker_seed)
        random.seed(worker_seed)
    g = torch.Generator()
    g.manual_seed(worker_seed)

    dataloader_num_workers = multiprocessing.cpu_count()
    dataloader_num_workers = min(dataloader_num_workers, args.max_num_worker)
    print("dataloader_num_workers: " + str(dataloader_num_workers))

    test_dataset = MyDataset(args, "test")

    eval_list = []
    eval_dataset = MyDataset(args, args.eval_dataset_source)

    if args.eval_dataset_generate_method == "cluster":
        # encode the eval dataset
        os.environ["TOKENIZERS_PARALLELISM"] = "true"
        encoder = SentenceTransformer(args.encoder)
        encoded_eval_dataset = encoder.encode(
            eval_dataset.questions, show_progress_bar=True, convert_to_numpy=True)

        # using k-means to cluster the eval dataset
        clustering_model = KMeans(n_clusters=args.num_clusters, random_state=42)
        clustering_model.fit(encoded_eval_dataset)
        cluster_assignment = clustering_model.labels_

        clusterer_idx = [[] for i in range(args.num_clusters)]
        for sentence_id, cluster_id in enumerate(cluster_assignment):
            clusterer_idx[cluster_id].append(sentence_id)

        for i in range(args.num_clusters):
            random.seed(args.random_seed)
            eval_list += random.sample(
                clusterer_idx[i], min(args.question_num_per_cluster, len(clusterer_idx[i])))
    elif args.eval_dataset_generate_method == "random":
        random.seed(args.random_seed)
        eval_list = random.sample(
            range(len(eval_dataset)), args.eval_dataset_size)
    
    if args.eval_dataset_size == 0:
        eval_dataset = test_dataset
    else:
        eval_dataset = torch.utils.data.Subset(test_dataset, eval_list)

    if args.test_dataset_size != 0:
        random.seed(args.random_seed)
        test_list = random.sample(
            range(len(test_dataset)), args.test_dataset_size)
        test_dataset = torch.utils.data.Subset(test_dataset, test_list)
        

    eval_dataloader = torch.utils.data.DataLoader(eval_dataset,
                                                   shuffle=False,  # because we will match the answer with the prediction later
                                                   drop_last=False,
                                                   num_workers=dataloader_num_workers,
                                                   worker_init_fn=seed_worker,
                                                   generator=g,
                                                   pin_memory=True)

    test_dataloader = torch.utils.data.DataLoader(test_dataset,
                                                  shuffle=False,
                                                  drop_last=False,
                                                  num_workers=dataloader_num_workers,
                                                  worker_init_fn=seed_worker,
                                                  generator=g,
                                                  pin_memory=True)
    
    print("dataset : {}".format(args.dataset))
    print("eval dataset size : {}".format(len(eval_dataloader)))
    print("test dataset size : {}".format(len(test_dataloader)))

    return eval_dataloader, test_dataloader

# ver 0.2


def answer_cleansing(args, pred):

    print("pred_before : " + pred, flush=True)

    if args.dataset == "aqua":
        pred = re.findall(r'A|B|C|D|E', pred)
    elif args.dataset == "bigbench_date":
        pred = re.findall(r'A|B|C|D|E|F', pred)
    elif args.dataset in ("gsm8k", "addsub", "multiarith", "svamp"):
        pred = pred.replace(",", "")
        if "=" in pred:
            pred = pred.split("=")[-1]
        pred = [s for s in re.findall(r'-?\d+\.?\d*', pred)]
    else:
        raise ValueError("dataset is not properly defined ...")

    # If there is no candidate in list, null is set.
    if len(pred) == 0:
        pred = ""
    else:
        # choose the first element in list ...
        pred = pred[0]

    # (For arithmetic tasks) if a word ends with period, it will be omitted ...
    if pred != "":
        # modify demicals
        if pred[-1] == ".":
            pred = pred[:-1]

        if "." in pred:
            integer_part, decimal_part = pred.split(".")
            if decimal_part == "0" * len(decimal_part):
                pred = integer_part

    print("pred_after : " + pred, flush=True)

    return pred


def extract_prompt_in_brackets(input_string):
    pattern = r'\[(.*?)\]'
    match = [s for s in re.findall(pattern, input_string)]
    if match:
        new_cot_trigger = match[-1]
        if '.' in new_cot_trigger:
            new_cot_trigger = new_cot_trigger.split('.')[0] + '.'
        return new_cot_trigger
    else:
        return None
