import argparse
import sys
from utils import *
from scipy.stats import spearmanr
import os

def main():

    args = parse_arguments()

    if args.log == "True":
        if not os.path.exists(args.log_dir):
            os.makedirs(args.log_dir)
        log_file = open(args.log_file, "w")
        sys.stdout = log_file

    print('*****************************')
    print(args)
    print('*****************************')

    fix_seed(args.random_seed)

    # Initialize decoder class (load model and tokenizer) ...
    decoder = Decoder(args)

    print("setup data loader ...")
    print('*****************************')
    eval_dataloader, test_dataloader= setup_data_loader(
        args)
    print_now()
    print('*************************')

    if args.test_dataset_size == 0:
        args.test_dataset_size = len(test_dataloader)

    # test all the initial cot triggers and store their SC
    cot_triggers = ["Let's think step by step."] # you can initialize with your own cot triggers
    accuracies = [] # average accuracy
    SCs = [] # average self-consistency
    SC_lists = [] # self-consistency list on each question
    answer_lists = [] # answer list on each question

    print("start testing all the initial cot triggers ...")
    print('*************************')

    for cot_trigger in cot_triggers:
        SC, accuracy, SC_list, answer_list = predict(cot_trigger, args, decoder, eval_dataloader,
                                                                     args.eval_temperature, args.eval_dataset_size, args.eval_generate_times)
        SCs.append(SC)
        accuracies.append(accuracy)
        SC_lists.append(SC_list)
        answer_lists.append(answer_list)

    # update the cot trigger score
    Scores = optimize(args, answer_lists, SC_lists, 0.5)
    cot_trigger_scores = (torch.sum(Scores, axis=1) /
                        args.eval_dataset_size).detach().tolist()

    print("cot triggers are {}".format(cot_triggers))
    print("accuracies are {}".format(accuracies))
    print("SCs are {}".format(SCs))
    print("scores are {}".format(cot_trigger_scores))
    print("SC lists are {}".format(SC_lists))
    print("answer lists are {}".format(answer_lists))
    print(
        f"the spearman correlation between the cot trigger score and the accuracy is {spearmanr(cot_trigger_scores, accuracies)[0]:.2f}")
    print(
        f"the spearman correlation between the cot trigger SC and the accuracy is {spearmanr(SCs, accuracies)[0]:.2f}")
    print('*************************')

    # generate new cot trigger
    print("start generating new cot triggers ...")
    print('*************************')

    for iter1 in range(args.cot_generate_times):
        if args.evaluation_metric == "accuracy":
            eval_metrics = accuracies
        elif args.evaluation_metric == "SC":
            eval_metrics = SCs
        elif args.evaluation_metric == "Ours":
            eval_metrics = cot_trigger_scores
        else:
            raise ValueError("evaluation metric is not properly defined ...")
        
        sorted_data = sorted(
            zip(eval_metrics, cot_triggers), key=lambda x: x[0])
        sorted_texts = [cot_trigger for _, cot_trigger in sorted_data]
        sorted_scores = [eval_metric for eval_metric, _ in sorted_data]

        sorted_history = "\n".join([f"text:\n{text}\nscore:\n{score:.4f}\n" for text, score in zip(
            sorted_texts[-15:], sorted_scores[-15:])])

        pre_prompt = "I have some texts along with their corresponding scores. The texts are arranged in ascending order based on their scores, where higher scores indicate better quality."
        post_prompt = "Write one new text(one sentence) that is different from the old ones and has a score as high as possible. Write the text in square brackets."

        middle_prompt = "The following exemplars show how to apply your text: you replace <INS> in each input with your text, then read the input and give an output. We say your output is wrong if your output is different from the given output, and we say your output is correct if they are the same.\n\n"

        prompt = pre_prompt + "\n\n" + sorted_history + "\n" + \
            middle_prompt + args.example1 + args.example2 + args.example3 + post_prompt

        max_length = 256

        for iter2 in range(args.cot_generate_num):
            print("the prompt to generate new cot triggers is:\n{}".format(prompt))
            print('*************************')
            cot_generate_temperature = args.cot_generate_temperature
            output = decoder.decode(
                args, prompt, max_length, cot_generate_temperature)
            new_cot_trigger = extract_prompt_in_brackets(output)

            if new_cot_trigger is not None:
                print("the {}st new cot trigger is {}".format(
                    iter1 * args.cot_generate_num + iter2 + 1, new_cot_trigger))
                print('*************************')

                # skip if the new cot trigger is not valid (duplicated or too short)
                if not ("INS" in new_cot_trigger or new_cot_trigger in cot_triggers or len(new_cot_trigger) < 5):
                    SC, accuracy, SC_list, answer_list = predict(new_cot_trigger, args, decoder, eval_dataloader,
                                                                                 args.eval_temperature, args.eval_dataset_size, args.eval_generate_times)

                    cot_triggers.append(new_cot_trigger)
                    accuracies.append(accuracy)
                    SCs.append(SC)
                    SC_lists.append(SC_list)
                    answer_lists.append(answer_list)

                    # update the cot trigger score
                    Scores = optimize(args, answer_lists, SC_lists, 0.5)
                    cot_trigger_scores = (
                        torch.sum(Scores, axis=1) / args.eval_dataset_size).detach().tolist()

                    print("cot triggers are {}".format(cot_triggers))
                    print("accuracies are {}".format(accuracies))
                    print("SCs are {}".format(SCs))
                    print("scores are {}".format(cot_trigger_scores))
                    print("SC lists are {}".format(SC_lists))
                    print("answer lists are {}".format(answer_lists))
                    print(
                        f"the spearman correlation between the cot trigger score and the accuracy is {spearmanr(cot_trigger_scores, accuracies)[0]:.2f}")
                    print(
                        f"the spearman correlation between the cot trigger SC and the accuracy is {spearmanr(SCs, accuracies)[0]:.2f}")
                    print('*************************')
                else:
                    print("the {}st new cot trigger is {}".format(
                        iter1 * args.cot_generate_num + iter2 + 1, new_cot_trigger))
                    print("the new cot trigger is not valid")
                    print('*************************')

    # test the cot trigger with highest score
    print("start testing the cot trigger with highest score ...")
    print('*************************')

    best_cot_trigger = cot_triggers[np.argmax(cot_trigger_scores)]
    print("the best cot trigger is {}".format(best_cot_trigger))
    print('*************************')

    SC, accuracy, SC_list, answer_list = predict(best_cot_trigger, args, decoder, test_dataloader,
                                                                 args.test_temperature, args.test_dataset_size, args.test_generate_times)

    if args.log == "True":
        sys.stdout = sys.__stdout__
        log_file.close()


def parse_arguments():
    parser = argparse.ArgumentParser(description="CoT-Prompt-Optimization")

    parser.add_argument("--random_seed", type=int,
                        default=42, help="random seed")

    parser.add_argument(
        "--dataset", type=str, default="gsm8k", choices=["aqua", "gsm8k", "addsub", "multiarith", "svamp", "bigbench_date"], help="dataset used for experiment"
    )

    parser.add_argument("--max_num_worker", type=int, default=3,
                        help="maximum number of workers for dataloader")
    
    parser.add_argument("--model", type=str, default="gpt-3.5-turbo", help="which model to use")

    parser.add_argument(
        "--max_length_cot", type=int, default=512, help="maximum length of output tokens by model for reasoning extraction"
    )
    parser.add_argument(
        "--max_length_direct", type=int, default=128, help="maximum length of output tokens by model for answer extraction"
    )
    parser.add_argument(
        "--eval_generate_times", type=int, default=10, help="the number of times to generate the cot for each question when evaluating the cot trigger"
    )
    parser.add_argument(
        "--eval_dataset_size", type=int, default=10, help="the number of questions to evaluate the cot trigger, 0 means test all the questions"
    )
    parser.add_argument(
        "--test_dataset_size", type=int, default=0, help="the number of questions to test the cot trigger, 0 means test all the questions"
    )
    parser.add_argument(
        "--test_generate_times", type=int, default=10, help="the number of times to generate the cot for each question when testing the cot trigger"
    )
    parser.add_argument(
        "--eval_temperature", type=float, default=0.5, help="temperature for decoding when evaluating the cot trigger"
    )
    parser.add_argument(
        "--test_temperature", type=float, default=0.5, help="temperature for decoding when testing the cot trigger"
    )
    parser.add_argument(
        "--cot_generate_temperature", type=float, default=0.7, help="temperature for decoding when generating new cot triggers"
    )
    parser.add_argument(
        "--cot_generate_times", type=int, default=2, help="the number of times to generate new cot triggers"
    )
    parser.add_argument(
        "--cot_generate_num", type=int, default=2, help="the number of new cot triggers to generate for each iteration"
    )
    parser.add_argument(
        "--log", type=str, default="True", help="whether to log the output"
    )
    parser.add_argument(
        "--log_dir", type=str, default="./log/", help="log directory"
    )
    parser.add_argument(
        "--log_file_name", type=str, default="log.txt", help="the name of the log file"
    )
    parser.add_argument(
        "--encoder", type=str, default="all-MiniLM-L6-v2", help="which sentence-transformer encoder for clustering"
    )
    parser.add_argument(
        "--num_clusters", type=int, default=10, help="the number of clusters"
    )
    parser.add_argument(
        "--question_num_per_cluster", type=int, default=10, help="the number of questions per cluster"
    )
    parser.add_argument(
        "--eval_dataset_generate_method", type=str, default="random", choices=["random", "cluster"], help="the method to generate test dataset"
    )
    parser.add_argument(
        "--eval_dataset_source", type=str, default="test", choices=["train", "test"], help="the source of evaluation dataset"
    )
    parser.add_argument(
        "--device", type=str, default="default", choices=["cuda", "cpu", "default"], help="the device to use"
    )
    parser.add_argument(
        "--evaluation_metric", type=str, default="Ours", choices=["accuracy", "SC", "Ours"], help="the evaluation metric"
    )

    args = parser.parse_args()

    if args.device == "default":
        args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if args.evaluation_metric == "Ours":
        args.log_file_name = "GLaPE_" + args.log_file_name
    else:
        args.log_file_name = args.evaluation_metric + "_" + args.log_file_name

    if args.eval_dataset_generate_method == "cluster":
        args.eval_dataset_size = args.num_clusters * args.question_num_per_cluster

    args.log_dir = args.log_dir + args.dataset + "/"
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    if args.eval_dataset_generate_method == "cluster":
        args.log_file_name = str(args.num_clusters) + "cls*" + str(args.question_num_per_cluster) + "q_" + args.log_file_name
    else :
        args.log_file_name = args.eval_dataset_generate_method + str(args.eval_dataset_size) + "_" + args.log_file_name

    args.log_file = args.log_dir + args.log_file_name

    if args.dataset == "aqua":
        args.train_dataset_path = "./dataset/AQuA/aqua.json"
        args.test_dataset_path = "./dataset/AQuA/aqua.json"
        args.direct_answer_trigger = "\nTherefore, among A through E, the answer is"
        args.example1 = "input:\nQ: A rectangle has a length of 8 centimeters and a width of 3 centimeters. Find the perimeter. Answer Choices: (A) 18cm (B) 22cm (C) 20cm (D) 30cm (E) 28cm\nA:<INS>\noutput:\nB\n\n"
        args.example2 = "input:\nQ: I have a money pouch containing Rs. 700. There are equal number of 25 paise coins, 50 paise coins and one rupee coins.How many of each are there? Answer Choices: (A) 453 (B) 651 (C) 400 (D) 487 (E) 286\nA:<INS>\noutput:\nC\n\n"
        args.example3 = "input:\nQ: Find out which of the following values is the multiple of X, if it is divisible by 9 and 12? Answer Choices: (A) 36 (B) 15 (C) 17 (D) 5 (E) 7\nA:<INS>\noutput:\nA\n\n"
    elif args.dataset == "gsm8k":
        args.train_dataset_path = "./dataset/grade-school-math/train.jsonl"
        args.test_dataset_path = "./dataset/grade-school-math/test.jsonl"
        args.direct_answer_trigger = "\nTherefore, the answer (arabic numerals) is"
        args.example1 = "input:\nQ: James decides to run 3 sprints 3 times a week.  He runs 60 meters each sprint.  How many total meters does he run a week?\nA: <INS>\noutput:\n540\n\n"
        args.example2 = "input:\nQ: Toulouse has twice as many sheep as Charleston. Charleston has 4 times as many sheep as Seattle. How many sheep do Toulouse, Charleston, and Seattle have together if Seattle has 20 sheep?\nA: <INS>\noutput:\n260\n\n"
        args.example3 = "input:\nQ: Josh decides to try flipping a house.  He buys a house for $80,000 and then puts in $50,000 in repairs.  This increased the value of the house by 150%.  How much profit did he make?\nA: <INS>\noutput:\n70000\n\n"
    elif args.dataset == "addsub":
        args.train_dataset_path = "./dataset/AddSub/AddSub.json"
        args.test_dataset_path = "./dataset/AddSub/AddSub.json"
        args.direct_answer_trigger = "\nTherefore, the answer (arabic numerals) is"
        args.example1 = "input:\nQ: Joan grew 8 watermelons and 4 turnips . Tom grew 9 watermelons . How many watermelons did they grow in total ?\nA: <INS>\noutput:\n17\n\n"
        args.example2 = "input:\nQ: Dan grew 42 turnips and 38 cantelopes . Jessica grew 47 turnips . How many turnips did they grow in total ?\nA: <INS>\noutput:\n89\n\n"
        args.example3 = "input:\nQ: Melanie grew 139 turnips . Benny grew 113 turnips . How many turnips did they grow in all ?\nA: <INS>\noutput:\n252\n\n"
    elif args.dataset == "multiarith":
        args.train_dataset_path = "./dataset/MultiArith/MultiArith.json"
        args.test_dataset_path = "./dataset/MultiArith/MultiArith.json"
        args.direct_answer_trigger = "\nTherefore, the answer (arabic numerals) is"
        args.example1 = "input:\nQ: For Halloween Megan received 11 pieces of candy from neighbors and 5 pieces from her older sister. If she only ate 8 pieces a day, how long would the candy last her?\nA: <INS>\noutput:\n2\n\n"
        args.example2 = "input:\nQ: Luke was trying to expand his game collection. He bought 2 games from a friend and bought 2 more at a garage sale. If 2 of the games didn't work, how many good games did he end up with?\nA: <INS>\noutput:\n2\n\n"
        args.example3 = "input:\nQ: Lana picked 36 tulips and 37 roses to make flower bouquets. If she only used 70 of the flowers though, how many extra flowers did Lana pick?\nA: <INS>\noutput:\n3\n\n"
    elif args.dataset == "svamp":
        args.train_dataset_path = "./dataset/SVAMP/SVAMP.json"
        args.test_dataset_path = "./dataset/SVAMP/SVAMP.json"
        args.direct_answer_trigger = "\nTherefore, the answer (arabic numerals) is"
        args.example1 = "input:\nQ: There were 13 roses in the vase. Jessica cut some more roses from her flower garden which had a total of 12 roses. There are now 21 roses in the vase. How many roses are left in the garden?\nA: <INS>\noutput:\n4\n\n"
        args.example2 = "input:\nQ: Allan brought 6 balloons and Jake brought 3 balloons to the park. Jake then bought 4 more balloons at the park. How many more balloons did Jake have than Allan in the park?\nA: <INS>\noutput:\n1\n\n"
        args.example3 = "input:\nQ: There were 15 roses and 62 orchids in the vase. Jessica cut some more roses and orchids from her flower garden. There are now 17 roses and 96 orchids in the vase. How many roses did she cut?\nA: <INS>\noutput:\n2\n\n"
    elif args.dataset == "bigbench_date":
        args.train_dataset_path = "./dataset/Bigbench_Date/task.json"
        args.test_dataset_path = "./dataset/Bigbench_Date/task.json"
        args.direct_answer_trigger = "\nTherefore, among A through F, the answer is"
        args.example1 = "input:\nQ: The deadline is Jun 1, 2021, which is 2 days away from now. What is the date 10 days ago in MM/DD/YYYY? Answer Choices: (A) 05/20/2021 (B) 04/29/2021 (C) 04/29/2021 (D) 04/06/2021 (E) 06/24/2021\nA: <INS>\noutput:\nA\n\n"
        args.example2 = "input:\nQ: Tomorrow is 11/12/2019. What is the date yesterday in MM/DD/YYYY? Answer Choices: (A) 11/10/2076 (B) 11/06/2019 (C) 11/10/2019 (D) 11/17/2019 (E) 09/10/2019 (F) 11/11/2019\nA: <INS>\noutput:\nC\n\n"
        args.example3 = "input:\nQ: It is 4/19/1969 today. What is the date one week from today in MM/DD/YYYY? Answer Choices: (A) 04/05/1969 (B) 05/11/1969 (C) 04/25/1969 (D) 05/14/1969 (E) 04/26/1969 (F) 04/27/1969\nA: <INS>\noutput:\nE\n\n"
    else:
        raise ValueError("dataset is not properly defined ...")

    # "Therefore, the answer ..." -> "The answer ..."
    args.direct_answer_trigger_for_zeroshot_cot = args.direct_answer_trigger.replace("\nTherefore, ", "")

    return args


if __name__ == "__main__":
    main()
