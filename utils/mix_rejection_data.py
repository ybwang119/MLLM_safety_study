import argparse
import json
import random
random.seed(0)
def read_json_to_list(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

def get_random_samples(data_list, sample_size, iexclude_indices=list(), similar=False):
    if similar:
        available_samples = [i for i in data_list if i['id'] in iexclude_indices] 
        selected_samples = random.sample(available_samples, sample_size)
    else:
        available_samples = [i for i in data_list if i['id'] not in iexclude_indices] 
        selected_samples = random.sample(available_samples, sample_size)
    return selected_samples, [i['id'] for i in selected_samples]

def save_list_to_json(data_list, file_path):
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data_list, file, ensure_ascii=False, indent=4)

def main():
    parser = argparse.ArgumentParser(description="Put rejection data in file1")

    parser.add_argument('--file1',  default="", help="dir for the first json file")
    parser.add_argument('--file2',  default='',help="dir for the second json file")
    parser.add_argument('--output', default="./trial_dataset", help="output json file dir")
    parser.add_argument('--sample_size1', type=int, default=1000, help="number of data sampled from first json file")
    parser.add_argument('--sample_size2', type=int, default=0, help="number of data sampled from the second json file")
    parser.add_argument('--similar', action="store_true", help="whether to keep the sampled data id identical")
    parser.add_argument('--change_reply', action="store_true")
    parser.add_argument('--reply', type=str, default="I'm sorry, but I cannot assist with your request")
    parser.add_argument('--suffix', type=str, default="")
    parser.add_argument('--random_choose', action="store_true")

    
    args = parser.parse_args()
    
    list1 = read_json_to_list(args.file1)
    # print(len(list1))
    list2 = read_json_to_list(args.file2)
    # print(len(list2))
    random_sample_list1, indices1 = get_random_samples(list1, min(args.sample_size1,len(list1)))
    random_sample_list2, _ = get_random_samples(list2, min(args.sample_size2,len(list2)), iexclude_indices=indices1,similar=args.similar)
    if args.random_choose:
        for i in random_sample_list1:
            length=len(i['conversations'])/2
            while True:
                pick=random.choice(range(1, int(length)+1))
                temp = i['conversations'][2*pick-2:2*pick]
                assert temp[0]['from'] == 'human' and temp[1]['from'] == 'gpt'
                if "what about" not in temp[0]['value'].lower():
                    temp[0]['value']=temp[0]['value'].replace("\n<image>","")
                    if not temp[0]['value'].startswith("<image>\n"):
                        temp[0]['value']=f"<image>\n{temp[0]['value']}"
                    i['conversations'] = temp
                    break
    if args.change_reply:
        for i in random_sample_list1:
            i['conversations'][1]["value"] = args.reply
    combined_random_samples = random_sample_list1 + random_sample_list2
    random.shuffle(combined_random_samples)

    if args.similar:
        save_name = f"{args.output}/{min(args.sample_size1,len(list1))}_{args.file1.split('/')[-1][:-5]}_{min(args.sample_size2,len(list2))}_{args.file2.split('/')[-1][:-5]}{args.suffix}_similar.json"
    else:
        save_name = f"{args.output}/{min(args.sample_size1,len(list1))}_{args.file1.split('/')[-1][:-5]}_{min(args.sample_size2,len(list2))}_{args.file2.split('/')[-1][:-5]}{args.suffix}.json"
    save_list_to_json(combined_random_samples, save_name)

    print("Save output datafile to", save_name)

if __name__ == "__main__":
    main()