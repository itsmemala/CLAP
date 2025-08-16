from tqdm import tqdm
import argparse
from utils import load_prompt_responses

verbal_unc_list = ['dont know','don\'t know','do not know','dont have','don\'t have','do not have','cannot','cant','can\'t','unable to']

def main(): 

    parser = argparse.ArgumentParser()
    parser.add_argument('save_path', type=str, default=None)
    parser.add_argument('model_name', type=str, default=None)
    parser.add_argument('dataset_name', type=str, default=None, help="One of {strqa,gsm8k,nq,tqa,city_country,player_date_birth,movie_cast)")
    parser.add_argument('file_name', type=str, default=None)
    parser.add_argument('num_samples', type=int, default=None, help="Number of responses sampled per prompt")
    args = parser.parse_args()

    questions, answers = load_prompt_responses(args)
    
    count_verbal_unc = 0
    for question,answer in tqdm(zip(questions,answers)):
        if any(substring in answer for substring in verbal_unc_list):
            print('Input Prompt:',question)
            print('Model Response:',answer)
            count_verbal_unc += 1
    print('# instances with verbal uncertainty:',count_verbal_unc)

if __name__ == '__main__':
    main()