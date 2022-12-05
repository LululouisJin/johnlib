import json, re, io, heapq, os
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import as_completed
import numpy as np
import pandas as pd

# For NLP task
import torch
from stanfordcorenlp import StanfordCoreNLP
from sentence_transformers import SentenceTransformer, util
from torch.nn import functional as F
from transformers import AutoTokenizer

SYNTAX_MODEL = StanfordCoreNLP(r'D:\some_planes\stanfordnlp\stanford-corenlp-4.4.0')

# model_path = r'D:\data\test\sentence-transformers_all-MiniLM-L6-v2'
MODEL_MAP = {
    'sentence_similarity': 'all-MiniLM-L6-v2',
    'token_classification': 'xlm-roberta-large-finetuned-conll03-english',
    'translation': 'Helsinki-NLP/opus-mt-en-zh',
    'question_answer': 'facebook/bart-large-mnli'
}


class TaskExecutor:
    def __init__(self) -> None:
        pass

    def prepare_data(self):
        pass

    def compare_simi(self, target_list, compare_list, n=1, simi_threshold=0.4):
        target_embedding = SemanticSimiTool.vectorize(target_list)
        target_embedding = F.normalize(target_embedding)
        compare_embedding = SemanticSimiTool.vectorize(compare_list)
        compare_embedding = F.normalize(compare_embedding)
        simi_matrix = torch.mm(target_embedding, compare_embedding.T)
        result = []
        for i in range(simi_matrix.shape[0]):
            target = simi_matrix[i]
            target = target.tolist()
            sorted_values = heapq.nlargest(n, target)  # 求最大的n个元素，并排序
            sorted_args = map(target.index, heapq.nlargest(n, target))
            sorted_args = list(sorted_args)
            print(f'max values: {sorted_values}')
            print(f'max args: {sorted_args}')
            dict_ = dict()
            dict_['target'] = target_list[i]
            for j in range(n):
                # print(f'{target_list[i]} - {sorted_values[j]}: {compare_list[sorted_args[j]]}')
                if sorted_values[j] >= simi_threshold:
                    dict_[f'max_{j}'] = compare_list[sorted_args[j]]
                    dict_[f'max_{j}_score'] = sorted_values[j]
                else:
                    dict_[f'max_{j}'] = ''
                    dict_[f'max_{j}_score'] = 0
            result.append(dict_)
        # result = pd.DataFrame(result)
        return result

    def compare_simi_one(self, target_list, compare_str):
        target_embedding = SemanticSimiTool.vectorize(target_list)
        target_embedding = F.normalize(target_embedding)
        compare_embedding = SemanticSimiTool.vectorize([compare_str])
        compare_embedding = F.normalize(compare_embedding)
        simi_matrix = torch.mm(target_embedding, compare_embedding.T)
        result = []
        for i in range(simi_matrix.shape[0]):
            score = simi_matrix[i]
            result.append(score)
        return result

    @staticmethod
    def most_n_simi(query, comp_list, simi_func, n=3, **kw):
        result_dict = dict()
        top_n = [10.] + [0.] * n
        result_dict.update({'query': query})
        for compare in comp_list:
            simi_score = simi_func(query, compare, **kw)
            for i in range(1, n + 1):
                if top_n[i - 1] >= simi_score > top_n[i]:
                    # Move residual values back.
                    if i < n:
                        top_n[i + 1:] = top_n[i:-1]
                    # Insert simi_score as i-th max value.
                    top_n[i] = simi_score
                    # result_dict.update({
                    #     f'top_{i}': {
                    #         'name': compare,
                    #         'score': simi_score
                    #     }
                    # })
                    result_dict.update({f'top_{i}': f'{compare} - {simi_score}'})
                    break
        return result_dict

    @staticmethod
    def process_all_tasks(tasks_list, func, max_worker=8, pool=ProcessPoolExecutor, **kw):
        result = []
        with pool(max_workers=max_worker) as executor:
            # future = executor.map()
            future = [executor.submit(func, query, **kw) for query in tasks_list]
            for f in future:
                if f.running():
                    print(f'{f} is running.')
            for f in as_completed(future):
                try:
                    ret = f.done()
                    if ret:
                        f_ret = f.result()
                        if f_ret:
                            result.append(f_ret)
                            print(f_ret)
                except Exception as e:
                    f.cancel()
                    print(e)
        if not result:
            return []
        return result

    @staticmethod
    def query_df(dataframe, field, query, method='in_list'):
        df = pd.DataFrame(columns=dataframe.columns)
        if method == 'in_list':
            df = dataframe.query(f'{field} in {query}', engine='python')
        if method == 'contain':
            df = dataframe.query(f'{field}.str.contains("{query}", case=False, na=False)', engine='python')
        if method == 'equal':
            if isinstance(query, str):
                df = dataframe.query(f'{field} == "{query}"', engine='python')
            else:
                if isinstance(query, int):
                    dataframe[field] = dataframe[field].apply(lambda x: int(x))
                if isinstance(query, float):
                    dataframe[field] = dataframe[field].apply(lambda x: float(x))
                df = dataframe.query(f'{field} == {query}', engine='python')
        return df


class SemanticTool:
    def __init__(self, task='sentence_similarity', model_map=MODEL_MAP) -> None:
        model_info = model_map.get(task)
        assert model_info, NameError('Illegal task name.')
        if task == 'sentence_similarity':
            self.model = SentenceTransformer(model_info)
        else:
            if task == 'token_classification':
                from transformers import AutoModelForTokenClassification
                self.model = AutoModelForTokenClassification.from_pretrained(model_info)
            elif task == 'translation':
                from transformers import AutoModelForSeq2SeqLM
                self.model = AutoModelForSeq2SeqLM.from_pretrained(model_info)
            elif task == 'question_answer':
                from transformers import AutoModelForSequenceClassification
                self.model = AutoModelForSequenceClassification.from_pretrained(model_info)
            self.tokenizer = AutoTokenizer.from_pretrained(model_info)

    @staticmethod
    def embedding_cos_simi(sent1, sent2, model=None):
        if not model:
            model = self.model
        embedding_1 = model.encode(sent1, convert_to_tensor=True)
        embedding_2 = model.encode(sent2, convert_to_tensor=True)
        score = util.cos_sim(embedding_1, embedding_2).item()
        return score

    @staticmethod
    def vectorize(sent_list: list, model=None):
        if not model:
            model = self.model
        embedding = model.encode(sent_list, convert_to_tensor=True)
        return embedding

    def token_classification(self, text, tokenizer=None, model=None):
        if not model:
            model = self.model
        if not tokenizer:
            tokenizer = self.tokenizer
        inputs = tokenizer(text, add_special_tokens=False, return_tensors="pt")
        with torch.no_grad():
            logits = model(**inputs).logits
        predicted_token_class_ids = logits.argmax(-1)
        predicted_tokens_classes = [model.config.id2label[t.item()] for t in predicted_token_class_ids[0]]
        return dict(zip(inputs.tokens(), predicted_tokens_classes))

    @staticmethod
    def ner(tokens_dict):
        ner_dict = dict()
        word = ''
        word_c = 'O'
        phrase = ''
        phrase_c = 'O'
        for token, c in tokens_dict.items():
            # word start
            if token.startswith('▁'):
                # Combine phrase
                if word_c == phrase_c:
                    phrase += word
                # Establish new phrase
                else:
                    # Update meaningful phrase
                    if phrase_c != 'O':
                        ner_dict.update({phrase: phrase_c})
                    # Refresh phrase and phrase class
                    phrase = word
                    phrase_c = word_c
                # Establish new word
                word = token
                # Initialize word class to make sure word class consistency in a word
                word_c = 'O'
            # word residuals
            else:
                # Complete a word
                word += token
            # Update word class if the class is meaningful
            if c != 'O':
                word_c = c
        # Complete last phrase
        if phrase_c == word_c:
            phrase += word
            if phrase and word_c != 'O':
                ner_dict.update({phrase: word_c})
        elif phrase and phrase_c != 'O':
            ner_dict.update({phrase, phrase_c})
            if word and word_c != 'O':
                ner_dict.update({word: word_c})
        return ner_dict

    def zero_shot_classify(self, premise: str, label: str = None, hypothesis: str = None, model=None, threshold=0.8, device='cpu', exemptions=[]):
        '''
        Load the zero-shot classification model to deal with single-label classification task.
        :param premise: The given text which is raw language material used to infer the hypothesis.
        :param label: Effective when execute an "is or is not" inference.
        :param hypothesis: Effective when execute a "true or false" judgement for hypothesis.
        :param model: NLP model used to generate output value.
        :param threshold: The threshold score to accept hypothesis.
        :return: Accept the hypothesis or not, and probability of model result, like: "True, 0.98"
        '''
        if not model:
            model = self.model
        assert (label or hypothesis), NameError('label or hypothesis must be provided.')
        if label:
            hypothesis = f'This example is {label}.'
        x = self.tokenizer.encode(premise, hypothesis, return_tensors='pt', truncation='only_first')
        with torch.no_grad():
            logits = model(x.to(device))[0]
        entail_contradiction_logits = logits[:, [0, 2]]
        probs = entail_contradiction_logits.softmax(dim=1)
        prob_label_is_true = probs[:, 1].item()
        if prob_label_is_true > threshold:
            inference = True
        else:
            inference = False
        if premise in exemptions:
            inference = True
            prob_label_is_true = 1.
        return inference, prob_label_is_true

    def helsinki_translate(self, text):
        batch = self.tokenizer([text], return_tensors="pt")
        generated_ids = self.model.generate(**batch)
        answer = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return answer


class TextSimiTool:
    def __init__(self) -> None:
        self.CHARS = '''
        0123456789qwertyuiopasdfghjklzxcvbnmQWERTYUIOPASDFGHJKLZXCVBNM.- ()[]\{\}_*
        '''
        self.CHAR_MAP = self.generate_map(self.CHARS)

    @staticmethod
    def generate_map(text):
        char_vec_map = dict()
        chars_set = set(text)
        length = len(list(chars_set))
        for n, char in enumerate(list(chars_set)):
            one_hot_vec = np.zeros(length)
            one_hot_vec[n] = 1
            char_vec_map[char] = one_hot_vec.reshape(1, length)
        return char_vec_map

    @classmethod
    def one_hot_encode(cls, text, char_map=None):
        if not char_map:
            char_map = cls().CHAR_MAP
        result = np.empty((0, len(char_map)))
        for char in text:
            result = np.concatenate((result, char_map[char]), axis=0)
        return result

    @staticmethod
    def sub_char(string, split=True):
        pattern = '[^0-9a-zA-Z\.\-()]'
        result = re.sub(pattern, ' ', string.strip()).replace('\r\n', ' ').replace('\t', ' ').strip()
        result = re.sub(' +', ' ', result)
        if split:
            result = result.split(' ')
        return result

    @staticmethod
    def len_list_str(str_list):
        length = 0
        for string in str_list:
            length += len(string)
        return length

    @staticmethod
    def cos_simi(vec1, vec2):
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    @staticmethod
    def normal_levenshtein(str1, str2):
        len_str1 = len(str1) + 1
        len_str2 = len(str2) + 1
        # 创建矩阵
        matrix = [0 for n in range(len_str1 * len_str2)]
        # 矩阵的第一行
        for i in range(len_str1):
            matrix[i] = i
        # 矩阵的第一列
        for j in range(0, len(matrix), len_str1):
            if j % len_str1 == 0:
                matrix[j] = j // len_str1
        # 根据状态转移方程逐步得到编辑距离
        for i in range(1, len_str1):
            for j in range(1, len_str2):
                if str1[i - 1] == str2[j - 1]:
                    cost = 0
                else:
                    cost = 1
                matrix[j * len_str1 + i] = min(matrix[(j - 1) * len_str1 + i] + 1,
                                               matrix[j * len_str1 + (i - 1)] + 1,
                                               matrix[(j - 1) * len_str1 + (i - 1)] + cost)
        return matrix[-1]  # 返回矩阵的最后一个值，也就是编辑距离

    @classmethod
    def levenshtein_ratio(cls, str1, str2):
        dist = cls().normal_levenshtein(str1, str2)
        sum_len = len(str1) + len(str2)
        return (sum_len - dist) / sum_len

    @classmethod
    def john_matrix_simi(cls, str1, str2):
        str1 = cls.sub_char(str1, split=False)
        str2 = cls.sub_char(str2, split=False)
        if len(str1) > len(str2):
            if len(str2) < 3:
                return 0
            long_str = str1
            short_str = str2
        elif len(str1) <= len(str2):
            long_str = str2
            short_str = str1
        long_len = len(long_str)
        short_len = len(short_str)
        vec_long = cls().one_hot_encode(long_str)
        vec_short = cls().one_hot_encode(short_str)
        if long_len == short_len:
            # print('equal')
            result = np.multiply(vec_long, vec_short)
            result = np.sum(result, axis=0).sum()
            return (short_len + result) / (short_len + long_len)
        # print('not equal')
        max_simi = 0
        for i in range(len(long_str) - len(short_str) + 1):
            long_slice = vec_long[i:i + short_len]
            simi = np.multiply(vec_short, long_slice)
            # print(simi)
            simi = np.sum(simi, axis=0).sum()
            max_simi = np.maximum(max_simi, simi)
        return (short_len + max_simi) / (short_len + long_len)

    @classmethod
    def john_split_simi(cls, str_o, str_c):
        if len(str_o) > len(str_c):
            long = str_o
            short = str_c
        else:
            long = str_c
            short = str_o
        str_o = short
        str_c = long
        list_o = cls().sub_char(str_o)
        list_c = cls().sub_char(str_c)
        len_o = cls().len_list_str(list_o)
        len_c = cls().len_list_str(list_c)
        sum_simi = 0
        for word_o in list_o:
            max_simi = 0
            for word_c in list_c:
                simi = cls().enhanced_matrix_simi(word_o, word_c) * len(word_o)
                if simi > max_simi:
                    max_simi = simi
            sum_simi += max_simi
        str_simi = sum_simi / (len_o + 0.05 * len_c)
        return str_simi


if __name__ == '__main__':
    sent = 'Johns Hopkins University[a] (Johns Hopkins, Hopkins, or JHU) is a private research university in Baltimore, Maryland. Founded in 1876, Johns Hopkins is the oldest research university in the United States and in the western hemisphere.[6] It consistently ranks among the most prestigious universities in the United States and the world.[7][8][9]'
    tool = SemanticTool('token_classification')
    info = tool.token_classification(sent)
    ner_res = tool.ner(info)
    pass