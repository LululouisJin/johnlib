import re, os, json, tomotopy
import pandas as pd
import numpy as np
# from transformers import AutoTokenizer
from stanfordcorenlp import StanfordCoreNLP
from tqdm import tqdm

nlp = StanfordCoreNLP(r'D:\some_planes\stanfordnlp\stanford-corenlp-4.4.0')


class LDAInfer:
    def __init__(self) -> None:
        pass

    @staticmethod
    def generate_sample_list(data_list, index_list):
        sampled_list = []
        for i in index_list:
            sampled_list.append(data_list[i])
        return sampled_list

    @staticmethod
    def train_test_data(df, data_field, id_field='trial_id', label_filed='is_device', total_size=50000, ratio=0.8):
        doc_list = []
        label_list = []
        id_list = []
        for _, row in df.iterrows():
            if not pd.isna(row[data_field]):
                doc_list.append(row[data_field])
                label_list.append(row[label_filed])
                id_list.append(row[id_field])
        total_size = min(len(id_list), total_size)
        train_size = round(total_size * ratio)
        train_index = list(np.random.choice(total_size, train_size, replace=False))
        train_data = []
        test_data = []
        for i in range(total_size):
            data = {
                'trial_id': id_list[i],
                'doc': doc_list[i],
                'label': label_list[i]
            }
            if i in train_index:
                train_data.append(data)
            else:
                test_data.append(data)
        return train_data, test_data

    def train(self, model_save_path, train_data=None, train_data_path=None, topic_num=2):
        assert (train_data or train_data_path)
        if train_data_path:
            with open(train_data_path, 'r', encoding='utf-8') as f:
                train_data = json.load(f)
        mdl = tomotopy.LLDAModel(k=topic_num)
        for t in tqdm(train_data):
            text = t['doc']
            text = re.sub('\s|\n|\r|\t', ' ', text)
            text = re.sub(' +', ' ', text).strip()
            tok = nlp.word_tokenize(text)
            label = t['label']
            mdl.add_doc(tok, labels=[label])
        mdl.train()
        mdl.save(model_save_path)
        print(f'Model saved: {model_save_path}')

    @staticmethod
    def infer_doc(text, mdl):
        text = re.sub('\s|\n|\r|\t', ' ', text)
        text = re.sub(' +', ' ', text).strip()
        tok = nlp.word_tokenize(text)
        if not tok:
            print('Empty text')
            return
        doc_inst = mdl.make_doc(tok)
        topic_dist, ll = mdl.infer(doc_inst)
        topics = list(mdl.topic_label_dict)
        inference = list(topic_dist)
        max_value = max(inference)
        inference = inference.index(max_value)
        label = topics[inference]
        return label, max_value

    @staticmethod
    def obtain_topics(mdl):
        return list(mdl.topic_label_dict)

    @staticmethod
    def load_model(model_path):
        mdl = tomotopy.LLDAModel.load(model_path)
        return mdl

    # def infer_record(self, record, labels=['MD', 'IVD']):
    #     infer_fields = {
    #         'Scientific_Summary': 0.5, 
    #         'Scientific_Title': 0.2, 
    #         'Study_Title': 0.1, 
    #         'Summary': 0.4
    #     }
    #     result_dict = dict()
    #     infer_fields_list = list(infer_fields.keys())
    #     label_av = 0.
    #     label_bv = 0.
    #     for field in record.keys():
    #         if field in infer_fields_list:
    #             text = record[field]
    #             if isinstance(text, str):
    #                 result = self.infer_doc(text)
    #                 if result == labels[0]:
    #                     label_av += infer_fields.get(field)
    #                 else:
    #                     label_bv += infer_fields.get(field)
    #             else:
    #                 pass
    #         else:
    #             result_dict.update({field: record[field]})
    #     if label_av > label_bv:
    #         result_dict.update({'inference': labels[0]})
    #     else:
    #         result_dict.update({'inference': labels[0]})
    #     return result_dict

    def clinical_trials_md_infer(self, records, labels=['MD', 'IVD']):
        result = []
        for rec in records:
            res = self.infer_record(rec, labels)
            result.append(res)
        return result

if __name__ == '__main__':
    # data_path = r'D:\Project_2022\clinical_trial\clinical_trial_data\clinical_trial_data\output_data\clinical_trials_20221108.csv'
    # data = pd.read_csv(data_path, sep='|')
    # data.fillna('', inplace=True)

    # # train_data, test_data = train_test_data(data, 'summary')
    # with open('train_data.json', 'r', encoding='utf-8') as f:
    #     train_data = json.load(f)
    # for t in train_data:
    #     text = t['doc']
    #     text = re.sub('\s|\n|\r|\t', ' ', text)
    #     text = re.sub(' +', ' ', text).strip()
    #     tok = nlp.word_tokenize(text)
    #     label = t['label']
    #     if label:
    #         label = 'device'
    #     else:
    #         label = 'other'
    #     mdl.add_doc(tok, labels=[label])

    # mdl.train()

    # mdl.save('LLDA_device_infer_model_20221117.tmt')

    # input_path = r'C:\Users\LouisJin\Documents\For_Third_Party\NMPA_South_Section\2022_book\raw_data\device_diag_trials.xlsx'
    # df = pd.read_excel(input_path, sheet_name=0)
    # records = df.to_dict('records')

    # output_folder = r'C:\Users\LouisJin\Documents\For_Third_Party\NMPA_South_Section\2022_book\topic_inference'
    # output_file = 'Southern_institute_inference.json'
    # output_path = os.path.join(output_folder, output_file)

    # mdl = mdl.load('LLDA_fda510k.tmt')

    # with open('test_data.json', 'r', encoding='utf-8') as f:
    #     test_data = json.load(f)
    # count = 0
    # for data in test_data:
    #     text = data['doc']
    #     text = re.sub('\s|\n|\r|\t', ' ', text)
    #     text = re.sub(' +', ' ', text).strip()
    #     tok = nlp.word_tokenize(text)
    #     if not tok:
    #         tok = ['a']
    #     doc_inst = mdl.make_doc(tok)
    #     topic_dist, ll = mdl.infer(doc_inst)
    #     inference = list(topic_dist)
    #     max_value = max(inference)
    #     inference = inference.index(max_value)
    #     if inference == 1:
    #         inference =0
    #     else:
    #         inference = 1
    #     data.update({'inference': inference})
    #     topics = list(mdl.topic_label_dict)
    #     if topics[inference] == data['label']:
    #         count += 1
    # print(count / len(test_data))
    # with open('inference_data_20221117.json', 'w', encoding='utf-8') as f:
    #     f.write(json.dumps(test_data, indent=4))

    # raw_data_path = r'C:\Users\LouisJin\Documents\For_Third_Party\NMPA_South_Section\2022_book\raw_data\ivd_sample_data\combined_data.csv'
    # df = pd.read_csv(raw_data_path)
    # infer = LDAInfer()
    # data_field = ['detailed_description', 'scientific_title', 'scientific_summary', 'study_title']
    # for field in data_field:
    #     train_data, test_data = infer.train_test_data(df, data_field=field, id_field='Unnamed: 0', label_filed='label', ratio=1.)
    #     save_folder = r'C:\Users\LouisJin\Documents\For_Third_Party\NMPA_South_Section\2022_book\topic_inference'
    #     file_name = f'clinical_infer_{field}.tmt'
    #     save_path = os.path.join(save_folder, file_name)
    #     infer.train(train_data=train_data, model_save_path=save_path)
    pass


