from Models.DataBase import PostgresqlDB
from Models.hf_models import HFBertEncoder
import faiss
import numpy as np
import torch
from transformers import DPRContextEncoder, BertTokenizer, DPRQuestionEncoder, XLMRobertaTokenizer
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer, DistilBertForQuestionAnswering


class DPRDataset(Dataset):
    def __init__(self, docs, context_model, device, max_length=256, do_lower_case=True, model_type="hf_bert"):
        if model_type == "hf_bert":
            self.tokenizer = BertTokenizer.from_pretrained(context_model, do_lower_case=do_lower_case)
        else:
            self.tokenizer = XLMRobertaTokenizer.from_pretrained(context_model)
        self.device = device
        self.docs = docs
        self.max_length = max_length

    def __getitem__(self, idx):
        input = self.tokenizer.encode_plus(self.docs[idx]["meta"]["title"], text_pair=self.docs[idx]["text"],
                                           max_length=self.max_length, return_tensors="pt",
                                           truncation=True, padding='max_length')
        token_type_ids = torch.zeros_like(input["input_ids"].squeeze())

        return {"input_ids": input["input_ids"].squeeze(),
                "attention_mask": input["attention_mask"].squeeze(),
                "token_type_ids": token_type_ids}

    def __len__(self):
        return len(self.docs)


class DPRDatasetQuestions(Dataset):
    def __init__(self, docs, context_model, device, max_length=256, do_lower_case=True, model_type="hf_bert", field="question"):
        if model_type == "hf_bert":
            self.tokenizer = BertTokenizer.from_pretrained(context_model, do_lower_case=do_lower_case)
        elif model_type == "hf_roberta_encoder":
            self.tokenizer = XLMRobertaTokenizer.from_pretrained(context_model)
            self.tokenizer.add_tokens(["<q>", '<ctx>'])
            self.tokenizer.add_special_tokens({'cls_token': '<q>'})
        else:
            self.tokenizer = XLMRobertaTokenizer.from_pretrained(context_model)

        self.device = device
        self.docs = docs
        self.field = field
        self.max_length = max_length

    def __getitem__(self, idx):
        input = self.tokenizer.encode_plus(self.docs[idx][self.field],
                                           max_length=self.max_length, return_tensors="pt",
                                           truncation=True, padding='max_length')
        token_type_ids = torch.zeros_like(input["input_ids"].squeeze())

        return {"input_ids": input["input_ids"].squeeze().to(self.device),
                "attention_mask": input["attention_mask"].squeeze().to(self.device),
                "token_type_ids": token_type_ids.to(self.device)}

    def __len__(self):
        return len(self.docs)


class DPREncoder:
    def __init__(self, device="cpu", question_model=None,
                 model_type="hf_bert", config=None):
        self.model_type = model_type
        self.config = config
        self.reader_tokenizer = DistilBertTokenizer.from_pretrained(config["reader"]["model"])
        self.reader = DistilBertForQuestionAnswering.from_pretrained(config["reader"]["model"])

        if question_model is not None:
            self.context_model_path = question_model
            if model_type == "hf_bert":
                self.tokenizer = BertTokenizer.from_pretrained(question_model)
                self.question_model = HFBertEncoder.from_pretrained(question_model)
                self.question_model.eval()

            self.question_model.to(device)

        self.device = device
        self.faiss_index = None
        self.vector_dim = None
        self.db = PostgresqlDB(**config["db"])

    def encode_question(self, question, max_length=256):
        input = self.tokenizer.encode_plus(question,
                                   max_length=max_length, return_tensors="pt",
                                   truncation=True, padding='max_length')
        attention_mask = input["attention_mask"].reshape(1, max_length).to(self.device)
        sample = input["input_ids"].reshape(1, max_length).to(self.device)
        token_type_ids = torch.zeros_like(input["input_ids"]).reshape(1, max_length).to(self.device)
        input["input_ids"].reshape(1, max_length).to(self.device)
        out = self.question_model(sample, attention_mask=attention_mask, token_type_ids=token_type_ids)
        if self.model_type == "hf_bert":
            out = out[0].detach().cpu().numpy()[:, 0, :]
        else:
            out = out[1].detach().cpu().numpy()
        return out

    def create_faiss(self, index, n_jobs=10):
        faiss.omp_set_num_threads(n_jobs)
        self.faiss_index = faiss.read_index(index)
        self.faiss_index.nprobe = 50

    def write_documents(self, documents):
        self.documents += documents
        embeddings = np.zeros((len(documents), self.vector_dim))
        for i, document in enumerate(documents):
            embeddings[i] = document["embedding"]
        embeddings = embeddings.astype(np.dtype('float32'))

        self.faiss_index.add(embeddings)

    def read(self, question, text):
        inputs = self.reader_tokenizer(question, text, return_tensors='pt')

        outputs = self.reader(**inputs)
        return self.reader_tokenizer.decode(inputs["input_ids"][0][outputs[0][0].argmax(): outputs[1][0].argmax() + 1])

    def retrieve(self, q, top_k):
        query_emb = self.encode_question(q)
        score_matrix, vector_id_matrix = self.faiss_index.search(query_emb, top_k)

        contexts = [self.db.get_on_id(i) for i in vector_id_matrix[0]]

        out = [[self.read(q, item[2]),
                item[1], item[2]] for item in contexts]

        return out

    def retrieve_batchs(self, docs, top_k=10, field="question", score=False,
                        return_vector_id_matrix=False, batch_size=32):
        dataset = DPRDatasetQuestions(docs, self.context_model_path, self.device,
                                      256, self.do_lower_case, model_type=self.model_type, field=field, )

        loader = DataLoader(dataset, batch_size, shuffle=False)

        embeddings = np.zeros((len(docs), 768))

        count = 0
        for batch in loader:
            if self.model_type == "hf_bert":
                out = self.question_model(batch["input_ids"],
                                          attention_mask=batch["attention_mask"],
                                          token_type_ids=batch["token_type_ids"])[0]
            else:
                out = self.question_model(batch["input_ids"],
                                          attention_mask=batch["attention_mask"],
                                          token_type_ids=batch["token_type_ids"])[1]

            for j in range(len(out)):
                embeddings[count] = out[j].detach().cpu().numpy()[0, :]
                count += 1
            del out, batch

        score_matrix, vector_id_matrix = self.faiss_index.search(embeddings.astype(np.dtype('float32')), top_k)
        if return_vector_id_matrix:
            return [[{"document": self.documents[j], "id": int(j)} for index, j in
                     enumerate(vector_id_matrix[i])]
                    for i in range(vector_id_matrix.shape[0])]
        if score:
            return [[{"document": self.documents[j], "score": float(score_matrix[i][index])} for index, j in enumerate(vector_id_matrix[i])]
                    for i in range(vector_id_matrix.shape[0])]

        return [[self.documents[j] for j in vector_id_matrix[i]] for i in range(vector_id_matrix.shape[0])]








