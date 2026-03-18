import os
import pickle
import contextlib
import heapq
import time
import math

from index import InvertedIndexReader, InvertedIndexWriter
from util import IdMap, sorted_merge_posts_and_tfs
from compression import StandardPostings, VBEPostings
from tqdm import tqdm

class BSBIIndex:
    """
    Attributes
    ----------
    term_id_map(IdMap): Untuk mapping terms ke termIDs
    doc_id_map(IdMap): Untuk mapping relative paths dari dokumen (misal,
                    /collection/0/gamma.txt) to docIDs
    data_dir(str): Path ke data
    output_dir(str): Path ke output index files
    postings_encoding: Lihat di compression.py, kandidatnya adalah StandardPostings,
                    VBEPostings, dsb.
    index_name(str): Nama dari file yang berisi inverted index
    """
    def __init__(self, data_dir, output_dir, postings_encoding, index_name = "main_index"):
        self.term_id_map = IdMap()
        self.doc_id_map = IdMap()
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.index_name = index_name
        self.postings_encoding = postings_encoding

        # Untuk menyimpan nama-nama file dari semua intermediate inverted index
        self.intermediate_indices = []

    def save(self):
        """Menyimpan doc_id_map and term_id_map ke output directory via pickle"""

        with open(os.path.join(self.output_dir, 'terms.dict'), 'wb') as f:
            pickle.dump(self.term_id_map, f)
        with open(os.path.join(self.output_dir, 'docs.dict'), 'wb') as f:
            pickle.dump(self.doc_id_map, f)

    def load(self):
        """Memuat doc_id_map and term_id_map dari output directory"""

        with open(os.path.join(self.output_dir, 'terms.dict'), 'rb') as f:
            self.term_id_map = pickle.load(f)
        with open(os.path.join(self.output_dir, 'docs.dict'), 'rb') as f:
            self.doc_id_map = pickle.load(f)

    def parse_block(self, block_dir_relative):
        """
        Lakukan parsing terhadap text file sehingga menjadi sequence of
        <termID, docID> pairs.

        Gunakan tools available untuk Stemming Bahasa Inggris

        JANGAN LUPA BUANG STOPWORDS!

        Untuk "sentence segmentation" dan "tokenization", bisa menggunakan
        regex atau boleh juga menggunakan tools lain yang berbasis machine
        learning.

        Parameters
        ----------
        block_dir_relative : str
            Relative Path ke directory yang mengandung text files untuk sebuah block.

            CATAT bahwa satu folder di collection dianggap merepresentasikan satu block.
            Konsep block di soal tugas ini berbeda dengan konsep block yang terkait
            dengan operating systems.

        Returns
        -------
        List[Tuple[Int, Int]]
            Returns all the td_pairs extracted from the block
            Mengembalikan semua pasangan <termID, docID> dari sebuah block (dalam hal
            ini sebuah sub-direktori di dalam folder collection)

        Harus menggunakan self.term_id_map dan self.doc_id_map untuk mendapatkan
        termIDs dan docIDs. Dua variable ini harus 'persist' untuk semua pemanggilan
        parse_block(...).
        """
        dir = "./" + self.data_dir + "/" + block_dir_relative
        td_pairs = []
        for filename in next(os.walk(dir))[2]:
            docname = dir + "/" + filename
            with open(docname, "r", encoding = "utf8", errors = "surrogateescape") as f:
                for token in f.read().split():
                    td_pairs.append((self.term_id_map[token], self.doc_id_map[docname]))

        return td_pairs

    def invert_write(self, td_pairs, index):
        """
        Melakukan inversion td_pairs (list of <termID, docID> pairs) dan
        menyimpan mereka ke index. Diterapkan konsep BSBI dengan strategi SPIMI.
        Upper bound score BM25 per term juga dihitung dan disimpan untuk WAND.

        Parameters
        ----------
        td_pairs : List[Tuple[Int, Int]]
            List of termID-docID pairs.
        index : InvertedIndexWriter
            Inverted index pada disk (file) yang terkait dengan suatu block.
        """
        term_dict = {}
        term_tf = {}
        for term_id, doc_id in td_pairs:
            if term_id not in term_dict:
                term_dict[term_id] = set()
                term_tf[term_id] = {}
            term_dict[term_id].add(doc_id)
            if doc_id not in term_tf[term_id]:
                term_tf[term_id][doc_id] = 0
            term_tf[term_id][doc_id] += 1
        for term_id in sorted(term_dict.keys()):
            sorted_doc_id = sorted(list(term_dict[term_id]))
            assoc_tf = [term_tf[term_id][doc_id] for doc_id in sorted_doc_id]
            # hitung upper bound: max TF component BM25 across all docs
            upper_bound = max(assoc_tf)
            index.append(term_id, sorted_doc_id, assoc_tf, upper_bound)

    def merge(self, indices, merged_index):
        """
        Lakukan merging ke semua intermediate inverted indices menjadi
        sebuah single index.

        Ini adalah bagian yang melakukan EXTERNAL MERGE SORT

        Gunakan fungsi orted_merge_posts_and_tfs(..) di modul util

        Parameters
        ----------
        indices: List[InvertedIndexReader]
            A list of intermediate InvertedIndexReader objects, masing-masing
            merepresentasikan sebuah intermediate inveted index yang iterable
            di sebuah block.

        merged_index: InvertedIndexWriter
            Instance InvertedIndexWriter object yang merupakan hasil merging dari
            semua intermediate InvertedIndexWriter objects.
        """
        # kode berikut mengasumsikan minimal ada 1 term
        merged_iter = heapq.merge(*indices, key = lambda x: x[0])
        curr, postings, tf_list = next(merged_iter) # first item
        for t, postings_, tf_list_ in merged_iter: # from the second item
            if t == curr:
                zip_p_tf = sorted_merge_posts_and_tfs(list(zip(postings, tf_list)), \
                                                      list(zip(postings_, tf_list_)))
                postings = [doc_id for (doc_id, _) in zip_p_tf]
                tf_list = [tf for (_, tf) in zip_p_tf]
            else:
                merged_index.append(curr, postings, tf_list)
                curr, postings, tf_list = t, postings_, tf_list_
        merged_index.append(curr, postings, tf_list)

    def _advance_ptr(self, postings, ptr, target):
        """
        Memajukan pointer ptr pada postings list hingga mencapai
        posisi dengan DID >= target.

        Parameters
        ----------
        postings : List[int]
            Postings list.
        ptr : int
            Posisi pointer saat ini.
        target : int
            Target DID minimum yang ingin dicapai.

        Returns
        -------
        int
            Posisi pointer baru.
        """
        while ptr < len(postings) and postings[ptr] < target:
            ptr += 1
        return ptr

    def retrieve_tfidf(self, query, k = 10):
        """
        Melakukan Ranked Retrieval dengan skema TaaT (Term-at-a-Time).
        Method akan mengembalikan top-K retrieval results.

        w(t, D) = (1 + log tf(t, D))       jika tf(t, D) > 0
                = 0                        jika sebaliknya

        w(t, Q) = IDF = log (N / df(t))

        Score = untuk setiap term di query, akumulasikan w(t, Q) * w(t, D).
                (tidak perlu dinormalisasi dengan panjang dokumen)

        catatan: 
            1. informasi DF(t) ada di dictionary postings_dict pada merged index
            2. informasi TF(t, D) ada di tf_li
            3. informasi N bisa didapat dari doc_length pada merged index, len(doc_length)

        Parameters
        ----------
        query: str
            Query tokens yang dipisahkan oleh spasi

            contoh: Query "universitas indonesia depok" artinya ada
            tiga terms: universitas, indonesia, dan depok

        Result
        ------
        List[(int, str)]
            List of tuple: elemen pertama adalah score similarity, dan yang
            kedua adalah nama dokumen.
            Daftar Top-K dokumen terurut mengecil BERDASARKAN SKOR.

        JANGAN LEMPAR ERROR/EXCEPTION untuk terms yang TIDAK ADA di collection.

        """
        if len(self.term_id_map) == 0 or len(self.doc_id_map) == 0:
            self.load()

        terms = [self.term_id_map[word] for word in query.split()]
        with InvertedIndexReader(self.index_name, self.postings_encoding, directory=self.output_dir) as merged_index:

            scores = {}
            for term in terms:
                if term in merged_index.postings_dict:
                    df = merged_index.postings_dict[term][1]
                    N = len(merged_index.doc_length)
                    postings, tf_list = merged_index.get_postings_list(term)
                    for i in range(len(postings)):
                        doc_id, tf = postings[i], tf_list[i]
                        if doc_id not in scores:
                            scores[doc_id] = 0
                        if tf > 0:
                            scores[doc_id] += math.log(N / df) * (1 + math.log(tf))

            # Top-K
            docs = [(score, self.doc_id_map[doc_id]) for (doc_id, score) in scores.items()]
            return sorted(docs, key = lambda x: x[0], reverse = True)[:k]

    def index(self):
        """
        Base indexing code
        BAGIAN UTAMA untuk melakukan Indexing dengan skema BSBI (blocked-sort
        based indexing)

        Method ini scan terhadap semua data di collection, memanggil parse_block
        untuk parsing dokumen dan memanggil invert_write yang melakukan inversion
        di setiap block dan menyimpannya ke index yang baru.
        """
        # loop untuk setiap sub-directory di dalam folder collection (setiap block)
        for block_dir_relative in tqdm(sorted(next(os.walk(self.data_dir))[1])):
            td_pairs = self.parse_block(block_dir_relative)
            index_id = 'intermediate_index_'+block_dir_relative
            self.intermediate_indices.append(index_id)
            with InvertedIndexWriter(index_id, self.postings_encoding, directory = self.output_dir) as index:
                self.invert_write(td_pairs, index)
                td_pairs = None
    
        self.save()

        with InvertedIndexWriter(self.index_name, self.postings_encoding, directory = self.output_dir) as merged_index:
            with contextlib.ExitStack() as stack:
                indices = [stack.enter_context(InvertedIndexReader(index_id, self.postings_encoding, directory=self.output_dir))
                               for index_id in self.intermediate_indices]
                self.merge(indices, merged_index)
    
    def retrieve_bm25(self, query, k=10, k1=1.2, b=0.75):
        """
        Melakukan Ranked Retrieval dengan skema TaaT (Term-at-a-Time)
        menggunakan scoring BM25. Method akan mengembalikan top-K retrieval results.

        Formula BM25:
            RSV_BM25 = sum over t in Q∩D of:
                log(N / df_t) * ((k1 + 1) * tf_t) / (k1 * ((1 - b) + b * (dl / avgdl)) + tf_t)

        Parameters
        ----------
        query : str
            Query tokens yang dipisahkan oleh spasi.
        k : int
            Banyaknya dokumen teratas yang dikembalikan.
        k1 : float
            Parameter saturasi TF. Nilai default 1.2.
        b : float
            Parameter normalisasi panjang dokumen. Nilai default 0.75.

        Result
        ------
        List[(int, str)]
            List of tuple: elemen pertama adalah score similarity, dan yang
            kedua adalah nama dokumen.
            Daftar Top-K dokumen terurut mengecil BERDASARKAN SKOR.

        JANGAN LEMPAR ERROR/EXCEPTION untuk terms yang TIDAK ADA di collection.
        """
        if len(self.term_id_map) == 0 or len(self.doc_id_map) == 0:
            self.load()

        terms = [self.term_id_map[word] for word in query.split()]
        with InvertedIndexReader(self.index_name, self.postings_encoding, directory=self.output_dir) as merged_index:
            
            # hitung average document length
            avgdl = sum(merged_index.doc_length.values()) / len(merged_index.doc_length)
            N = len(merged_index.doc_length)

            scores = {}
            for term in terms:
                if term in merged_index.postings_dict:
                    df = merged_index.postings_dict[term][1]
                    postings, tf_list = merged_index.get_postings_list(term)
                    for i in range(len(postings)):
                        doc_id, tf = postings[i], tf_list[i]
                        if doc_id not in scores:
                            scores[doc_id] = 0
                        dl = merged_index.doc_length[doc_id]
                        idf = math.log(N / df)
                        tf_norm = ((k1 + 1) * tf) / (k1 * ((1 - b) + b * (dl / avgdl)) + tf)
                        scores[doc_id] += idf * tf_norm

            # Top-K
            docs = [(score, self.doc_id_map[doc_id]) for (doc_id, score) in scores.items()]
            return sorted(docs, key=lambda x: x[0], reverse=True)[:k]

    def retrieve_bm25_wand(self, query, k=10, k1=1.2, b=0.75):
        """
        Melakukan Ranked Retrieval dengan algoritma WAND (Weak AND) Top-K
        menggunakan scoring BM25. Dokumen yang tidak mungkin masuk top-K
        di-skip tanpa menghitung score penuh, sehingga lebih efisien.

        Parameters
        ----------
        query : str
            Query tokens yang dipisahkan oleh spasi.
        k : int
            Banyaknya dokumen teratas yang dikembalikan.
        k1 : float
            Parameter saturasi TF. Nilai default 1.2.
        b : float
            Parameter normalisasi panjang dokumen. Nilai default 0.75.

        Result
        ------
        List[(int, str)]
            List of tuple: elemen pertama adalah score similarity, dan yang
            kedua adalah nama dokumen.
            Daftar Top-K dokumen terurut mengecil BERDASARKAN SKOR.

        JANGAN LEMPAR ERROR/EXCEPTION untuk terms yang TIDAK ADA di collection.
        """
        if len(self.term_id_map) == 0 or len(self.doc_id_map) == 0:
            self.load()

        with InvertedIndexReader(self.index_name, self.postings_encoding, directory=self.output_dir) as merged_index:

            avgdl = sum(merged_index.doc_length.values()) / len(merged_index.doc_length)
            N = len(merged_index.doc_length)

            # setup per-term data
            query_terms = []
            for word in query.split():
                term_id = self.term_id_map[word]
                if term_id in merged_index.postings_dict:
                    postings, tf_list = merged_index.get_postings_list(term_id)
                    ub = merged_index.postings_dict[term_id][4]
                    df = merged_index.postings_dict[term_id][1]
                    idf = math.log(N / df)
                    query_terms.append({
                        'term_id': term_id,
                        'postings': postings,
                        'tf_list': tf_list,
                        'ub': ub * idf,
                        'idf': idf,
                        'ptr': 0  # pointer ke posisi saat ini di postings list
                    })

            if not query_terms:
                return []

            # threshold awal
            threshold = 0.0
            top_k = []  # min-heap of (score, doc_id)

            cur_doc = -1

            while True:
                # sort terms by current DID (postings[ptr])
                query_terms = [t for t in query_terms if t['ptr'] < len(t['postings'])]
                if not query_terms:
                    break
                query_terms.sort(key=lambda t: t['postings'][t['ptr']])

                # find pivot term
                pivot_term_idx = None
                accum_ub = 0.0
                for i, t in enumerate(query_terms):
                    accum_ub += t['ub']
                    if accum_ub >= threshold:
                        pivot_term_idx = i
                        break

                if pivot_term_idx is None:
                    break

                pivot = query_terms[pivot_term_idx]['postings'][query_terms[pivot_term_idx]['ptr']]

                if pivot == cur_doc:
                    break

                if pivot <= cur_doc:
                    # advance one of the preceding terms past cur_doc
                    query_terms[0]['ptr'] = self._advance_ptr(
                        query_terms[0]['postings'], query_terms[0]['ptr'], cur_doc + 1)
                else:
                    if query_terms[0]['postings'][query_terms[0]['ptr']] == pivot:
                        # success - all terms before pivot point to pivot
                        cur_doc = pivot
                        # compute full BM25 score for cur_doc
                        score = 0.0
                        for t in query_terms:
                            if t['postings'][t['ptr']] == cur_doc:
                                tf = t['tf_list'][t['ptr']]
                                dl = merged_index.doc_length[cur_doc]
                                tf_norm = ((k1 + 1) * tf) / (k1 * ((1 - b) + b * (dl / avgdl)) + tf)
                                score += t['idf'] * tf_norm
                                t['ptr'] += 1
                        # update top-k heap
                        if len(top_k) < k:
                            heapq.heappush(top_k, (score, cur_doc))
                            if len(top_k) == k:
                                threshold = top_k[0][0]
                        elif score > top_k[0][0]:
                            heapq.heapreplace(top_k, (score, cur_doc))
                            threshold = top_k[0][0]
                    else:
                        # advance one preceding term to pivot
                        query_terms[0]['ptr'] = self._advance_ptr(
                            query_terms[0]['postings'], query_terms[0]['ptr'], pivot)

            docs = [(score, self.doc_id_map[doc_id]) for (score, doc_id) in top_k]
            return sorted(docs, key=lambda x: x[0], reverse=True)
if __name__ == "__main__":

    BSBI_instance = BSBIIndex(data_dir = 'collection', \
                              postings_encoding = VBEPostings, \
                              output_dir = 'index')
    BSBI_instance.index() # memulai indexing!
