import os
import sys
import contextlib

from bsbi import BSBIIndex
from index import InvertedIndexReader, InvertedIndexWriter
from compression import VBEPostings
from tqdm import tqdm


class SPIMIIndex(BSBIIndex):
    """
    Implementasi Search Engine dengan skema SPIMI (Single-Pass In-Memory Indexing).
    
    Berbeda dengan BSBI yang mengumpulkan semua term-doc pairs terlebih dahulu
    lalu melakukan sorting global, SPIMI langsung membangun dictionary (hashtable)
    yang memetakan term ke postings list secara langsung. Sorting hanya dilakukan
    pada level block, bukan secara global.
    
    SPIMIIndex mewarisi BSBIIndex dan meng-override method index() dengan
    logika SPIMI. Semua method retrieval dan evaluasi tetap sama.

    Attributes
    ----------
    memory_threshold : int
        Batas jumlah token yang diproses sebelum block ditulis ke disk.
        Digunakan sebagai aproksimasi penggunaan memori.
    """

    def __init__(self, data_dir, output_dir, postings_encoding,
                 index_name="main_index", memory_threshold=100000):
        """
        Parameters
        ----------
        data_dir : str
            Path ke data collection.
        output_dir : str
            Path ke output index files.
        postings_encoding : class
            Encoding yang digunakan untuk postings list.
        index_name : str
            Nama dari file yang berisi inverted index.
        memory_threshold : int
            Batas jumlah token per block sebelum ditulis ke disk.
        """
        super().__init__(data_dir, output_dir, postings_encoding, index_name)
        self.memory_threshold = memory_threshold

    def spimi_invert(self, token_stream, index_id):
        """
        Implementasi SPIMI-INVERT dari pseudocode di literatur.
        Membangun dictionary langsung dari token stream dan menulis
        block ke disk ketika memory threshold tercapai.

        Parameters
        ----------
        token_stream : generator
            Generator yang menghasilkan pasangan (term, doc_path) satu per satu.
        index_id : str
            Nama identifier untuk intermediate index block.

        Returns
        -------
        str
            index_id dari block yang sudah ditulis ke disk.
        """
        dictionary = {}  # term_id -> list of (doc_id, tf)
        token_count = 0

        for term, doc_path in token_stream:
            term_id = self.term_id_map[term]
            doc_id = self.doc_id_map[doc_path]

            if term_id not in dictionary:
                dictionary[term_id] = {}
            if doc_id not in dictionary[term_id]:
                dictionary[term_id][doc_id] = 0
            dictionary[term_id][doc_id] += 1

            token_count += 1
            if token_count >= self.memory_threshold:
                self._write_block_to_disk(dictionary, index_id)
                return index_id

        # write remaining tokens to disk
        self._write_block_to_disk(dictionary, index_id)
        return index_id

    def _write_block_to_disk(self, dictionary, index_id):
        """
        Menulis block dictionary ke disk sebagai intermediate index.
        Terms diurutkan sebelum ditulis ke disk.

        Parameters
        ----------
        dictionary : dict
            Dictionary yang memetakan term_id ke dict of {doc_id: tf}.
        index_id : str
            Nama identifier untuk intermediate index block.
        """
        with InvertedIndexWriter(index_id, self.postings_encoding,
                                 directory=self.output_dir) as index:
            for term_id in sorted(dictionary.keys()):
                sorted_doc_ids = sorted(dictionary[term_id].keys())
                tf_list = [dictionary[term_id][doc_id] for doc_id in sorted_doc_ids]
                upper_bound = max(tf_list)
                index.append(term_id, sorted_doc_ids, tf_list, upper_bound)

    def _token_stream(self, block_dir_relative):
        """
        Generator yang menghasilkan pasangan (term, doc_path) satu per satu
        dari semua dokumen dalam sebuah block directory.

        Parameters
        ----------
        block_dir_relative : str
            Relative path ke directory block.

        Yields
        ------
        Tuple[str, str]
            Pasangan (term, doc_path) untuk setiap token dalam block.
        """
        dir_path = "./" + self.data_dir + "/" + block_dir_relative
        for filename in next(os.walk(dir_path))[2]:
            doc_path = dir_path + "/" + filename
            with open(doc_path, "r", encoding="utf8", errors="surrogateescape") as f:
                for token in f.read().split():
                    yield token, doc_path

    def index(self):
        """
        Melakukan indexing dengan skema SPIMI (Single-Pass In-Memory Indexing).
        
        Berbeda dengan BSBI, SPIMI tidak mengumpulkan semua term-doc pairs
        terlebih dahulu. Sebaliknya, SPIMI langsung membangun dictionary
        per block dan menulis ke disk ketika memory threshold tercapai.
        Proses merge tetap sama dengan BSBI.
        """
        for block_dir_relative in tqdm(sorted(next(os.walk(self.data_dir))[1])):
            index_id = 'intermediate_index_' + block_dir_relative
            self.intermediate_indices.append(index_id)
            token_stream = self._token_stream(block_dir_relative)
            self.spimi_invert(token_stream, index_id)

        self.save()

        with InvertedIndexWriter(self.index_name, self.postings_encoding,
                                 directory=self.output_dir) as merged_index:
            with contextlib.ExitStack() as stack:
                indices = [stack.enter_context(
                    InvertedIndexReader(index_id, self.postings_encoding,
                                       directory=self.output_dir))
                    for index_id in self.intermediate_indices]
                self.merge(indices, merged_index)


if __name__ == "__main__":
    SPIMI_instance = SPIMIIndex(data_dir='collection',
                                postings_encoding=VBEPostings,
                                output_dir='index',
                                memory_threshold=100000)
    SPIMI_instance.index()