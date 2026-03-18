import re
import math
from bsbi import BSBIIndex
from compression import VBEPostings

######## >>>>> sebuah IR metric: RBP p = 0.8

def rbp(ranking, p = 0.8):
  """ menghitung search effectiveness metric score dengan 
      Rank Biased Precision (RBP)

      Parameters
      ----------
      ranking: List[int]
         vektor biner seperti [1, 0, 1, 1, 1, 0]
         gold standard relevansi dari dokumen di rank 1, 2, 3, dst.
         Contoh: [1, 0, 1, 1, 1, 0] berarti dokumen di rank-1 relevan,
                 di rank-2 tidak relevan, di rank-3,4,5 relevan, dan
                 di rank-6 tidak relevan
        
      Returns
      -------
      Float
        score RBP
  """
  score = 0.
  for i in range(1, len(ranking)):
    pos = i - 1
    score += ranking[pos] * (p ** (i - 1))
  return (1 - p) * score

def dcg(ranking, p=10):
    """
    Menghitung Discounted Cumulative Gain (DCG) pada posisi p.
    
    DCG mengakumulasi relevansi dokumen dengan diskon logaritmik
    berdasarkan posisi rank. Dokumen relevan di posisi lebih tinggi
    berkontribusi lebih besar terhadap score.

    Parameters
    ----------
    ranking : List[int]
        Vektor biner seperti [1, 0, 1, 1, 1, 0]
        gold standard relevansi dari dokumen di rank 1, 2, 3, dst.
    p : int
        Posisi cutoff untuk perhitungan DCG.

    Returns
    -------
    float
        Score DCG pada posisi p.
    """
    score = 0.0
    for i in range(1, min(p, len(ranking)) + 1):
        score += ranking[i - 1] / math.log2(i + 1)
    return score


def ndcg(ranking, p=10):
    """
    Menghitung Normalized Discounted Cumulative Gain (NDCG) pada posisi p.
    
    NDCG = DCG / IDCG, dimana IDCG adalah DCG ideal (ranking terbaik
    yang mungkin). Nilai NDCG berada di antara 0 dan 1.

    Parameters
    ----------
    ranking : List[int]
        Vektor biner seperti [1, 0, 1, 1, 1, 0]
        gold standard relevansi dari dokumen di rank 1, 2, 3, dst.
    p : int
        Posisi cutoff untuk perhitungan NDCG.

    Returns
    -------
    float
        Score NDCG pada posisi p.
    """
    ideal_ranking = sorted(ranking, reverse=True)
    idcg = dcg(ideal_ranking, p)
    if idcg == 0:
        return 0.0
    return dcg(ranking, p) / idcg


def ap(ranking):
    """
    Menghitung Average Precision (AP).
    
    AP menghitung rata-rata nilai precision pada setiap posisi
    dimana dokumen relevan ditemukan. Metrik ini memberikan reward
    untuk menemukan dokumen relevan lebih awal dalam ranking.

    Parameters
    ----------
    ranking : List[int]
        Vektor biner seperti [1, 0, 1, 1, 1, 0]
        gold standard relevansi dari dokumen di rank 1, 2, 3, dst.

    Returns
    -------
    float
        Score Average Precision.
    """
    score = 0.0
    relevant_count = 0
    for i in range(1, len(ranking) + 1):
        if ranking[i - 1] == 1:
            relevant_count += 1
            score += relevant_count / i
    if relevant_count == 0:
        return 0.0
    return score / relevant_count

######## >>>>> memuat qrels

def load_qrels(qrel_file = "qrels.txt", max_q_id = 30, max_doc_id = 1033):
  """ memuat query relevance judgment (qrels) 
      dalam format dictionary of dictionary
      qrels[query id][document id]

      dimana, misal, qrels["Q3"][12] = 1 artinya Doc 12
      relevan dengan Q3; dan qrels["Q3"][10] = 0 artinya
      Doc 10 tidak relevan dengan Q3.

  """
  qrels = {"Q" + str(i) : {i:0 for i in range(1, max_doc_id + 1)} \
                 for i in range(1, max_q_id + 1)}
  with open(qrel_file) as file:
    for line in file:
      parts = line.strip().split()
      qid = parts[0]
      did = int(parts[1])
      qrels[qid][did] = 1
  return qrels

######## >>>>> EVALUASI !

def eval(qrels, query_file="queries.txt", k=1000, scoring_method="tfidf"):
    """
    Loop ke semua 30 query, hitung score di setiap query,
    lalu hitung MEAN SCORE over those 30 queries.
    Untuk setiap query, kembalikan top-1000 documents.

    Parameters
    ----------
    qrels : dict
        Query relevance judgments.
    query_file : str
        Path ke file yang berisi daftar query.
    k : int
        Banyaknya dokumen teratas yang dikembalikan per query.
    scoring_method : str
        Metode scoring yang digunakan, either "tfidf" or "bm25".
    """
    BSBI_instance = BSBIIndex(data_dir='collection',
                              postings_encoding=VBEPostings,
                              output_dir='index')
    with open(query_file) as file:
        rbp_scores = []
        dcg_scores = []
        ndcg_scores = []
        ap_scores = []
        for qline in file:
            parts = qline.strip().split()
            qid = parts[0]
            query = " ".join(parts[1:])
            ranking = []
            if scoring_method == "bm25":
                results = BSBI_instance.retrieve_bm25(query, k=k)
            elif scoring_method == "wand":
                results = BSBI_instance.retrieve_bm25_wand(query, k=k)
            else:
                results = BSBI_instance.retrieve_tfidf(query, k=k)
            for (score, doc) in results:
                did = int(re.search(r'\/.*\/.*\/(.*)\.txt', doc).group(1))
                ranking.append(qrels[qid][did])
            rbp_scores.append(rbp(ranking))
            dcg_scores.append(dcg(ranking))
            ndcg_scores.append(ndcg(ranking))
            ap_scores.append(ap(ranking))

    print(f"Hasil evaluasi {scoring_method.upper()} terhadap 30 queries")
    print("RBP  score =", sum(rbp_scores) / len(rbp_scores))
    print("DCG  score =", sum(dcg_scores) / len(dcg_scores))
    print("NDCG score =", sum(ndcg_scores) / len(ndcg_scores))
    print("AP   score =", sum(ap_scores) / len(ap_scores))

if __name__ == '__main__':
    qrels = load_qrels()
    assert qrels["Q1"][166] == 1, "qrels salah"
    assert qrels["Q1"][300] == 0, "qrels salah"
    eval(qrels, scoring_method="tfidf")
    eval(qrels, scoring_method="bm25")
    eval(qrels, scoring_method="wand")