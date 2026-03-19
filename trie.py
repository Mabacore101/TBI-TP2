class TrieNode:
    """
    Representasi sebuah node dalam struktur data Trie.

    Attributes
    ----------
    children : dict
        Dictionary yang memetakan karakter ke TrieNode child.
    term_id : int or None
        ID yang diasosiasikan dengan term yang berakhir di node ini.
        None jika node ini bukan akhir dari sebuah term.
    """

    def __init__(self):
        self.children = {}
        self.term_id = None


class TrieIdMap:
    """
    Implementasi IdMap menggunakan struktur data Trie (Prefix Tree)
    sebagai pengganti hashtable dictionary biasa.

    Trie memungkinkan kompresi dictionary dengan cara berbagi prefix
    yang sama antar term. Misalnya, "inform", "informed", "information"
    berbagi path i→n→f→o→r→m di dalam Trie.

    Interface kelas ini sama dengan IdMap di util.py sehingga bisa
    digunakan sebagai drop-in replacement tanpa mengubah kode lain.

    Attributes
    ----------
    root : TrieNode
        Root node dari Trie.
    id_to_str : List[str]
        List untuk mapping ID ke string term.
        Digunakan untuk lookup sebaliknya (ID → term).
    """

    def __init__(self):
        self.root = TrieNode()
        self.id_to_str = []

    def __len__(self):
        """Mengembalikan banyaknya term yang tersimpan di TrieIdMap."""
        return len(self.id_to_str)

    def _insert(self, word):
        """
        Menyisipkan sebuah term ke dalam Trie dan mengassign ID baru.
        Jika term sudah ada, langsung kembalikan ID yang sudah ada.

        Parameters
        ----------
        word : str
            Term yang akan disisipkan.

        Returns
        -------
        int
            ID yang diasosiasikan dengan term.
        """
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        if node.term_id is None:
            node.term_id = len(self.id_to_str)
            self.id_to_str.append(word)
        return node.term_id

    def _search(self, word):
        """
        Mencari sebuah term di dalam Trie dan mengembalikan ID-nya.
        Jika term tidak ditemukan, assign ID baru.

        Parameters
        ----------
        word : str
            Term yang akan dicari.

        Returns
        -------
        int
            ID yang diasosiasikan dengan term.
        """
        node = self.root
        for char in word:
            if char not in node.children:
                return self._insert(word)
            node = node.children[char]
        if node.term_id is None:
            return self._insert(word)
        return node.term_id

    def _get_str(self, i):
        """
        Mengembalikan string term yang terasosiasi dengan ID i.

        Parameters
        ----------
        i : int
            ID term yang akan dicari.

        Returns
        -------
        str
            String term yang terasosiasi dengan ID i.
        """
        return self.id_to_str[i]

    def __getitem__(self, key):
        """
        Mengizinkan akses elemen dengan syntax [..] seperti IdMap.

        Jika key adalah string, cari atau insert term dan kembalikan ID.
        Jika key adalah integer, kembalikan string term yang terasosiasi.

        Parameters
        ----------
        key : str or int
            Term string atau term ID.

        Returns
        -------
        int or str
            ID jika key adalah string, string jika key adalah integer.
        """
        if type(key) is int:
            return self._get_str(key)
        elif type(key) is str:
            return self._search(key)
        else:
            raise TypeError


if __name__ == '__main__':
    # test TrieIdMap
    trie = TrieIdMap()

    doc = ["halo", "semua", "selamat", "pagi", "semua"]
    assert [trie[term] for term in doc] == [0, 1, 2, 3, 1], "term_id salah"
    assert trie[1] == "semua", "term_id salah"
    assert trie[0] == "halo", "term_id salah"
    assert trie["selamat"] == 2, "term_id salah"
    assert trie["pagi"] == 3, "term_id salah"
    print("semua assertions passed!")

    # test prefix sharing
    words = ["inform", "informed", "information", "informing"]
    trie2 = TrieIdMap()
    for word in words:
        trie2[word]
    assert trie2["inform"] == 0
    assert trie2["informed"] == 1
    assert trie2["information"] == 2
    assert trie2["informing"] == 3
    assert len(trie2) == 4
    print("prefix sharing test passed!")