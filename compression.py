import array
import math

class StandardPostings:
    """ 
    Class dengan static methods, untuk mengubah representasi postings list
    yang awalnya adalah List of integer, berubah menjadi sequence of bytes.
    Kita menggunakan Library array di Python.

    ASUMSI: postings_list untuk sebuah term MUAT di memori!

    Silakan pelajari:
        https://docs.python.org/3/library/array.html
    """

    @staticmethod
    def encode(postings_list):
        """
        Encode postings_list menjadi stream of bytes

        Parameters
        ----------
        postings_list: List[int]
            List of docIDs (postings)

        Returns
        -------
        bytes
            bytearray yang merepresentasikan urutan integer di postings_list
        """
        # Untuk yang standard, gunakan L untuk unsigned long, karena docID
        # tidak akan negatif. Dan kita asumsikan docID yang paling besar
        # cukup ditampung di representasi 4 byte unsigned.
        return array.array('L', postings_list).tobytes()

    @staticmethod
    def decode(encoded_postings_list):
        """
        Decodes postings_list dari sebuah stream of bytes

        Parameters
        ----------
        encoded_postings_list: bytes
            bytearray merepresentasikan encoded postings list sebagai keluaran
            dari static method encode di atas.

        Returns
        -------
        List[int]
            list of docIDs yang merupakan hasil decoding dari encoded_postings_list
        """
        decoded_postings_list = array.array('L')
        decoded_postings_list.frombytes(encoded_postings_list)
        return decoded_postings_list.tolist()

    @staticmethod
    def encode_tf(tf_list):
        """
        Encode list of term frequencies menjadi stream of bytes

        Parameters
        ----------
        tf_list: List[int]
            List of term frequencies

        Returns
        -------
        bytes
            bytearray yang merepresentasikan nilai raw TF kemunculan term di setiap
            dokumen pada list of postings
        """
        return StandardPostings.encode(tf_list)

    @staticmethod
    def decode_tf(encoded_tf_list):
        """
        Decodes list of term frequencies dari sebuah stream of bytes

        Parameters
        ----------
        encoded_tf_list: bytes
            bytearray merepresentasikan encoded term frequencies list sebagai keluaran
            dari static method encode_tf di atas.

        Returns
        -------
        List[int]
            List of term frequencies yang merupakan hasil decoding dari encoded_tf_list
        """
        return StandardPostings.decode(encoded_tf_list)

class VBEPostings:
    """ 
    Berbeda dengan StandardPostings, dimana untuk suatu postings list,
    yang disimpan di disk adalah sequence of integers asli dari postings
    list tersebut apa adanya.

    Pada VBEPostings, kali ini, yang disimpan adalah gap-nya, kecuali
    posting yang pertama. Barulah setelah itu di-encode dengan Variable-Byte
    Enconding algorithm ke bytestream.

    Contoh:
    postings list [34, 67, 89, 454] akan diubah dulu menjadi gap-based,
    yaitu [34, 33, 22, 365]. Barulah setelah itu di-encode dengan algoritma
    compression Variable-Byte Encoding, dan kemudian diubah ke bytesream.

    ASUMSI: postings_list untuk sebuah term MUAT di memori!

    """

    @staticmethod
    def vb_encode_number(number):
        """
        Encodes a number using Variable-Byte Encoding
        Lihat buku teks kita!
        """
        bytes = []
        while True:
            bytes.insert(0, number % 128) # prepend ke depan
            if number < 128:
                break
            number = number // 128
        bytes[-1] += 128 # bit awal pada byte terakhir diganti 1
        return array.array('B', bytes).tobytes()

    @staticmethod
    def vb_encode(list_of_numbers):
        """ 
        Melakukan encoding (tentunya dengan compression) terhadap
        list of numbers, dengan Variable-Byte Encoding
        """
        bytes = []
        for number in list_of_numbers:
            bytes.append(VBEPostings.vb_encode_number(number))
        return b"".join(bytes)

    @staticmethod
    def encode(postings_list):
        """
        Encode postings_list menjadi stream of bytes (dengan Variable-Byte
        Encoding). JANGAN LUPA diubah dulu ke gap-based list, sebelum
        di-encode dan diubah ke bytearray.

        Parameters
        ----------
        postings_list: List[int]
            List of docIDs (postings)

        Returns
        -------
        bytes
            bytearray yang merepresentasikan urutan integer di postings_list
        """
        gap_postings_list = [postings_list[0]]
        for i in range(1, len(postings_list)):
            gap_postings_list.append(postings_list[i] - postings_list[i-1])
        return VBEPostings.vb_encode(gap_postings_list)

    @staticmethod
    def encode_tf(tf_list):
        """
        Encode list of term frequencies menjadi stream of bytes

        Parameters
        ----------
        tf_list: List[int]
            List of term frequencies

        Returns
        -------
        bytes
            bytearray yang merepresentasikan nilai raw TF kemunculan term di setiap
            dokumen pada list of postings
        """
        return VBEPostings.vb_encode(tf_list)

    @staticmethod
    def vb_decode(encoded_bytestream):
        """
        Decoding sebuah bytestream yang sebelumnya di-encode dengan
        variable-byte encoding.
        """
        n = 0
        numbers = []
        decoded_bytestream = array.array('B')
        decoded_bytestream.frombytes(encoded_bytestream)
        bytestream = decoded_bytestream.tolist()
        for byte in bytestream:
            if byte < 128:
                n = 128 * n + byte
            else:
                n = 128 * n + (byte - 128)
                numbers.append(n)
                n = 0
        return numbers

    @staticmethod
    def decode(encoded_postings_list):
        """
        Decodes postings_list dari sebuah stream of bytes. JANGAN LUPA
        bytestream yang di-decode dari encoded_postings_list masih berupa
        gap-based list.

        Parameters
        ----------
        encoded_postings_list: bytes
            bytearray merepresentasikan encoded postings list sebagai keluaran
            dari static method encode di atas.

        Returns
        -------
        List[int]
            list of docIDs yang merupakan hasil decoding dari encoded_postings_list
        """
        decoded_postings_list = VBEPostings.vb_decode(encoded_postings_list)
        total = decoded_postings_list[0]
        ori_postings_list = [total]
        for i in range(1, len(decoded_postings_list)):
            total += decoded_postings_list[i]
            ori_postings_list.append(total)
        return ori_postings_list

    @staticmethod
    def decode_tf(encoded_tf_list):
        """
        Decodes list of term frequencies dari sebuah stream of bytes

        Parameters
        ----------
        encoded_tf_list: bytes
            bytearray merepresentasikan encoded term frequencies list sebagai keluaran
            dari static method encode_tf di atas.

        Returns
        -------
        List[int]
            List of term frequencies yang merupakan hasil decoding dari encoded_tf_list
        """
        return VBEPostings.vb_decode(encoded_tf_list)

class EliasGammaPostings:
    """
    Implementasi kompresi Elias-Gamma Encoding untuk postings list.
    
    Seperti VBEPostings, postings list di-encode dalam bentuk gap-based
    sebelum di-encode dengan Elias-Gamma. TF list di-encode langsung
    tanpa konversi gap.
    
    Elias-Gamma Encoding:
        Untuk sebuah bilangan bulat positif N:
        1. Hitung k = floor(log2(N))
        2. Tulis k buah bit '0' (unary prefix)
        3. Tulis representasi biner dari N dalam k+1 bit
        
        Contoh: N = 6
            k = floor(log2(6)) = 2
            prefix = '00'
            binary of 6 = '110'
            hasil = '00110'
    """

    @staticmethod
    def _encode_number(number):
        """
        Encode sebuah bilangan bulat positif menggunakan Elias-Gamma Encoding.
        Mengembalikan string of bits ('0' dan '1').

        Parameters
        ----------
        number : int
            Bilangan bulat positif yang akan di-encode.
            Harus >= 1.

        Returns
        -------
        str
            String of bits hasil encoding Elias-Gamma.
        """
        if number == 0:
            raise ValueError("Elias-Gamma cannot encode 0")
        k = int(math.floor(math.log2(number)))
        unary_prefix = '0' * k
        binary_rep = format(number, f'0{k + 1}b')
        return unary_prefix + binary_rep

    @staticmethod
    def _decode_number(bit_string, pos):
        """
        Decode sebuah bilangan bulat dari bit_string mulai dari posisi pos,
        menggunakan Elias-Gamma Decoding.

        Parameters
        ----------
        bit_string : str
            String of bits yang akan di-decode.
        pos : int
            Posisi awal dalam bit_string untuk memulai decoding.

        Returns
        -------
        Tuple[int, int]
            Tuple berisi (decoded_number, new_position) dimana new_position
            adalah posisi setelah bit yang sudah di-decode.
        """
        k = 0
        while pos < len(bit_string) and bit_string[pos] == '0':
            k += 1
            pos += 1
        number = int(bit_string[pos:pos + k + 1], 2)
        return number, pos + k + 1

    @staticmethod
    def _bits_to_bytes(bit_string):
        """
        Konversi string of bits menjadi bytes.
        Padding '0' ditambahkan di bagian akhir jika panjang bit_string
        tidak kelipatan 8. Panjang original disimpan di byte pertama
        sebagai informasi padding.

        Parameters
        ----------
        bit_string : str
            String of bits yang akan dikonversi.

        Returns
        -------
        bytes
            Hasil konversi dalam bentuk bytes.
        """
        padding = (8 - len(bit_string) % 8) % 8
        bit_string = bit_string + '0' * padding
        # simpan jumlah padding di byte pertama
        result = bytes([padding])
        for i in range(0, len(bit_string), 8):
            result += bytes([int(bit_string[i:i + 8], 2)])
        return result

    @staticmethod
    def _bytes_to_bits(byte_data):
        """
        Konversi bytes kembali menjadi string of bits.
        Byte pertama dibaca sebagai informasi padding untuk
        membuang bit padding di akhir.

        Parameters
        ----------
        byte_data : bytes
            Data dalam bentuk bytes yang akan dikonversi.

        Returns
        -------
        str
            String of bits hasil konversi, tanpa padding bits.
        """
        padding = byte_data[0]
        bit_string = ''.join(format(byte, '08b') for byte in byte_data[1:])
        if padding > 0:
            bit_string = bit_string[:-padding]
        return bit_string

    @staticmethod
    def encode(postings_list):
        """
        Encode postings_list menjadi stream of bytes menggunakan
        Elias-Gamma Encoding. Postings list terlebih dahulu dikonversi
        ke gap-based representation sebelum di-encode.

        Parameters
        ----------
        postings_list : List[int]
            List of docIDs (postings) yang akan di-encode.

        Returns
        -------
        bytes
            Bytearray hasil encoding dari postings_list.
        """
        gap_list = [postings_list[0]]
        for i in range(1, len(postings_list)):
            gap_list.append(postings_list[i] - postings_list[i - 1])
        bit_string = ''.join(EliasGammaPostings._encode_number(n) for n in gap_list)
        return EliasGammaPostings._bits_to_bytes(bit_string)

    @staticmethod
    def decode(encoded_postings_list):
        """
        Decode postings_list dari stream of bytes hasil Elias-Gamma Encoding.
        Hasil decoding masih berupa gap-based list, sehingga perlu direkonstruksi
        kembali menjadi postings list original.

        Parameters
        ----------
        encoded_postings_list : bytes
            Bytearray hasil encoding dari postings_list.

        Returns
        -------
        List[int]
            List of docIDs hasil decoding.
        """
        bit_string = EliasGammaPostings._bytes_to_bits(encoded_postings_list)
        pos = 0
        gap_list = []
        while pos < len(bit_string):
            number, pos = EliasGammaPostings._decode_number(bit_string, pos)
            gap_list.append(number)
        total = gap_list[0]
        postings_list = [total]
        for i in range(1, len(gap_list)):
            total += gap_list[i]
            postings_list.append(total)
        return postings_list

    @staticmethod
    def encode_tf(tf_list):
        """
        Encode list of term frequencies menjadi stream of bytes menggunakan
        Elias-Gamma Encoding. TF list di-encode langsung tanpa konversi gap.

        Parameters
        ----------
        tf_list : List[int]
            List of term frequencies yang akan di-encode.

        Returns
        -------
        bytes
            Bytearray hasil encoding dari tf_list.
        """
        bit_string = ''.join(EliasGammaPostings._encode_number(n) for n in tf_list)
        return EliasGammaPostings._bits_to_bytes(bit_string)

    @staticmethod
    def decode_tf(encoded_tf_list):
        """
        Decode list of term frequencies dari stream of bytes hasil
        Elias-Gamma Encoding.

        Parameters
        ----------
        encoded_tf_list : bytes
            Bytearray hasil encoding dari tf_list.

        Returns
        -------
        List[int]
            List of term frequencies hasil decoding.
        """
        bit_string = EliasGammaPostings._bytes_to_bits(encoded_tf_list)
        pos = 0
        tf_list = []
        while pos < len(bit_string):
            number, pos = EliasGammaPostings._decode_number(bit_string, pos)
            tf_list.append(number)
        return tf_list

if __name__ == '__main__':
    
    postings_list = [34, 67, 89, 454, 2345738]
    tf_list = [12, 10, 3, 4, 1]
    for Postings in [StandardPostings, VBEPostings, EliasGammaPostings]:
        print(Postings.__name__)
        encoded_postings_list = Postings.encode(postings_list)
        encoded_tf_list = Postings.encode_tf(tf_list)
        print("byte hasil encode postings: ", encoded_postings_list)
        print("ukuran encoded postings   : ", len(encoded_postings_list), "bytes")
        print("byte hasil encode TF list : ", encoded_tf_list)
        print("ukuran encoded TF list    : ", len(encoded_tf_list), "bytes")
        
        decoded_posting_list = Postings.decode(encoded_postings_list)
        decoded_tf_list = Postings.decode_tf(encoded_tf_list)
        print("hasil decoding (postings): ", decoded_posting_list)
        print("hasil decoding (TF list) : ", decoded_tf_list)
        assert decoded_posting_list == postings_list, "hasil decoding tidak sama dengan postings original"
        assert decoded_tf_list == tf_list, "hasil decoding tidak sama dengan postings original"
        print()
