# snack-recommendation
Proyek ini dibuat untuk memenuhi UAS Data Mining 2021
Demo: https://share.streamlit.io/galangfs/snack-recommendation/main/main.py

1.	Perekenalan
Asalamualaikum wr.wb
Perkenalkan kami dari kelompok 7, pada kesempatan kali ini kami ingin mendemonstrasikan hasil kerja kelompok kami yaitu sistem rekomendasi dengan menggunakan NCF (neural collaborative filtering) yang memiliki 2 komponen yaitu GMF dan MLP dengan keunggulan sebagai berikut :
a.	GMF yang menerapkan kernel linier untuk memodelkan interaksi item pengguna seperti MF murni, dan
b.	MLP yang menggunakan banyak lapisan saraf untuk melapisi interaksi nonlinier
Sistem rekomendasi ini bertujuan memberikan rekomendasi makanan dan genre makanan kepada pembeli berdasarkan data emplisit yang diperoleh dari dataset snact_data dan rating_snack.

2.	Menginstall library python yang diperlukan
Sebelum menjalankan Sistem Rekomendasi Neural Collaborative filtering, Install terlebih dahulu library yang diperlukan dengan cara:
pip install -r requirements.txt

3.	Menjalankana sistem rekomendasi
Setelah menginstall library yang diperlukan, kemudian jalankan sistem rekomendasi menggunakan CMD berdasarkan lokasi direktori penyimpanan folder
Setelah cmd terbuka, ketik: streamlit run main.py

4.	Penjelasan Sistem rekomendasi
Snack NeuralMF Hybrid Recommender
Pada sistem rekomendasi ini, kami menerapkan model rekomendasi dengan dataset Rekomendasi Makanan Ringan. Dataset dibuat dengan Faker dan inspirasi diambil dari beberapa dataset rekomendasi film NCF tentang cara membuat dataset kami
Model ini disusun berdasarkan makalah Neural Collaborative Filtering: Xiangnan He, Lizi Liao, Hanwang Zhang, Liqiang Nie, Xia Hu dan Tat-Seng Chua (2017). Penyaringan Kolaboratif Saraf. Dalam Prosiding WWW '17, Perth, Australia, 03-07 April 2017.
Berikut ini adalah sedikit penjelesan untuk sistem Hybrid Recommender.
Terdapat dua jenis utama sistem pemberi rekomendasi: Berbasis konten atau Content-Based dan Berbasis penyaringan kolaboratif atau Collaborative Filtering.
a.	Pemberi rekomendasi Content-Based menyarankan pilihan serupa untuk _item_ tertentu (makanan ringan dalam kasus kami), memberi tahu pengguna tentang item serupa dengan yang telah mereka tonton/beri peringkat positif. Metode ini biasanya menggunakan fitur item bersama dengan metode unsupervised dalam upaya menghasilkan ruang produk dan menghitung kesamaan antar item. Namun, metode ini mungkin berakhir dengan menyarankan campuran item yang terbatas, memberikan _faktor kejutan_ yang rendah bagi pengguna.
b.	Di sisi lain, pemberi rekomendasi pemfilteran kolaboratif mengandalkan riwayat pengguna sebelumnya dari item yang ditonton/diberi peringkat, meningkatkan kemungkinan merekomendasikan item secara kebetulan atau serendipitous kepada pengguna target. Metode klasik hanya mengandalkan matriks item-pengguna, yang memetakan interaksi yang dimiliki semua pengguna dengan setiap item. Metode matriks ini sangat intensif memori dan berbasis jaringan saraf yang lebih baru lebih umum. Meskipun demikian, metode ini dapat melewatkan item serupa -tetapi biasanya supervised-, dibandingkan dengan yang ditonton/diulas oleh pengguna target.
Untuk mendapatkan rekomendasi yang lebih kuat, model hybrid dapat menggabungkan fitur item dan fitur item pengguna.
NeuralMF adalah campuran dari rekomendasi General Matrix Factorization (GMF) dan Multi Layer Perceptron (MLP), menyerupai model Wide&Deep, memiliki kekuatan generalisasi yang tinggi. Dan juga, jaring saraf membuat lebih mudah untuk menangani volume data yang besar!

