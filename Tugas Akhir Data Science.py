# Import Library
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import FancyBboxPatch
from matplotlib.gridspec import GridSpec
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np
import warnings
import streamlit as st # Import Streamlit

warnings.filterwarnings('ignore')

# Konfigurasi Visual (ini akan dieksekusi sekali saat aplikasi dimulai)
# Pengaturan ini berlaku secara global untuk semua plot Matplotlib
plt.style.use('default')
CORPORATE_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2"]
sns.set_palette(CORPORATE_COLORS)

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'axes.titleweight': 'bold',
    'axes.titlesize': 16,
    'axes.labelsize': 13,
    'legend.fontsize': 12,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'figure.facecolor': 'white'
})

# Dataset Aktual PT TIMAH TBK (2022-2024)
# Menggunakan cache Streamlit untuk data agar tidak di-load ulang setiap kali ada interaksi
@st.cache_data
def load_data():
    data = {
        'Tahun': [2022, 2023, 2024],
        'Produksi Timah (ton)': [19825, 15340, 15500],
        'Penjualan Timah (ton)': [20805, 15000, 16000],
        'Pendapatan (Rp Triliun)': [12.5, 8.39, 9.2],
        'Laba Bersih (Rp Triliun)': [1.04, -0.45, 0.3],
        'Biaya Eksplorasi (Rp Miliar)': [35, 38, 42],
        'Cadangan (ribu ton)': [330, 320, 312.5],
        'Sumber Daya (ribu ton)': [810, 805, 807.2]
    }
    df = pd.DataFrame(data)

    # Kalkulasi Perubahan Persentase
    for col in ['Penjualan Timah (ton)', 'Pendapatan (Rp Triliun)', 'Laba Bersih (Rp Triliun)']:
        df[f'Perubahan {col.split()[0]} (%)'] = df[col].pct_change() * 100
    return df

df = load_data()

# Fungsi untuk membuat Grafik 1: Tren Produksi & Penjualan
def plot_produksi_penjualan(df):
    # Ukuran figure disesuaikan untuk tampil dalam kolom Streamlit
    fig, ax1 = plt.subplots(figsize=(8, 5)) 

    line1 = ax1.plot(df['Tahun'], df['Produksi Timah (ton)'],
                     marker='o', markersize=8, linewidth=3,
                     label='Produksi Timah', color=CORPORATE_COLORS[0])
    line2 = ax1.plot(df['Tahun'], df['Penjualan Timah (ton)'],
                     marker='s', markersize=8, linewidth=3,
                     label='Penjualan Timah', color=CORPORATE_COLORS[1])

    # Anotasi nilai pada setiap titik
    for i, row in df.iterrows():
        ax1.annotate(f"{row['Produksi Timah (ton)']:,}",
                     (row['Tahun'], row['Produksi Timah (ton)']),
                     textcoords="offset points", xytext=(0,10), # Sesuaikan posisi
                     ha='center', fontsize=11, fontweight='bold')
        ax1.annotate(f"{row['Penjualan Timah (ton)']:,}",
                     (row['Tahun'], row['Penjualan Timah (ton)']),
                     textcoords="offset points", xytext=(0,-20), # Sesuaikan posisi
                     ha='center', fontsize=11, fontweight='bold')

    ax1.set_title("Tren Produksi vs Penjualan Timah", pad=15)
    ax1.set_xlabel("Tahun")
    ax1.set_ylabel("Volume (ton)")
    ax1.legend(loc='upper right', frameon=True, shadow=True)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(12000, 22000)
    
    # Caption dipindahkan ke luar fungsi plot, akan ditampilkan oleh Streamlit
    return fig

# Fungsi untuk membuat Grafik 2: Distribusi Pasar Ekspor
def plot_ekspor_distribusi():
    fig, ax2 = plt.subplots(figsize=(7, 7)) # Ukuran figure disesuaikan

    ekspor_data = {
        "Korea Selatan": 19, "Singapura": 18, "Jepang": 12,
        "Belanda": 12, "India": 10, "China": 7, "Domestik": 12
    }
    colors = plt.cm.Set3(np.linspace(0, 1, len(ekspor_data)))
    wedges, texts, autotexts = ax2.pie(ekspor_data.values(),
                                           labels=ekspor_data.keys(),
                                           autopct='%1.1f%%',
                                           startangle=90,
                                           colors=colors,
                                           pctdistance=0.85, # Menyesuaikan posisi persentase
                                           textprops={'fontsize': 11})

    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(11)
    for text in texts:
        text.set_fontsize(10)
        text.set_color('black')

    ax2.set_title("Distribusi Pasar Ekspor Timah 2024", pad=20)
    ax2.axis('equal') # Memastikan pie chart berbentuk lingkaran
    
    # Caption dipindahkan
    return fig

# Fungsi untuk membuat Grafik 3: Kinerja Keuangan
def plot_kinerja_keuangan(df):
    fig, ax3 = plt.subplots(figsize=(8, 5)) # Ukuran figure disesuaikan
    bars1 = ax3.bar(df['Tahun'] - 0.2, df['Pendapatan (Rp Triliun)'], # Geser sedikit ke kiri
                    alpha=0.7, label='Pendapatan', color=CORPORATE_COLORS[2], width=0.4)
    ax3b = ax3.twinx()
    bars2 = ax3b.bar(df['Tahun'] + 0.2, df['Laba Bersih (Rp Triliun)'], # Geser sedikit ke kanan
                    alpha=0.8, label='Laba Bersih', color=CORPORATE_COLORS[3], width=0.4)

    for i, (rev, profit) in enumerate(zip(df['Pendapatan (Rp Triliun)'], df['Laba Bersih (Rp Triliun)'])):
        ax3.annotate(f"Rp {rev:.1f}T", (df['Tahun'].iloc[i] - 0.2, rev),
                     textcoords="offset points", xytext=(0,8),
                     ha='center', fontsize=11, fontweight='bold')
        ax3b.annotate(f"Rp {profit:.2f}T", (df['Tahun'].iloc[i] + 0.2, profit),
                      textcoords="offset points", xytext=(0,8 if profit>=0 else -18),
                      ha='center', fontsize=11, fontweight='bold',
                      color='red' if profit < 0 else 'black')

    ax3.set_title("Kinerja Keuangan PT Timah", pad=20)
    ax3.set_xlabel("Tahun")
    ax3.set_ylabel("Pendapatan (Rp Triliun)", color=CORPORATE_COLORS[2])
    ax3b.set_ylabel("Laba Bersih (Rp Triliun)", color=CORPORATE_COLORS[3])
    ax3.legend(loc='upper left')
    ax3b.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)
    
    # Caption dipindahkan
    return fig

# Fungsi untuk membuat Grafik 4: Cadangan & Investasi Eksplorasi
def plot_cadangan_eksplorasi(df):
    fig, ax4 = plt.subplots(figsize=(8, 5)) # Ukuran figure disesuaikan

    line_cadangan = ax4.plot(df['Tahun'], df['Cadangan (ribu ton)'],
                              marker='D', markersize=10, linewidth=3,
                              label='Cadangan Timah', color=CORPORATE_COLORS[4])
    ax4b = ax4.twinx()
    bars_eksplorasi = ax4b.bar(df['Tahun'], df['Biaya Eksplorasi (Rp Miliar)'],
                               alpha=0.6, label='Biaya Eksplorasi',
                               color=CORPORATE_COLORS[5], width=0.5)

    for i, row in df.iterrows():
        ax4.annotate(f"{row['Cadangan (ribu ton)']:.1f}K ton",
                     (row['Tahun'], row['Cadangan (ribu ton)']),
                     textcoords="offset points", xytext=(0,15),
                     ha='center', fontsize=11, fontweight='bold')
        ax4b.annotate(f"Rp {int(row['Biaya Eksplorasi (Rp Miliar)'])}M",
                      (row['Tahun'], row['Biaya Eksplorasi (Rp Miliar)']),
                      textcoords="offset points", xytext=(0,8),
                      ha='center', fontsize=11, fontweight='bold')

    ax4.set_title("Cadangan Timah & Investasi Eksplorasi", pad=20)
    ax4.set_xlabel("Tahun")
    ax4.set_ylabel("Cadangan (ribu ton)", color=CORPORATE_COLORS[4])
    ax4b.set_ylabel("Biaya Eksplorasi (Rp Miliar)", color=CORPORATE_COLORS[5])
    ax4.legend(loc='upper left')
    ax4b.legend(loc='upper right')
    ax4.grid(True, alpha=0.3)
    ax4b.set_ylim(30, 45) # Menyesuaikan batas y untuk eksplorasi agar tidak terlalu sempit
    
    # Caption dipindahkan
    return fig

# Fungsi untuk membuat Grafik 5: Perubahan Kinerja (2023 vs 2022)
def plot_perubahan_kinerja(df):
    fig, ax5 = plt.subplots(figsize=(8, 5)) # Ukuran figure disesuaikan
    change_2023 = [df.iloc[1]['Perubahan Penjualan (%)'],
                    df.iloc[1]['Perubahan Pendapatan (%)'],
                    df.iloc[1]['Perubahan Laba (%)']]
    change_labels = ['Penjualan', 'Pendapatan', 'Laba/Rugi']

    bar_colors = ['red' if x < 0 else 'green' for x in change_2023]
    bars = ax5.barh(change_labels, change_2023, color=bar_colors, alpha=0.7)

    for i, v in enumerate(change_2023):
        ha = 'left' if v >= 0 else 'right'
        offset = 8 if v >= 0 else -8
        ax5.text(v + offset, i, f'{v:.1f}%',
                 ha=ha, va='center', fontweight='bold', fontsize=12)

    ax5.set_title('Perubahan Kinerja 2023 vs 2022', pad=20)
    ax5.set_xlabel('Perubahan (%)')
    ax5.axvline(0, color='black', linewidth=1)
    ax5.grid(True, axis='x', alpha=0.3)
    ax5.set_xlim(-150, 10) # Atur batas x agar anotasi tidak keluar
    
    # Caption dipindahkan
    return fig

# Fungsi untuk membuat Grafik 6: Proyeksi Produksi 2025
def plot_proyeksi_produksi(df):
    fig, ax6 = plt.subplots(figsize=(8, 5)) # Ukuran figure disesuaikan

    X = df['Tahun'].values.reshape(-1, 1)
    y = df['Produksi Timah (ton)'].values
    model = LinearRegression().fit(X, y)
    prediksi_2025 = model.predict([[2025]])[0]

    tahun_extended = list(df['Tahun']) + [2025]
    X_extended = np.array(tahun_extended).reshape(-1, 1)
    prediksi_extended = model.predict(X_extended)

    ax6.plot(df['Tahun'], y, marker='o', markersize=10, linewidth=3,
              label='Data Historis', color=CORPORATE_COLORS[0])
    ax6.plot(2025, prediksi_2025, 'ro', markersize=12,
              label=f'Prediksi 2025: {prediksi_2025:,.0f} ton')
    ax6.plot(tahun_extended, prediksi_extended,
              linestyle='--', color='gray', alpha=0.7, label='Tren Linear')

    for i, val in enumerate(y):
        ax6.annotate(f"{val:,}", (df['Tahun'].iloc[i], val),
                     textcoords="offset points", xytext=(0,10),
                     ha='center', fontsize=11, fontweight='bold')
    ax6.annotate(f"{prediksi_2025:,.0f}", (2025, prediksi_2025),
                 textcoords="offset points", xytext=(0,10),
                 ha='center', fontsize=11, fontweight='bold', color='red')

    ax6.set_title("Proyeksi Produksi Timah 2025", pad=20)
    ax6.set_xlabel("Tahun")
    ax6.set_ylabel("Produksi (ton)")
    ax6.legend(loc='lower left')
    ax6.grid(True, alpha=0.3)
    
    # Caption dipindahkan
    return fig, prediksi_2025

# Fungsi untuk bagian Analisis Strategis & Rekomendasi
def display_strategi_rekomendasi():
    st.markdown("---")
    st.markdown("<h2 style='text-align: center; color: #2D7DD2;'>ANALISIS STRATEGIS & REKOMENDASI PT TIMAH TBK</h2>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center; color: #5C9E47;'>Rangkuman Eksekutif & Strategi Pemulihan 2025</h3>", unsafe_allow_html=True)
    st.markdown("---")

    # Menggunakan expander untuk setiap bagian agar lebih ringkas
    with st.expander("📊 **PERFORMA 2022-2024**"):
        st.markdown("""
        * **Produksi:** 19,825 -> 15,500 ton (**-21.8%**)
        * **Penjualan:** 20,805 -> 16,000 ton (**-23.1%**)
        * **Pendapatan:** Rp 12.5T -> Rp 9.2T (**-26.4%**)
        * **Laba:** Rp 1.04T -> Rp 0.3T (**-71.2%**)
        """)
        st.markdown("**TANTANGAN UTAMA:**")
        st.write("""
        * Volatilitas harga timah global
        * Peningkatan biaya operasional
        * Penurunan cadangan (-5.3%)
        * Ketergantungan pasar ekspor (88%)
        """)

    with st.expander("🚀 **STRATEGI PEMULIHAN 2025**"):
        st.markdown("**DIVERSIFIKASI PRODUK:**")
        st.write("""
        * Tingkatkan produk olahan bernilai tambah
        * Kembangkan produk turunan elektronik
        """)
        st.markdown("**EFISIENSI OPERASIONAL:**")
        st.write("""
        * Target penghematan biaya 15%
        * Teknologi penambangan modern
        * Optimasi *supply chain*
        """)
        st.markdown("**EKSPANSI PASAR:**")
        st.write("""
        * Penetrasi pasar India (+15%)
        * Pengembangan pasar domestik
        * Diversifikasi regional ekspor
        """)

    with st.expander("🎯 **TARGET & PROYEKSI 2025**"):
        st.markdown("**SASARAN OPERASIONAL:**")
        st.write("""
        * Produksi: 17,000 ton (+9.7%)
        * Penjualan: 18,000 ton (+12.5%)
        * Margin laba: 8-10%
        * Debt-to-equity: <1.0
        """)
        st.markdown("**MANAJEMEN RISIKO:**")
        st.write("""
        * *Hedging* harga komoditas
        * Diversifikasi sumber pendanaan
        * Investasi R&D teknologi
        """)
        st.markdown("**INDIKATOR KEBERHASILAN:**")
        st.write("""
        * ROA > 5% | ROE > 8%
        * *Current ratio* > 1.5
        * *Debt service coverage* > 2.0
        """)
    st.markdown("---")
    st.caption("Sumber: Laporan Tahunan PT TIMAH TBK 2022-2024 | Analisis: Strategic Management Team | Proyeksi berdasarkan tren historis & analisis fundamental")

# Fungsi untuk menampilkan Summary Statistik
def display_summary_statistik(df, prediksi_2025):
    st.markdown("---")
    st.markdown("<h2 style='text-align: center; color: #2A3F5F;'>RINGKASAN ANALISIS KINERJA PT TIMAH TBK (2022-2024)</h2>", unsafe_allow_html=True)
    st.markdown("---")

    metrics = ['Produksi Timah (ton)', 'Penjualan Timah (ton)', 'Pendapatan (Rp Triliun)', 'Laba Bersih (Rp Triliun)']
    summary_data = []
    for metric in metrics:
        vals = df[metric].tolist()
        chg23_val = 'N/A'
        chg24_val = 'N/A'
        try:
            # Pastikan kolom perubahan persentase ada sebelum mencoba mengakses
            if f'Perubahan {metric.split()[0]} (%)' in df.columns:
                chg23_val = f"{df.iloc[1][f'Perubahan {metric.split()[0]} (%)']:.1f}%"
                chg24_val = f"{df.iloc[2][f'Perubahan {metric.split()[0]} (%)']:.1f}%"
        except KeyError:
            pass # Hanya untuk jaga-jaga, seharusnya sudah ditangani oleh try-except

        summary_data.append({
            'METRIK': metric,
            '2022': f"{vals[0]:,.0f}",
            '2023': f"{vals[1]:,.1f}",
            '2024': f"{vals[2]:,.1f}",
            'Δ 2023': chg23_val,
            'Δ 2024': chg24_val
        })
    st.dataframe(pd.DataFrame(summary_data).set_index('METRIK')) # Menggunakan st.dataframe untuk tampilan tabel yang lebih interaktif

    st.markdown("<h3 style='text-align: center;'>PROYEKSI 2025</h3>", unsafe_allow_html=True)
    proj_data = {
        '': ['Prediksi Linear'],
        'Produksi': [f"{prediksi_2025:,.0f}"],
        'Target': ['17,000'],
        'Asumsi': ['Optimis']
    }
    st.dataframe(pd.DataFrame(proj_data).set_index('')) # Menggunakan st.dataframe

    st.markdown("---")
    st.markdown("<h2 style='text-align: center; color: #2A3F5F;'>REKOMENDASI PRIORITAS STRATEGIS:</h2>", unsafe_allow_html=True)
    st.markdown("---")
    recommendations = [
        "1. Fokus pada efisiensi biaya untuk meningkatkan profitabilitas",
        "2. Diversifikasi produk dengan nilai tambah tinggi (refined tin products)",
        "3. Ekspansi ke pasar non-tradisional untuk mengurangi risiko konsentrasi",
        "4. Intensifikasi eksplorasi untuk menambah cadangan jangka panjang",
        "5. Implementasi hedging strategy untuk mitigasi volatilitas harga"
    ]
    for rec in recommendations:
        st.write(f"- {rec}")
    st.markdown("---")


# Fungsi untuk Analisis Clustering
def run_clustering_analysis(df):
    st.markdown("---")
    st.markdown("<h2 style='text-align: center; color: #2A3F5F;'>ANALISIS CLUSTERING KINERJA PT TIMAH TBK (2022-2024)</h2>", unsafe_allow_html=True)
    st.markdown("---")

    X = df[['Penjualan Timah (ton)', 'Pendapatan (Rp Triliun)']]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Inisialisasi KMeans dengan n_init yang eksplisit untuk menghindari FutureWarning
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10) 
    df['Cluster'] = kmeans.fit_predict(X_scaled)
    centers = scaler.inverse_transform(kmeans.cluster_centers_)

    cluster_mapping = {
        df.loc[df['Tahun'] == 2022, 'Cluster'].values[0]: 'Kinerja Tinggi',
        df.loc[df['Tahun'] == 2023, 'Cluster'].values[0]: 'Kinerja Rendah',
        df.loc[df['Tahun'] == 2024, 'Cluster'].values[0]: 'Kinerja Sedang',
    }
    df['Kategori Kinerja'] = df['Cluster'].map(cluster_mapping)

    palette = {
        'Kinerja Tinggi': '#2ca02c',
        'Kinerja Sedang': '#ff7f0e',
        'Kinerja Rendah': '#d62728'
    }

    fig_cluster, ax1_cluster = plt.subplots(figsize=(10, 7)) # Sesuaikan ukuran figure
    sns.scatterplot(
        data=df,
        x='Penjualan Timah (ton)',
        y='Pendapatan (Rp Triliun)',
        hue='Kategori Kinerja',
        palette=palette,
        s=400,
        edgecolor='black',
        ax=ax1_cluster
    )

    ax1_cluster.plot(df['Penjualan Timah (ton)'], df['Pendapatan (Rp Triliun)'],
                     color='gray', linestyle='--', linewidth=1.5, alpha=0.6)

    # Offset disesuaikan untuk posisi anotasi tahun agar tidak tumpang tindih
    offset = [(300, 0.25), (-600, -0.35), (400, 0.25)]
    for i in range(len(df)):
        x = df.loc[i, 'Penjualan Timah (ton)']
        y = df.loc[i, 'Pendapatan (Rp Triliun)']
        tahun = df.loc[i, 'Tahun']
        label = df.loc[i, 'Kategori Kinerja']
        ax1_cluster.annotate(f"{tahun}\n({label})", (x, y),
                             textcoords="offset points", xytext=offset[i],
                             fontsize=11, weight='bold',
                             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    ax1_cluster.scatter(centers[:, 0], centers[:, 1], c='black', s=300, marker='X', label='Pusat Cluster')
    for i, (cx, cy) in enumerate(centers):
        ax1_cluster.text(cx + 150, cy - 0.3, f'Pusat Cluster {i+1}', fontsize=10, color='black', weight='bold')

    ax1_cluster.set_title("Clustering Kinerja PT Timah Tbk (2022–2024)", fontsize=16, weight='bold')
    ax1_cluster.set_xlabel("Penjualan Timah (ton)", fontsize=12)
    ax1_cluster.set_ylabel("Pendapatan (Rp Triliun)", fontsize=12)
    ax1_cluster.legend(title='Kategori Kinerja', fontsize=11, title_fontsize=12, loc='upper left')

    st.pyplot(fig_cluster) # Tampilkan grafik clustering

    # Menggunakan expander untuk detail analisis clustering
    with st.expander("🔍 **Detail Analisis Clustering**"):
        col_info1, col_info2 = st.columns(2)
        with col_info1:
            st.markdown("### Kategori Kinerja:")
            st.write("""
            * **Kinerja Tinggi**: Penjualan > 19.000 ton, Pendapatan > Rp 10T
            * **Kinerja Sedang**: Penjualan 16.000–18.000 ton, Pendapatan Rp 8.5–10T
            * **Kinerja Rendah**: Penjualan < 16.000 ton, Pendapatan < Rp 8.5T

            **Contoh:**
            * 2022 → Tinggi
            * 2023 → Rendah
            * 2024 → Sedang
            """)
        with col_info2:
            st.markdown("### Analisis Pergerakan Kinerja:")
            st.write("""
            **2022 → 2023**: Penurunan tajam
            * Penjualan: 20.805 → 15.000 ton (↓ 27.9%)
            * Pendapatan: Rp 12.5T → Rp 8.39T (↓ 32.9%)
            * Laba: Rp 1.04T → -Rp 0.45T (↓ 143.3%)

            **2023 → 2024**: Pemulihan parsial
            * Penjualan: 15.000 → 16.000 ton (↑ 6.7%)
            * Pendapatan: Rp 8.39T → Rp 9.2T (↑ 9.7%)
            * Laba: -Rp 0.45T → Rp 0.3T (↑ 166.7%)

            **Target 2025:**
            * Penjualan: 18.000 ton
            * Pendapatan: Rp 10.5T
            * Laba: Rp 0.84T
            """)
    st.markdown("---")

# --- Main Streamlit App ---
def main():
    # Mengatur konfigurasi halaman Streamlit
    st.set_page_config(layout="wide", page_title="Analisis Kinerja PT Timah Tbk", initial_sidebar_state="expanded")

    # Judul Utama Aplikasi
    st.markdown(
        """
        <style>
        .title-text {
            text-align: center;
            color: #2A3F5F;
            font-size: 40px;
            font-weight: bold;
        }
        </style>
        <div class="title-text">ANALISIS KOMPREHENSIF KINERJA OPERASIONAL & KEUANGAN PT TIMAH TBK (2022-2025) 📈</div>
        """,
        unsafe_allow_html=True
    )

    # Navigasi Sidebar
    st.sidebar.header("Navigasi Dashboard")
    page_selection = st.sidebar.radio(
        "Pilih Bagian Analisis:",
        ["Dashboard Utama", "Ringkasan Statistik", "Analisis Clustering"],
        index=0 # Default ke Dashboard Utama
    )

    # Konten berdasarkan pilihan navigasi
    if page_selection == "Dashboard Utama":
        st.markdown("---")
        st.markdown("<h2 style='text-align: center; color: #1f77b4;'>📊 Dashboard Kinerja Utama</h2>", unsafe_allow_html=True)
        st.markdown("---")

        # Baris Grafik 1: Produksi, Penjualan, Keuangan
        st.container()
        col1, col2, col3 = st.columns(3)
        with col1:
            st.subheader("📈 Tren Produksi & Penjualan Timah")
            st.pyplot(plot_produksi_penjualan(df))
            st.info("Produksi dan penjualan mengalami penurunan signifikan pada 2023, diikuti sedikit pemulihan pada 2024.")
        with col2:
            st.subheader("🌍 Distribusi Pasar Ekspor Timah")
            st.pyplot(plot_ekspor_distribusi())
            st.info("Korea Selatan dan Singapura mendominasi ekspor (37%), peluang diversifikasi pasar masih terbuka lebar.")
        with col3:
            st.subheader("💰 Kinerja Keuangan (Pendapatan & Laba)")
            st.pyplot(plot_kinerja_keuangan(df))
            st.info("Tahun 2023 mengalami kerugian Rp 0.45T, namun mulai pulih pada 2024 dengan laba Rp 0.3T meski pendapatan belum sepenuhnya recover.")

        st.markdown("---") # Garis pemisah antar baris grafik

        # Baris Grafik 2: Cadangan, Perubahan Kinerja, Proyeksi
        st.container()
        col4, col5, col6 = st.columns(3)
        with col4:
            st.subheader("⛰️ Cadangan Timah & Investasi Eksplorasi")
            st.pyplot(plot_cadangan_eksplorasi(df))
            st.info("Cadangan menurun 5.3% dalam 3 tahun meski investasi eksplorasi meningkat 20%, menunjukkan tantangan dalam penambahan cadangan baru.")
        with col5:
            st.subheader("📉 Perubahan Kinerja (2023 vs 2022)")
            st.pyplot(plot_perubahan_kinerja(df))
            st.info("Semua indikator kinerja mengalami penurunan drastis pada 2023: Laba turun 143% menjadi rugi, penjualan dan pendapatan turun >25%.")
        with col6:
            st.subheader("🔮 Proyeksi Produksi Timah 2025")
            fig_proj, prediksi_2025_val = plot_proyeksi_produksi(df)
            st.pyplot(fig_proj)
            # Simpan prediksi di session_state agar bisa diakses di halaman lain
            st.session_state['prediksi_2025'] = prediksi_2025_val
            improvement_pct = ((prediksi_2025_val / df.loc[df['Tahun'] == 2024, 'Produksi Timah (ton)'].values[0]) - 1) * 100
            st.info(f"Model linear memprediksi produksi 2025: {prediksi_2025_val:,.0f} ton ({improvement_pct:+.1f}% vs 2024), asumsi tren pemulihan berlanjut.")


        # Bagian Analisis Strategis & Rekomendasi
        display_strategi_rekomendasi()

    elif page_selection == "Ringkasan Statistik":
        # Pastikan prediksi_2025 sudah tersedia sebelum menampilkan
        # Jika user langsung ke halaman ini, prediksi akan dihitung ulang
        pred_2025_val = st.session_state.get('prediksi_2025', None)
        if pred_2025_val is None:
            # Jalankan fungsi prediksi jika belum dihitung (misal user refresh halaman)
            _, pred_2025_val = plot_proyeksi_produksi(df)
            st.session_state['prediksi_2025'] = pred_2025_val 
        display_summary_statistik(df, pred_2025_val)

    elif page_selection == "Analisis Clustering":
        run_clustering_analysis(df)

if __name__ == "__main__":
    main()

