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

# --- Konfigurasi Visual Global ---
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
@st.cache_data # Menggunakan cache Streamlit untuk data
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

# --- Fungsi-fungsi Plotting ---

# Fungsi untuk membuat Grafik 1: Tren Produksi & Penjualan
def plot_produksi_penjualan(df):
    fig, ax1 = plt.subplots(figsize=(8, 5)) 

    line1 = ax1.plot(df['Tahun'], df['Produksi Timah (ton)'],
                     marker='o', markersize=8, linewidth=3,
                     label='Produksi Timah', color=CORPORATE_COLORS[0])
    line2 = ax1.plot(df['Tahun'], df['Penjualan Timah (ton)'],
                     marker='s', markersize=8, linewidth=3,
                     label='Penjualan Timah', color=CORPORATE_COLORS[1])

    for i, row in df.iterrows():
        ax1.annotate(f"{row['Produksi Timah (ton)']:,}",
                     (row['Tahun'], row['Produksi Timah (ton)']),
                     textcoords="offset points", xytext=(0,10), 
                     ha='center', fontsize=11, fontweight='bold')
        ax1.annotate(f"{row['Penjualan Timah (ton)']:,}",
                     (row['Tahun'], row['Penjualan Timah (ton)']),
                     textcoords="offset points", xytext=(0,-20), 
                     ha='center', fontsize=11, fontweight='bold')

    ax1.set_title("Tren Produksi vs Penjualan Timah", pad=15)
    ax1.set_xlabel("Tahun")
    ax1.set_ylabel("Volume (ton)")
    ax1.legend(loc='upper right', frameon=True, shadow=True)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(12000, 22000)
    return fig

# Fungsi untuk membuat Grafik 2: Distribusi Pasar Ekspor
def plot_ekspor_distribusi():
    fig, ax2 = plt.subplots(figsize=(7, 7)) 

    ekspor_data = {
        "Korea Selatan": 19, "Singapura": 18, "Jepang": 12,
        "Belanda": 12, "India": 10, "China": 7, "Domestik": 12
    }
    
    # Sort data for better visualization in pie chart (optional)
    sorted_ekspor = sorted(ekspor_data.items(), key=lambda x: x[1], reverse=True)
    labels = [f"{k} ({v}%)" for k, v in sorted_ekspor] # Labels with percentage for legend
    sizes = [v for k, v in sorted_ekspor]

    colors = plt.cm.tab20(np.linspace(0, 1, len(ekspor_data))) # Use tab20 for more distinct colors
    
    wedges, texts, autotexts = ax2.pie(sizes,
                                       autopct='%1.1f%%',
                                       startangle=90,
                                       colors=colors,
                                       pctdistance=0.85, 
                                       textprops={'fontsize': 11})

    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(11)
    
    # Place labels outside with lines
    ax2.legend(wedges, labels,
               title="Pasar",
               loc="center left",
               bbox_to_anchor=(1, 0, 0.5, 1))

    ax2.set_title("Distribusi Pasar Ekspor Timah 2024", pad=20)
    ax2.axis('equal') 
    return fig

# Fungsi untuk membuat Grafik 3: Kinerja Keuangan
def plot_kinerja_keuangan(df):
    fig, ax3 = plt.subplots(figsize=(8, 5)) 
    bars1 = ax3.bar(df['Tahun'] - 0.2, df['Pendapatan (Rp Triliun)'], 
                    alpha=0.7, label='Pendapatan', color=CORPORATE_COLORS[2], width=0.4)
    ax3b = ax3.twinx()
    bars2 = ax3b.bar(df['Tahun'] + 0.2, df['Laba Bersih (Rp Triliun)'], 
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
    
    # Combine legends
    lines, labels = ax3.get_legend_handles_labels()
    lines2, labels2 = ax3b.get_legend_handles_labels()
    ax3b.legend(lines + lines2, labels + labels2, loc='upper left')

    ax3.grid(True, alpha=0.3)
    return fig

# Fungsi untuk membuat Grafik 4: Cadangan & Investasi Eksplorasi
def plot_cadangan_eksplorasi(df):
    fig, ax4 = plt.subplots(figsize=(8, 5)) 

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
    
    # Combine legends
    lines, labels = ax4.get_legend_handles_labels()
    lines2, labels2 = ax4b.get_legend_handles_labels()
    ax4b.legend(lines + lines2, labels + labels2, loc='upper left')

    ax4.grid(True, alpha=0.3)
    ax4b.set_ylim(30, 45) 
    return fig

# Fungsi untuk membuat Grafik 5: Perubahan Kinerja (2023 vs 2022)
def plot_perubahan_kinerja(df):
    fig, ax5 = plt.subplots(figsize=(8, 5)) 
    
    # Perbaikan: Ambil data perubahan untuk 2023 (index 1) dan 2024 (index 2)
    change_metrics = ['Penjualan Timah', 'Pendapatan', 'Laba Bersih']
    change_2023 = [df.iloc[1][f'Perubahan {m} (%)'] for m in change_metrics]
    change_2024 = [df.iloc[2][f'Perubahan {m} (%)'] for m in change_metrics]
    
    bar_width = 0.35
    index = np.arange(len(change_metrics))

    bars_2023 = ax5.barh(index + bar_width/2, change_2023, height=bar_width, 
                         color=['red' if x < 0 else 'green' for x in change_2023], 
                         alpha=0.7, label='Perubahan 2023 vs 2022')
    bars_2024 = ax5.barh(index - bar_width/2, change_2024, height=bar_width, 
                         color=['red' if x < 0 else 'green' for x in change_2024], 
                         alpha=0.7, label='Perubahan 2024 vs 2023') # Label untuk 2024

    for i, (v23, v24) in enumerate(zip(change_2023, change_2024)):
        ax5.text(v23 + (8 if v23 >= 0 else -8), i + bar_width/2, f'{v23:.1f}%',
                 ha=('left' if v23 >= 0 else 'right'), va='center', fontweight='bold', fontsize=10)
        ax5.text(v24 + (8 if v24 >= 0 else -8), i - bar_width/2, f'{v24:.1f}%',
                 ha=('left' if v24 >= 0 else 'right'), va='center', fontweight='bold', fontsize=10)

    ax5.set_title('Perubahan Kinerja Tahunan (%)', pad=20)
    ax5.set_xlabel('Perubahan (%)')
    ax5.set_yticks(index)
    ax5.set_yticklabels(change_metrics)
    ax5.axvline(0, color='black', linewidth=1)
    ax5.grid(True, axis='x', alpha=0.3)
    ax5.set_xlim(-150, 10) 
    ax5.legend(loc='lower right')
    return fig

# Fungsi untuk membuat Grafik 6: Proyeksi Produksi 2025
def plot_proyeksi_produksi(df):
    fig, ax6 = plt.subplots(figsize=(8, 5)) 

    X = df['Tahun'].values.reshape(-1, 1)
    y = df['Produksi Timah (ton)'].values
    model = LinearRegression().fit(X, y)
    
    # Adjusted prediction for 2025 based on the trend from 2023-2024 data if possible,
    # otherwise, linear regression from all data is fine.
    # For now, keeping your original linear regression for 2022-2024 data.
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
    return fig, prediksi_2025

# --- Fungsi untuk Konten Streamlit ---

# Fungsi untuk menampilkan KPI Cards
def display_kpi_cards(df, prediksi_2025):
    st.markdown("<h3 style='text-align: center; color: #333;'>Metrik Kinerja Utama 2024 & Proyeksi 2025</h3>", unsafe_allow_html=True)
    col_kpi1, col_kpi2, col_kpi3, col_kpi4 = st.columns(4)

    # Ambil data tahun 2024
    data_2024 = df[df['Tahun'] == 2024].iloc[0]
    
    # Hitung perubahan vs 2023 untuk produksi, penjualan, pendapatan, laba
    prod_chg_2024 = ((data_2024['Produksi Timah (ton)'] / df[df['Tahun'] == 2023].iloc[0]['Produksi Timah (ton)']) - 1) * 100
    penj_chg_2024 = ((data_2024['Penjualan Timah (ton)'] / df[df['Tahun'] == 2023].iloc[0]['Penjualan Timah (ton)']) - 1) * 100
    pend_chg_2024 = ((data_2024['Pendapatan (Rp Triliun)'] / df[df['Tahun'] == 2023].iloc[0]['Pendapatan (Rp Triliun)']) - 1) * 100
    laba_chg_2024 = None # Akan dihitung jika laba 2023 tidak 0
    if df[df['Tahun'] == 2023].iloc[0]['Laba Bersih (Rp Triliun)'] != 0:
        laba_chg_2024 = ((data_2024['Laba Bersih (Rp Triliun)'] / df[df['Tahun'] == 2023].iloc[0]['Laba Bersih (Rp Triliun)']) - 1) * 100

    def format_kpi_card(title, value, unit="", change_val=None):
        delta_str = ""
        delta_color = "normal"
        if change_val is not None:
            if change_val > 0:
                delta_str = f"↑ {change_val:.1f}% vs 2023"
                delta_color = "green"
            elif change_val < 0:
                delta_str = f"↓ {abs(change_val):.1f}% vs 2023"
                delta_color = "red"
            else:
                delta_str = "↔ 0% vs 2023"
        
        st.markdown(f"""
        <div style="
            background-color: #f0f2f6; 
            padding: 15px; 
            border-radius: 10px; 
            text-align: center; 
            box-shadow: 2px 2px 5px rgba(0,0,0,0.1);">
            <h4 style="color: #4A4A4A; margin-bottom: 5px;">{title}</h4>
            <p style="font-size: 28px; font-weight: bold; color: #2A3F5F; margin-bottom: 0;">
                {value}{unit}
            </p>
            <p style="font-size: 14px; color: {delta_color}; margin-top: 5px;">
                {delta_str}
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col_kpi1:
        format_kpi_card("Produksi Timah (2024)", f"{data_2024['Produksi Timah (ton)']:,}", " ton", prod_chg_2024)
    with col_kpi2:
        format_kpi_card("Penjualan Timah (2024)", f"{data_2024['Penjualan Timah (ton)']:,}", " ton", penj_chg_2024)
    with col_kpi3:
        format_kpi_card("Pendapatan (2024)", f"Rp {data_2024['Pendapatan (Rp Triliun)']:.2f}", "T", pend_chg_2024)
    with col_kpi4:
        # Menangani laba negatif untuk tampilan
        laba_str = f"Rp {data_2024['Laba Bersih (Rp Triliun)']:.2f}"
        if data_2024['Laba Bersih (Rp Triliun)'] < 0:
            laba_str = f"-Rp {abs(data_2024['Laba Bersih (Rp Triliun)']):.2f}"
        format_kpi_card("Laba Bersih (2024)", laba_str, "T", laba_chg_2024)

    # KPI untuk Proyeksi 2025
    st.markdown("<br>", unsafe_allow_html=True) # Spacer
    st.markdown("<h3 style='text-align: center; color: #333;'>Proyeksi Utama 2025</h3>", unsafe_allow_html=True)
    col_proj1, col_proj2 = st.columns(2)

    with col_proj1:
        format_kpi_card("Proyeksi Produksi (2025)", f"{prediksi_2025:,.0f}", " ton", ((prediksi_2025 / data_2024['Produksi Timah (ton)']) - 1) * 100)
    with col_proj2:
        st.markdown(f"""
        <div style="
            background-color: #f0f2f6; 
            padding: 15px; 
            border-radius: 10px; 
            text-align: center; 
            box-shadow: 2px 2px 5px rgba(0,0,0,0.1);">
            <h4 style="color: #4A4A4A; margin-bottom: 5px;">Target Produksi (2025)</h4>
            <p style="font-size: 28px; font-weight: bold; color: #2A3F5F; margin-bottom: 0;">
                17,000 ton
            </p>
            <p style="font-size: 14px; color: green; margin-top: 5px;">
                ↑ 9.7% vs 2024
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

# Fungsi untuk bagian Analisis Strategis & Rekomendasi
def display_strategi_rekomendasi():
    st.markdown("<h2 style='text-align: center; color: #2D7DD2;'>ANALISIS STRATEGIS & REKOMENDASI PT TIMAH TBK</h2>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center; color: #5C9E47;'>Rangkuman Eksekutif & Strategi Pemulihan 2025</h3>", unsafe_allow_html=True)
    st.markdown("---")

    # Menggunakan st.tabs untuk navigasi yang lebih intuitif di dalam bagian ini
    tab1, tab2, tab3 = st.tabs(["📊 Performa 2022-2024", "🚀 Strategi Pemulihan 2025", "🎯 Target & Proyeksi 2025"])

    with tab1:
        st.markdown("### Ringkasan Performa Historis:")
        st.markdown("""
        * **Produksi:** Dari 19,825 ton (2022) menjadi 15,500 ton (2024) – **penurunan signifikan 21.8%**.
        * **Penjualan:** Turun dari 20,805 ton (2022) menjadi 16,000 ton (2024) – **penurunan 23.1%**.
        * **Pendapatan:** Merosot dari Rp 12.5 Triliun (2022) menjadi Rp 9.2 Triliun (2024) – **penurunan 26.4%**.
        * **Laba Bersih:** Dari Rp 1.04 Triliun (2022) menjadi kerugian Rp 0.45 Triliun (2023), kemudian pulih menjadi Rp 0.3 Triliun (2024) – **penurunan drastis 71.2% dari 2022 ke 2024**.
        """)
        st.markdown("### Tantangan Utama yang Dihadapi:")
        st.markdown("""
        * **Volatilitas Harga Timah Global:** Fluktuasi harga komoditas sangat mempengaruhi pendapatan dan laba perusahaan.
        * **Peningkatan Biaya Operasional:** Biaya penambangan dan pemrosesan yang terus meningkat menekan margin keuntungan.
        * **Penurunan Cadangan:** Cadangan timah menurun 5.3% (dari 330 ribu ton menjadi 312.5 ribu ton), menandakan perlunya eksplorasi yang lebih agresif.
        * **Ketergantungan Pasar Ekspor:** 88% penjualan berasal dari ekspor, membuat perusahaan rentan terhadap dinamika pasar dan kebijakan perdagangan internasional.
        """)
        st.markdown("<br>", unsafe_allow_html=True)
        st.image("https://raw.githubusercontent.com/fadhilmuh/data_science/main/image_efab02.png", caption="Tantangan Industri Timah")
    
    with tab2:
        st.markdown("### Strategi untuk Pemulihan dan Pertumbuhan 2025:")
        st.markdown("""
        #### 📈 **DIVERSIFIKASI PRODUK:**
        * **Tingkatkan Produk Olahan Bernilai Tambah:** Fokus pada produksi *tin solder*, *tin chemicals*, atau *tin alloys* untuk meningkatkan margin keuntungan.
        * **Kembangkan Produk Turunan Elektronik:** Manfaatkan timah dalam komponen elektronik yang memiliki permintaan tinggi di pasar global.
        """)
        st.markdown("""
        #### ⚙️ **EFISIENSI OPERASIONAL:**
        * **Target Penghematan Biaya 15%:** Identifikasi area-area inefisiensi dan implementasikan program penghematan biaya yang agresif.
        * **Adopsi Teknologi Penambangan Modern:** Gunakan teknologi baru seperti *IoT* untuk optimasi peralatan, *predictive maintenance*, dan otomatisasi untuk mengurangi biaya operasional dan meningkatkan produktivitas.
        * **Optimasi Rantai Pasok (*Supply Chain*):** Perbaiki logistik dan manajemen inventaris untuk mengurangi biaya dan waktu tunggu.
        """)
        st.markdown("""
        #### 🌎 **EKSPANSI PASAR:**
        * **Penetrasi Pasar India (+15%):** Manfaatkan pertumbuhan ekonomi India untuk meningkatkan pangsa pasar.
        * **Pengembangan Pasar Domestik:** Tingkatkan konsumsi timah di dalam negeri melalui edukasi dan kolaborasi dengan industri lokal.
        * **Diversifikasi Regional Ekspor:** Kurangi ketergantungan pada beberapa pasar besar dengan mencari pasar baru di Asia Tenggara, Timur Tengah, atau Afrika.
        """)
        st.image("https://raw.githubusercontent.com/fadhilmuh/data_science/main/Gemini_Generated_Image_vej734vej734vej7.jpg", caption="Strategi Diversifikasi & Efisiensi")

    with tab3:
        st.markdown("### Target Kuantitatif dan Kualitatif untuk Tahun 2025:")
        st.markdown("""
        #### ✅ **SASARAN OPERASIONAL:**
        * **Produksi:** Target 17,000 ton (**↑ 9.7%** dari 2024).
        * **Penjualan:** Target 18,000 ton (**↑ 12.5%** dari 2024).
        * **Margin Laba:** Target 8-10% dari pendapatan.
        * ***Debt-to-Equity Ratio***: Di bawah 1.0 untuk menjaga kesehatan keuangan.
        """)
        st.markdown("""
        #### 🛡️ **MANAJEMEN RISIKO:**
        * ***Hedging* Harga Komoditas:** Lindungi perusahaan dari fluktuasi harga timah yang ekstrem melalui instrumen *hedging*.
        * **Diversifikasi Sumber Pendanaan:** Kurangi ketergantungan pada satu jenis pendanaan.
        * **Investasi R&D Teknologi:** Terus berinvestasi dalam penelitian dan pengembangan untuk menemukan metode penambangan dan pemrosesan yang lebih efisien dan ramah lingkungan.
        """)
        st.markdown("""
        #### 🏆 **INDIKATOR KEBERHASILAN (KPIs):**
        * ***Return on Assets (ROA)***: Di atas 5%.
        * ***Return on Equity (ROE)***: Di atas 8%.
        * ***Current Ratio***: Di atas 1.5 untuk memastikan likuiditas yang sehat.
        * ***Debt Service Coverage Ratio (DSCR)***: Di atas 2.0 untuk menunjukkan kemampuan membayar utang.
        """)
        st.markdown("<br>", unsafe_allow_html=True)
        st.image("https://raw.githubusercontent.com/fadhilmuh/data_science/main/Gemini_Generated_Image_sxa5m8sxa5m8sxa5.jpg", caption="Target & Indikator Keberhasilan")

    st.markdown("---")
    st.caption("Sumber: Laporan Tahunan PT TIMAH TBK 2022-2024 | Analisis: Strategic Management Team | Proyeksi berdasarkan tren historis & analisis fundamental")

# Fungsi untuk menampilkan Summary Statistik
def display_summary_statistik(df, prediksi_2025):
    st.markdown("---")
    st.markdown("<h2 style='text-align: center; color: #2A3F5F;'>RINGKASAN ANALISIS KINERJA PT TIMAH TBK (2022-2024)</h2>", unsafe_allow_html=True)
    st.markdown("---")

    metrics_df = df[['Tahun', 'Produksi Timah (ton)', 'Penjualan Timah (ton)', 'Pendapatan (Rp Triliun)', 'Laba Bersih (Rp Triliun)']].set_index('Tahun')

    st.markdown("### Tabel Ringkasan Kinerja Utama:")
    # Transpose DataFrame untuk tampilan yang lebih mudah dibaca di Streamlit
    transposed_metrics = metrics_df.T 
    st.dataframe(transposed_metrics.style.format({
        2022: '{:,.0f}', 
        2023: '{:,.0f}', 
        2024: '{:,.0f}', 
        'Pendapatan (Rp Triliun)': 'Rp {:.2f}T', 
        'Laba Bersih (Rp Triliun)': 'Rp {:.2f}T'
    }), use_container_width=True)

    # Menambahkan detail perubahan persentase secara terpisah atau dalam konteks
    st.markdown("### Perubahan Persentase Tahunan (vs Tahun Sebelumnya):")
    change_summary = []
    for i in range(1, len(df)):
        tahun_sekarang = df.loc[i, 'Tahun']
        tahun_sebelumnya = df.loc[i-1, 'Tahun']
        
        row_dict = {'Metrik': f'Perubahan {tahun_sekarang} vs {tahun_sebelumnya}'}
        for col in ['Penjualan Timah', 'Pendapatan', 'Laba Bersih']:
            key_perc = f'Perubahan {col} (%)'
            if key_perc in df.columns:
                val = df.loc[i, key_perc]
                row_dict[col] = f"{val:+.1f}%" if pd.notna(val) else 'N/A'
        change_summary.append(row_dict)
    
    st.dataframe(pd.DataFrame(change_summary).set_index('Metrik'), use_container_width=True)

    st.markdown("### Proyeksi Produksi Timah 2025:")
    proj_data = {
        'Tipe Proyeksi': ['Prediksi Linear', 'Target Perusahaan'],
        'Produksi (ton)': [f"{prediksi_2025:,.0f}", '17,000'],
        'Asumsi': ['Berdasarkan tren historis 2022-2024', 'Optimis, dengan strategi pemulihan']
    }
    st.dataframe(pd.DataFrame(proj_data).set_index('Tipe Proyeksi'), use_container_width=True)

    st.markdown("---")
    st.markdown("<h2 style='text-align: center; color: #2A3F5F;'>REKOMENDASI PRIORITAS STRATEGIS:</h2>", unsafe_allow_html=True)
    st.markdown("---")
    recommendations = [
        "1. **Fokus pada Efisiensi Biaya:** Implementasi program penghematan biaya yang ketat di seluruh operasi untuk meningkatkan profitabilitas, terutama mengingat penurunan laba.",
        "2. **Diversifikasi Produk Bernilai Tambah:** Tingkatkan portofolio produk olahan timah (misalnya, timah solder, bahan kimia timah) untuk mengurangi ketergantungan pada komoditas mentah dan meningkatkan margin.",
        "3. **Ekspansi Pasar Non-Tradisional:** Kurangi risiko konsentrasi pasar dengan menargetkan pasar baru di luar Korea Selatan, Singapura, dan Jepang.",
        "4. **Intensifikasi Eksplorasi Cadangan:** Tingkatkan investasi dan teknologi eksplorasi untuk mengidentifikasi cadangan baru guna menjamin keberlanjutan pasokan jangka panjang.",
        "5. **Implementasi Strategi Hedging:** Manfaatkan instrumen *hedging* untuk memitigasi risiko volatilitas harga timah global yang dapat berdampak signifikan pada pendapatan."
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

    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10) 
    df['Cluster'] = kmeans.fit_predict(X_scaled)
    centers = scaler.inverse_transform(kmeans.cluster_centers_)

    # Mapping cluster tahun agar konsisten
    cluster_2022 = df.loc[df['Tahun'] == 2022, 'Cluster'].values[0]
    cluster_2023 = df.loc[df['Tahun'] == 2023, 'Cluster'].values[0]
    cluster_2024 = df.loc[df['Tahun'] == 2024, 'Cluster'].values[0]

    cluster_mapping = {}
    if cluster_2022 == 0: # Ini hanya asumsi, sesuaikan jika cluster ID berubah
        cluster_mapping[cluster_2022] = 'Kinerja Tinggi'
        cluster_mapping[cluster_2023] = 'Kinerja Rendah'
        cluster_mapping[cluster_2024] = 'Kinerja Sedang'
    elif cluster_2022 == 1:
        cluster_mapping[cluster_2022] = 'Kinerja Tinggi'
        cluster_mapping[cluster_2023] = 'Kinerja Rendah'
        cluster_mapping[cluster_2024] = 'Kinerja Sedang'
    elif cluster_2022 == 2:
        cluster_mapping[cluster_2022] = 'Kinerja Tinggi'
        cluster_mapping[cluster_2023] = 'Kinerja Rendah'
        cluster_mapping[cluster_2024] = 'Kinerja Sedang'
    
    # Fallback/General mapping if specific year mapping is complex
    # Try to assign based on the actual values of the cluster centers
    sorted_centers_idx = np.argsort(centers[:, 1]) # Sort by Pendapatan (second column of centers)
    # Assign labels based on sorted order (assuming lowest income is "Rendah", highest is "Tinggi")
    cluster_mapping_dynamic = {
        sorted_centers_idx[0]: 'Kinerja Rendah',
        sorted_centers_idx[1]: 'Kinerja Sedang',
        sorted_centers_idx[2]: 'Kinerja Tinggi'
    }
    df['Kategori Kinerja'] = df['Cluster'].map(cluster_mapping_dynamic)

    palette = {
        'Kinerja Tinggi': '#2ca02c', # Green
        'Kinerja Sedang': '#ff7f0e', # Orange
        'Kinerja Rendah': '#d62728'  # Red
    }

    fig_cluster, ax1_cluster = plt.subplots(figsize=(10, 7)) 
    sns.scatterplot(
        data=df,
        x='Penjualan Timah (ton)',
        y='Pendapatan (Rp Triliun)',
        hue='Kategori Kinerja',
        palette=palette,
        s=400,
        edgecolor='black',
        ax=ax1_cluster,
        zorder=2 # Make sure points are on top
    )

    # Plot lines connecting years
    for i in range(len(df) - 1):
        ax1_cluster.plot(df['Penjualan Timah (ton)'][i:i+2], df['Pendapatan (Rp Triliun)'][i:i+2],
                         color='gray', linestyle='--', linewidth=1.5, alpha=0.6, zorder=1)

    # Offset disesuaikan untuk posisi anotasi tahun agar tidak tumpang tindih
    # Disesuaikan secara manual untuk contoh data ini
    offset_anno = {
        2022: (50, 20),
        2023: (-50, -40),
        2024: (50, 20)
    }

    for i in range(len(df)):
        x = df.loc[i, 'Penjualan Timah (ton)']
        y = df.loc[i, 'Pendapatan (Rp Triliun)']
        tahun = df.loc[i, 'Tahun']
        label = df.loc[i, 'Kategori Kinerja']
        
        # Get dynamic offset
        xytext_offset = offset_anno.get(tahun, (0,10)) # Default if year not in map
        
        ax1_cluster.annotate(f"{tahun}\n({label})", (x, y),
                             textcoords="offset points", xytext=xytext_offset,
                             fontsize=11, weight='bold',
                             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8),
                             arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2", color='black')) # Add arrow

    ax1_cluster.scatter(centers[:, 0], centers[:, 1], c='black', s=300, marker='X', label='Pusat Cluster', zorder=3)
    for i, (cx, cy) in enumerate(centers):
        # Annotate cluster centers more clearly
        ax1_cluster.text(cx, cy - (0.5 if i == sorted_centers_idx[0] else -0.5), # Adjust y for visibility
                         f'Cluster {i+1}\n({cluster_mapping_dynamic[i]})',
                         fontsize=10, color='black', weight='bold', ha='center',
                         bbox=dict(boxstyle='round,pad=0.2', facecolor='yellow', alpha=0.7))

    ax1_cluster.set_title("Clustering Kinerja PT Timah Tbk (2022–2024)", fontsize=16, weight='bold')
    ax1_cluster.set_xlabel("Penjualan Timah (ton)", fontsize=12)
    ax1_cluster.set_ylabel("Pendapatan (Rp Triliun)", fontsize=12)
    ax1_cluster.legend(title='Kategori Kinerja', fontsize=11, title_fontsize=12, loc='upper left')
    ax1_cluster.grid(True, alpha=0.3)

    st.pyplot(fig_cluster) 

    st.markdown("---")
    st.markdown("### Interpretasi Cluster:")
    st.info("""
    Analisis clustering membagi kinerja tahunan PT Timah Tbk menjadi 3 kategori berdasarkan Penjualan dan Pendapatan:
    * **Kinerja Tinggi (Hijau):** Ditunjukkan oleh tahun 2022, dengan volume penjualan dan pendapatan yang paling tinggi.
    * **Kinerja Rendah (Merah):** Ditunjukkan oleh tahun 2023, menunjukkan penurunan signifikan dalam penjualan dan pendapatan, yang juga diikuti oleh kerugian laba bersih.
    * **Kinerja Sedang (Oranye):** Ditunjukkan oleh tahun 2024, merepresentasikan pemulihan parsial dari tahun sebelumnya, dengan penjualan dan pendapatan yang membaik namun belum mencapai level 2022.

    Panah abu-abu menunjukkan **pergerakan kinerja dari tahun ke tahun**, menggambarkan bagaimana PT Timah Tbk mengalami penurunan tajam dari `Kinerja Tinggi` (2022) ke `Kinerja Rendah` (2023), lalu beranjak naik kembali menuju `Kinerja Sedang` (2024).
    """)
    st.markdown("---")


# --- Main Streamlit App ---
def main():
    st.set_page_config(layout="wide", page_title="Analisis Kinerja PT Timah Tbk", initial_sidebar_state="expanded")

    # Custom CSS untuk tampilan yang lebih profesional
    st.markdown("""
    <style>
    /* Mengatur lebar sidebar */
    [data-testid="stSidebar"] {
        background-color: #f0f2f6; /* Warna latar belakang sidebar */
    }
    /* Mengatur style tombol radio di sidebar */
    [data-testid="stSidebarNav"] li > a {
        font-size: 1.1em;
        padding: 10px 15px;
        margin: 5px 0;
        border-radius: 8px;
    }
    [data-testid="stSidebarNav"] li > a:hover {
        background-color: #e0e6ed;
        color: #1f77b4;
    }
    [data-testid="stSidebarNav"] li > a.st-emotion-cache-1f87s81.edgvbvh10 { /* Class untuk radio button aktif */
        background-color: #1f77b4; /* Warna latar belakang untuk yang aktif */
        color: white; /* Warna teks untuk yang aktif */
        font-weight: bold;
    }
    /* Mengatur background warna untuk header */
    .stApp > header {
        background-color: #ffffff; /* Putih bersih */
    }
    /* Judul utama */
    .title-text {
        text-align: center;
        color: #2A3F5F;
        font-size: 44px; /* Ukuran font lebih besar */
        font-weight: bold;
        padding-top: 20px;
        padding-bottom: 20px;
        border-bottom: 2px solid #e0e0e0; /* Garis bawah elegan */
        margin-bottom: 30px;
    }
    /* Subheader di dalam konten */
    h2, h3 {
        color: #1f77b4; /* Warna biru untuk subheader */
        text-align: center;
        margin-top: 25px;
        margin-bottom: 20px;
    }
    h3 {
        color: #2ca02c; /* Warna hijau untuk h3 di strategi */
    }
    /* Info box */
    .st-emotion-cache-1kv4i7x { /* Class untuk st.info box */
        background-color: #e6f2ff; /* Light blue */
        border-left: 5px solid #2196F3; /* Darker blue border */
        color: #2A3F5F;
        padding: 10px;
        border-radius: 8px;
    }
    /* Adjust text color in expander */
    .streamlit-expanderHeader {
        font-size: 1.1em;
        font-weight: bold;
        color: #2A3F5F;
    }
    </style>
    """, unsafe_allow_html=True)


    st.markdown(
        """
        <div class="title-text">ANALISIS KOMPREHENSIF KINERJA OPERASIONAL & KEUANGAN PT TIMAH TBK (2022-2025) 📈</div>
        """,
        unsafe_allow_html=True
    )

    st.sidebar.header("Navigasi Dashboard")
    page_selection = st.sidebar.radio(
        "Pilih Bagian Analisis:",
        ["Dashboard Utama", "Ringkasan Statistik", "Analisis Clustering", "Rekomendasi Strategis"],
        index=0 
    )

    # Konten berdasarkan pilihan navigasi
    if page_selection == "Dashboard Utama":
        st.markdown("---")
        st.markdown("<h2 style='text-align: center;'>📊 Dashboard Kinerja Utama PT Timah Tbk</h2>", unsafe_allow_html=True)
        st.markdown("---")
        
        # Hitung prediksi 2025 di awal agar bisa digunakan di KPI Card
        _, prediksi_2025_val = plot_proyeksi_produksi(df)
        st.session_state['prediksi_2025'] = prediksi_2025_val

        # Menampilkan KPI Cards
        display_kpi_cards(df, prediksi_2025_val)

        # Tabs untuk memisahkan grafik
        tab_tren, tab_keuangan, tab_eksplorasi = st.tabs(["📉 Tren Operasional", "💰 Kinerja Keuangan", "⛰️ Cadangan & Eksplorasi"])

        with tab_tren:
            st.markdown("<h3 style='text-align: center;'>Grafik Tren Produksi & Penjualan</h3>", unsafe_allow_html=True)
            col_t1, col_t2 = st.columns([1, 1]) # Gunakan 2 kolom untuk grafik dan keterangan
            with col_t1:
                st.pyplot(plot_produksi_penjualan(df))
            with col_t2:
                st.markdown("### Penjelasan:")
                st.info("""
                Grafik ini menunjukkan pergerakan produksi dan penjualan timah PT Timah Tbk dari tahun 2022 hingga 2024.
                Terlihat adanya **penurunan signifikan** pada tahun 2023 setelah performa kuat di 2022, dengan sedikit **pemulihan di tahun 2024**.
                """)
                st.markdown("<br>")
                st.subheader("🔮 Proyeksi Produksi Timah 2025")
                fig_proj, prediksi_2025_val = plot_proyeksi_produksi(df)
                st.pyplot(fig_proj)
                improvement_pct = ((prediksi_2025_val / df.loc[df['Tahun'] == 2024, 'Produksi Timah (ton)'].values[0]) - 1) * 100
                st.info(f"Menggunakan model regresi linear, produksi tahun 2025 diproyeksikan mencapai **{prediksi_2025_val:,.0f} ton**, menunjukkan potensi perbaikan **{improvement_pct:+.1f}%** dari tahun 2024.")

        with tab_keuangan:
            st.markdown("<h3 style='text-align: center;'>Grafik Kinerja Keuangan & Perubahan Tahunan</h3>", unsafe_allow_html=True)
            col_k1, col_k2 = st.columns([1,1])
            with col_k1:
                st.pyplot(plot_kinerja_keuangan(df))
                st.info("Kinerja keuangan menunjukkan penurunan pendapatan yang tajam pada tahun 2023, berujung pada kerugian bersih. Namun, ada tanda-tanda pemulihan di tahun 2024.")
            with col_k2:
                st.pyplot(plot_perubahan_kinerja(df))
                st.info("Grafik persentase perubahan menegaskan dampak negatif pada tahun 2023 di semua metrik utama, dengan Laba Bersih mengalami kontraksi terbesar.")
            
            st.markdown("<h3 style='text-align: center;'>🌍 Distribusi Pasar Ekspor Timah</h3>", unsafe_allow_html=True)
            col_e1, col_e2 = st.columns([0.7,1]) # Sesuaikan rasio kolom
            with col_e1:
                st.pyplot(plot_ekspor_distribusi())
            with col_e2:
                st.markdown("### Penjelasan Distribusi Pasar:")
                st.info("""
                Analisis distribusi pasar ekspor tahun 2024 menunjukkan **konsentrasi yang signifikan** pada beberapa negara.
                * **Korea Selatan** dan **Singapura** bersama-sama menyumbang **37%** dari total ekspor.
                * Ketergantungan pada pasar-pasar ini menimbulkan **risiko konsentrasi**.
                * **Peluang diversifikasi** ke pasar lain seperti India atau peningkatan pasar domestik sangat penting untuk stabilitas jangka panjang.
                """)
                st.markdown("---")
                st.image("https://raw.githubusercontent.com/fadhilmuh/data_science/main/Gemini_Generated_Image_llshlollshlollsh.jpg", caption="Peluang Pasar Baru")


        with tab_eksplorasi:
            st.markdown("<h3 style='text-align: center;'>Grafik Cadangan & Investasi Eksplorasi</h3>", unsafe_allow_html=True)
            col_c1, col_c2 = st.columns([1,1])
            with col_c1:
                st.pyplot(plot_cadangan_eksplorasi(df))
            with col_c2:
                st.markdown("### Penjelasan:")
                st.info("""
                Meskipun biaya eksplorasi telah meningkat secara konsisten dari 2022 hingga 2024, **cadangan timah terbukti terus menurun** (sekitar 5.3% dari 2022-2024).
                Ini mengindikasikan bahwa investasi eksplorasi yang ada belum cukup efektif dalam menemukan cadangan baru yang signifikan untuk menggantikan cadangan yang telah ditambang.
                Diperlukan strategi eksplorasi yang lebih efisien atau inovatif untuk menjaga keberlanjutan operasi jangka panjang.
                """)
                st.markdown("---")
                st.image("https://raw.githubusercontent.com/fadhilmuh/data_science/main/Gemini_Generated_Image_o7tpkwo7tpkwo7tp.jpg", caption="Eksplorasi Timah")


    elif page_selection == "Ringkasan Statistik":
        pred_2025_val = st.session_state.get('prediksi_2025', None)
        if pred_2025_val is None:
            # Jika user langsung ke halaman ini, hitung ulang prediksi
            _, pred_2025_val = plot_proyeksi_produksi(df)
            st.session_state['prediksi_2025'] = pred_2025_val 
        display_summary_statistik(df, pred_2025_val)

    elif page_selection == "Analisis Clustering":
        run_clustering_analysis(df)
    
    elif page_selection == "Rekomendasi Strategis":
        display_strategi_rekomendasi()


if __name__ == "__main__":
    main()
