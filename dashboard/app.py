import streamlit as st
import requests
import pandas as pd
import feedparser  
import time  


st.set_page_config(
    page_title="FinGuard AI ", 
    page_icon="🛡️", 
    layout="wide",
    initial_sidebar_state="expanded"
)


st.markdown("""
    <style>
        .main .block-container { padding-top: 2rem; }
        .stButton>button {
            width: 100%;
            border-radius: 8px;
            height: 3em;
            transition: all 0.3s ease;
        }
        .metric-card {
            background-color: #f0f4f8;
            border: 1px solid #d9e2ec;
            border-radius: 12px;
            padding: 20px;
            text-align: center;
            box-shadow: 0 4px 6px rgba(29, 78, 216, 0.03);
        }
        .metric-value {
            font-size: 26px;
            font-weight: 700;
            color: #102a43;
        }
        .metric-label {
            font-size: 13px;
            color: #627d98;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 5px;
        }
        .news-card-box {
            background-color: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 20px;
            border-left: 5px solid #2563eb;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.01);
        }
        .news-title-text {
            font-size: 18px;
            font-weight: 600;
            color: #1e293b;
            margin-bottom: 8px;
        }
        .news-date-text {
            font-size: 12px;
            color: #94a3b8;
            margin-bottom: 15px;
        }
        /* Yeni Eklenen Sinyal Kutusu Tasarımı */
        .signal-box {
            padding: 12px;
            border-radius: 8px;
            font-size: 13px;
            font-weight: 600;
            margin-bottom: 10px;
            display: flex;
            align-items: center;
            gap: 10px;
        }
    </style>
""", unsafe_allow_html=True)

API_URL = "http://127.0.0.1:8000"

transaction_types = {
    "Online Alışveriş (E-Ticaret)": 0,
    "Fiziksel POS İşlemi": 1,
    "ATM / Nakit Çekim": 2
}

merchant_categories = {
    "Elektronik & Teknoloji": 15,
    "Market & Gıda": 5,
    "Giyim & Aksesuar": 10,
    "Restoran & Kafe": 3,
    "Seyahat & Ulaşım": 25,
    "Diğer / Lüks Tüketim": 40
}

countries = {
    "Türkiye (Yerel)": 1,
    "Yurt Dışı (Avrupa)": 10,
    "Yurt Dışı (Amerika)": 20,
    "Riskli Kabul Edilen Bölge": 50
}

if 'current_page' not in st.session_state:
    st.session_state.current_page = 'Duygu Analizi (Sentiment)'


if 'fraud_history' not in st.session_state:
    st.session_state.fraud_history = []



with st.sidebar:
    st.image("https://img.icons8.com/fluent/96/shield.png", width=80)
    st.title("FinGuardAI")
    st.caption("Bütünleşik Finansal Risk ve Karar Destek Sistemi")
    st.divider()
    
    st.markdown("### 📊 Analiz Modülleri")
    
    if st.button(
        "📰 Duygu Analizi (Sentiment)", 
        key="btn_sent", 
        type="primary" if st.session_state.current_page == 'Duygu Analizi (Sentiment)' else "secondary"
    ):
        st.session_state.current_page = 'Duygu Analizi (Sentiment)'
        st.rerun()
        
    if st.button(
        "💳 Dolandırıcılık Tespiti (Fraud)", 
        key="btn_fraud", 
        type="primary" if st.session_state.current_page == 'Dolandırıcılık Tespiti (Fraud)' else "secondary"
    ):
        st.session_state.current_page = 'Dolandırıcılık Tespiti (Fraud)'
        st.rerun()
    
    st.divider()
    st.caption("🛡️ **Sistem Durumu:** Çevrimiçi")
    st.caption("⚙️ **Versiyon:** v1.5 (Pro Engine)")



if st.session_state.current_page == 'Duygu Analizi (Sentiment)':
    st.title("📰 Finansal Haber Duygu Analizi")
    st.markdown("Piyasa duyarlılığını ve finansal metinlerin risk/fırsat algısını yapay zeka ile ölçün.")
    st.divider()

    tab1, tab2 = st.tabs(["🌍 Canlı Global Finans Basını", "✍️ Manuel Metin Analizi"])

    with tab1:
        st.markdown("### Canlı Küresel Akış ve Yapay Zeka Yorumları")
        st.caption("Yahoo Finance altyapısından en güncel global finans gelişmeleri anlık olarak çekilir.")
        st.markdown("<br>", unsafe_allow_html=True)
        
        with st.spinner("Küresel finans ağından en taze haberler çekiliyor..."):
            try:
                fresh_url = f"https://finance.yahoo.com/news/rss?t={int(time.time())}"
                feed = feedparser.parse(fresh_url)
                
                if feed.entries:
                    for i, entry in enumerate(feed.entries[:5]):
                        news_text = f"{entry.title}. {entry.get('summary', '')}"
                        sentiment_status = "ANALİZ EDİLEMEDİ"
                        confidence_score = "%0.00"
                        sentiment_color = "#1d4ed8"
                        
                        try:
                            response = requests.post(f"{API_URL}/predict/sentiment", json={"text": news_text}, timeout=5)
                            if response.status_code == 200:
                                result = response.json()
                                sentiment_status = result['sentiment'].upper()
                                confidence_score = f"%{result['confidence']*100:.2f}"
                                if "POS" in sentiment_status:
                                    sentiment_color = "#10b981"
                                elif "NEG" in sentiment_status:
                                    sentiment_color = "#ef4444"
                        except Exception:
                            pass

                        st.markdown(f"""
                            <div class="news-card-box">
                                <div class="news-title-text">{entry.title}</div>
                                <div class="news-date-text">Yayınlanma Tarihi: {entry.get('published', 'Bilinmiyor')}</div>
                                <div style="display: flex; gap: 15px; margin-top: 10px;">
                                    <div class="metric-card" style="flex: 1; padding: 10px; background-color: #f8fafc;">
                                        <div class="metric-label" style="font-size: 11px;">Yapay Zeka Analizi</div>
                                        <div class="metric-value" style="font-size: 18px; color: {sentiment_color};">{sentiment_status}</div>
                                    </div>
                                    <div class="metric-card" style="flex: 1; padding: 10px; background-color: #f8fafc;">
                                        <div class="metric-label" style="font-size: 11px;">Güven Skoru</div>
                                        <div class="metric-value" style="font-size: 18px; color: #475569;">{confidence_score}</div>
                                    </div>
                                </div>
                            </div>
                        """, unsafe_allow_html=True)
                else:
                    st.warning("Şu an güncel global haber akışına ulaşılamıyor.")
            except Exception as e:
                st.error(f"Haber akışı otomatik çekilirken bir hata oluştu: {e}")

    with tab2:
        user_input = st.text_area("Haber / Analiz Metni:", placeholder="Metni buraya girin...", height=150)
        col_btn, _ = st.columns([1, 2])
        with col_btn:
            analyze_btn = st.button("🚀 Metni Analiz Et", type="primary", key="action_sent")

        if analyze_btn and user_input:
            with st.spinner('Analiz ediliyor...'):
                try:
                    response = requests.post(f"{API_URL}/predict/sentiment", json={"text": user_input})
                    result = response.json()
                    st.markdown("### 📊 Analiz Sonuçları")
                    res_col1, res_col2 = st.columns(2)
                    with res_col1:
                        st.markdown(f'<div class="metric-card"><div class="metric-label">Duygu</div><div class="metric-value" style="color: #1d4ed8;">{result["sentiment"].upper()}</div></div>', unsafe_allow_html=True)
                    with res_col2:
                        st.markdown(f'<div class="metric-card"><div class="metric-label">Güven</div><div class="metric-value" style="color: #2563eb;">%{result["confidence"]*100:.2f}</div></div>', unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Hata: {e}")


elif st.session_state.current_page == 'Dolandırıcılık Tespiti (Fraud)':
    st.title("💳 Gelişmiş İşlem Risk Analizi")
    st.markdown("İşlem metriklerini girerek yapay zeka tabanlı anomali ve dolandırıcılık skorlamasını başlatın.")
    st.divider()

    # Form Alanları
    with st.container():
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("#### 💰 Finansal Veriler")
            f_amount = st.number_input("İşlem Tutarı (Tutar)", min_value=0.0, value=100.0, step=50.0)
            selected_type_label = st.selectbox("İşlem Tipi", options=list(transaction_types.keys()))
            f_type = transaction_types[selected_type_label]
            f_hour = st.slider("İşlem Saati (0-23)", 0, 23, 12)

        with col2:
            st.markdown("#### 🌍 Konum ve Kategori")
            selected_merchant_label = st.selectbox("Mağaza Kategorisi", options=list(merchant_categories.keys()))
            f_merchant = merchant_categories[selected_merchant_label]
            
            selected_country_label = st.selectbox("İşlemin Yapıldığı Bölge", options=list(countries.keys()))
            f_country = countries[selected_country_label]

        with col3:
            st.markdown("#### 🛡️ Ağ ve Güvenlik Skorları")
            f_device_risk = st.slider("Cihaz Risk Skoru (Donanım)", 0.0, 100.0, 20.0)
            f_ip_risk = st.slider("IP Risk Skoru (Network)", 0.0, 100.0, 15.0)

    features = [float(f_amount), int(f_type), int(f_merchant), int(f_country), int(f_hour), float(f_device_risk), float(f_ip_risk)] 

    st.divider()
    
    col_btn, _ = st.columns([1, 2])
    with col_btn:
        fraud_btn = st.button("🔴 Risk Analizini Başlat", type="primary", key="action_fraud")
        
    if fraud_btn:
        try:
            with st.spinner('Pattern ve anomali analizi yapılıyor...'):
                response = requests.post(f"{API_URL}/predict/fraud", json={"features": features})
                
                if response.status_code == 200:
                    result = response.json()
                    prob = result['fraud_probability']
                    
                    st.markdown("### 🎯 Risk Değerlendirme Raporu")

                    if result['is_fraud'] == 1:
                        st.error(f"⚠️ {result['status']}: Yüksek dolandırıcılık riski tespit edildi!")
                    else:
                        st.info(f"🔵 {result['status']}: İşlem güvenli sınırlar içerisinde.")
                    
                    st.progress(prob)
                    
                    border_color = '#ef4444' if result['is_fraud'] == 1 else '#1d4ed8'
                    value_color = '#ef4444' if result['is_fraud'] == 1 else '#1d4ed8'
                    
                    st.markdown(f"""
                        <div class="metric-card" style="margin-top: 15px; border-left: 6px solid {border_color}; text-align: left; padding-left: 30px;">
                            <div class="metric-label">Hesaplanan Toplam Risk Skoru</div>
                            <div class="metric-value" style="color: {value_color}; font-size: 34px;">
                                %{result['risk_percentage']}
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    # --- YENİ ÖZELLİK 3: GEÇMİŞE KAYDETME MEKANİZMASI ---
                    st.session_state.fraud_history.append({
                        "Zaman Stampt": time.strftime("%H:%M:%S"),
                        "Tutar (TL)": f_amount,
                        "Bölge Skoru": f_country,
                        "Risk Skoru": f"%{result['risk_percentage']}",
                        "Durum": "ŞÜPHELİ (Fraud)" if result['is_fraud'] == 1 else "GÜVENLİ"
                    })
                else:
                    st.error(f"Sunucu Hatası: Kod {response.status_code}")
        except Exception as e:
            st.error(f"Bağlantı hatası: {e}")

    if st.session_state.fraud_history:
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown("### 📜 Bu Oturumda İncelenmiş İşlemler Denetim Günlüğü")
        df_history = pd.DataFrame(st.session_state.fraud_history)
        st.dataframe(df_history, use_container_width=True)
        
        
        csv_data = df_history.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Denetim Raporunu CSV Olarak İndir",
            data=csv_data,
            file_name="finguard_risk_report.csv",
            mime="text/csv"
        )