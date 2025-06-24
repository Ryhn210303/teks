import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation

from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

from gensim.models import CoherenceModel
from gensim.corpora import Dictionary

st.set_option('deprecation.showPyplotGlobalUse', False)
st.title("📰 Analisis Tren Topik Berita Detik.com")

# Upload file
uploaded_file = st.file_uploader("📤 Upload file Excel (.xlsx) dengan kolom 'content' dan 'tanggal'", type=["xlsx"])
jumlah_topik = st.slider("🔢 Pilih jumlah topik LDA", 2, 10, 5)

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    st.subheader("📄 Data Awal")
    st.dataframe(df.head())

    # ====================
    # Preprocessing
    # ====================
    st.subheader("🧹 Preprocessing Teks")

    stop_factory = StopWordRemoverFactory()
    stopwords = set(stop_factory.get_stop_words())
    stemmer = StemmerFactory().create_stemmer()

    def preprocess(text):
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        words = text.split()
        words = [w for w in words if w not in stopwords]
        text = ' '.join(words)
        text = stemmer.stem(text)
        return text

    df['clean_content'] = df['content'].astype(str).apply(preprocess)

    # WordCloud
    st.subheader(" WordCloud")
    text_korpus = " ".join(df['clean_content'])
    wordcloud = WordCloud(width=1000, height=600, background_color='white').generate(text_korpus)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    st.pyplot()

    # Distribusi kata
    st.subheader(" Distribusi Jumlah Kata per Artikel")
    df['word_count'] = df['clean_content'].apply(lambda x: len(x.split()))
    sns.histplot(df['word_count'], bins=50, color='green', kde=True)
    st.pyplot()

    # ====================
    # TF-IDF & LDA
    # ====================
    st.subheader(" Topik Modeling (LDA + TF-IDF)")

    tfidf = TfidfVectorizer(max_df=0.95, min_df=2, stop_words=list(stopwords))
    tfidf_matrix = tfidf.fit_transform(df['clean_content'])

    lda_model = LatentDirichletAllocation(n_components=jumlah_topik, max_iter=10, learning_method='online', random_state=42)
    lda_model.fit(tfidf_matrix)

    for i, topic in enumerate(lda_model.components_):
        st.markdown(f"**Topik {i+1}:**")
        top_words = [tfidf.get_feature_names_out()[j] for j in topic.argsort()[-10:][::-1]]
        st.write(", ".join(top_words))

    # ====================
    # Evaluasi Coherence
    # ====================
    st.subheader(" Evaluasi Coherence Score")

    tokenized = [text.split() for text in df['clean_content']]
    dictionary = Dictionary(tokenized)
    corpus = [dictionary.doc2bow(text) for text in tokenized]

    from gensim.models.ldamodel import LdaModel
    lda_gensim = LdaModel(corpus=corpus, id2word=dictionary, num_topics=jumlah_topik, passes=10)
    coherence_model = CoherenceModel(model=lda_gensim, texts=tokenized, dictionary=dictionary, coherence='c_v')
    score = coherence_model.get_coherence()

    st.metric(label="Coherence Score", value=f"{score:.4f}")
    st.info("Semakin tinggi nilainya (maks 1.0), semakin baik koherensi antar kata dalam topik.")

    # ====================
    # Distribusi Topik per Dokumen
    # ====================
    st.subheader(" Distribusi Topik (Heatmap)")
    topic_distribution = lda_model.transform(tfidf_matrix)
    df_topic = pd.DataFrame(topic_distribution, columns=[f'Topik_{i+1}' for i in range(jumlah_topik)])
    plt.figure(figsize=(12, 6))
    sns.heatmap(df_topic.head(20), annot=True, cmap="YlGnBu")
    st.pyplot()

    st.success(" Analisis selesai. Silakan eksplor hasil topik dan nilai koherensinya.")
