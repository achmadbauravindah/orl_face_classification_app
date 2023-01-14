### Aplikasi ini digunakan untuk mengklasifikasikan citra dengan kelas ORL

### Klasifikasi citra dibangun dengan mencari jarak terdekat dari matriks melalui algoritma
### Euclidean dan Manhattan Distances

### Proses klasifikasi menerapkan konsep PCA untuk mengekstraksi fitur citra yang paling penting

import streamlit as st
import model

st.markdown("<h1 style='text-align:center'>ORL Face Classification</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align:center'>This Classification Process has built with PCA Concept and Similar Pixel (calculate with distances euclidean or manhattan)</h4>", unsafe_allow_html=True)
st.markdown("<h6 style='text-align:center'>Dataset: <a href='https://paperswithcode.com/dataset/orl'>https://paperswithcode.com/dataset/orl</a></h6>", unsafe_allow_html=True)

st.markdown("<h5 style='text-align:center'>---Take your face and Classify it---</h5>", unsafe_allow_html=True)

image = st.camera_input("Take a image", label_visibility='hidden')

if image:
    st.markdown("<h4 style='text-align:center'>Your Face is Similar with This Person ORL 🫣</h4>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    image_classified = model.modelPCA(image)
    col2.image(image_classified)

