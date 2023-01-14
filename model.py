from tensorflow.keras.preprocessing.image import load_img, array_to_img, img_to_array
import numpy as np
import math # Membantu proses operasi matematika

def getDataORL():
    path = "orl_face/"
    nama_file_depan = 'ORL_'
    nama_file_belakang = '.jpg'
    images_arr = []
    for orang_ke in np.linspace(1, 40, 40).astype(np.int8): # Orang (40 Orang)
        for pose_ke in np.linspace(1, 10, 10).astype(np.int8): # Pose (10 Pose)
            nama_file = nama_file_depan + str(orang_ke) + '_' + str(pose_ke) + nama_file_belakang
            nama_file = path + nama_file
            image_arr = np.squeeze(img_to_array(load_img(nama_file, color_mode='grayscale')))
            images_arr.append(image_arr)
    return np.array(images_arr)

def PCAImages(data):
    # 1. Ukuran Data
    (n_images, n_Xpixel, n_Ypixel) = data.shape # (jumlah image, jumlah pixel image pada axis x, jumlah pixel image pada axis y)
    n_images = int(n_images)
    n_Xpixel = int(n_Xpixel)
    n_Ypixel = int(n_Ypixel)

    # 2. Hitung Rata-rata
    addition_image = 0
    for i in range(n_images):
        addition_image = np.add(addition_image, data[i])
    average = addition_image / n_images

    # 3. Hitung Kovarian dengan tiap Zero Mean pada image
    cov_matrix = 0
    for i in range(n_images):
        image = data[i]
        zero_mean = image - average
        cov_matrix = cov_matrix + (np.transpose(zero_mean) @ zero_mean)

    # 4. Hitung eigen vektor dan eigen value
    [eigen_value, eigen_vector] = np.linalg.eig(cov_matrix)
    # Proses Sorting (Mengurutkan eigen vektor dari yang terbesar)
    idx = np.argsort(eigen_value)[::-1][:10] # Proses memotong fitur
    eigen_value = eigen_value[idx]
    eigen_vector = eigen_vector[:,idx]

    # 5. Hitung Proyeksi Matriks
    matrix_projection = eigen_vector
    
    # 6. Hitung Matrix Bobot
    matrix_weights = []
    for i in range(n_images):
        image = data[i]
        matrix_weight = image @ eigen_vector
        matrix_weights.append(matrix_weight)
    matrix_weights = np.array(matrix_weights)

    # Contoh Hasil:
    # print("Zero Mean : ", zero_mean.shape) # (112, 92)
    # print("Kovarian : ", cov_matrix.shape) # (92, 92)
    # print("Eigen Value : ", eigen_value.shape) # (92,)
    # print("Eigen Vektor : ", eigen_vector.shape) # (92, 92)
    # print("Proyeksi Matriks : ", matrix_projection.shape) # (92, 92)
    # print("Bobot Matriks semua image : ", matrix_weights.shape) # (320, 112, 92)

    return matrix_projection, matrix_weights

def getWeightDataTest(data_test, projection_matrices):
    weight = data_test @ projection_matrices 
    return weight # keluaranya matrix 112 X 92

def getImagePosition(distances):
    # Mencari yang paling dekat
    minimum_index_distances = np.argmin(distances)
    # Menentukan OrangKe dan PoseKe
    jumlah_pose = 10
    baris_ke = minimum_index_distances + 1 # Baris ke berapa dalam dataset 400 itu
    orang_ke = int(math.ceil(baris_ke/jumlah_pose)) # 0.1 atau 0.9 akan menjadi 1 karena math.ceil
    pose_ke = np.mod(baris_ke, jumlah_pose) # Pose ke berapa dalam 8 pose tiap orang itu
    if pose_ke == 0:
        pose_ke = jumlah_pose
    return orang_ke, pose_ke

def classificationProcess(data_test_weight, data_train_weights, distance_type):
    # Calculate Distances
    distance_results = []
    for data_train_weight in data_train_weights:
        if distance_type == "manhattan":
            distance_result = np.abs(data_test_weight - data_train_weight).sum()
        elif distance_type == "euclidean":
            distance_result = np.sqrt(np.power((data_test_weight - data_train_weight), 2).sum())
        distance_results.append(distance_result)

    # PROSES KLASIFIKASI (menghitung jarak terdekat)
    orang_ke, pose_ke = getImagePosition(distance_results)

    # Search orang_ke, pose_ke
    filename = 'ORL_' + str(orang_ke) + '_' + str(pose_ke) + '.jpg'
    filename = 'orl_face/' + filename
    
    # Get Image Classified
    image_classified = load_img(filename, color_mode='grayscale', target_size=(112*3, 92*3))

    return image_classified

def modelPCA(data_test):
    data_test = np.squeeze(img_to_array(load_img(data_test, color_mode='grayscale', target_size=(112,92))))
    # Get Images Data Array from ORL Dataset
    dataset_image_orl_arr = getDataORL()

    # Get Matrices Projection and Weight Data Train (400 data)
    projection_data_train, weight_data_train = PCAImages(dataset_image_orl_arr)

    # Get Matrices Weight Data Test
    weight_data_test = getWeightDataTest(data_test, projection_data_train)

    # Classification Process
    image_classified = classificationProcess(weight_data_test, weight_data_train, 'euclidean')

    return image_classified

