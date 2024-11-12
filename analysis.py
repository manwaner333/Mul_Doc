import pickle
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler
import json
from sklearn.cluster import DBSCAN





if __name__ == "__main__":
    file_path = 'data/answer_file.bin'
    with open(file_path, "rb") as f:
        responses = pickle.load(f)
    
    hidden_states_list = []

    for idx, response in responses.items():
        question_id = response['question_id']
        x = response['x']
        y = response['y']
        tokens = response['tokens']
        token_logprobs = response['token_logprobs']
        token_entropies = response['token_entropies']
        tokens_idx = response['tokens_idx']
        token_logprob_entro = response['token_logprob_entro']
        hidden_states = response['hidden_states']
        hidden_states_list.append(hidden_states)
    
    
    normalized_data = 1 / (1 + np.exp(-np.array(hidden_states_list)))
    
    # TSNE
    # tsne = TSNE(n_components=3, perplexity=10, n_iter=2000)
    # X_tsne = tsne.fit_transform(normalized_data)

    # PCA
    pca = PCA(n_components=3)
    X_tsne = pca.fit_transform(normalized_data)


    # KMeans
    # kmeans = KMeans(n_clusters=2, random_state=0)
    # clusters = kmeans.fit_predict(X_tsne)
    # DBSCAN
    # dbscan = DBSCAN(eps=0.5, min_samples=5)
    # clusters = dbscan.fit_predict(X_tsne)

    # Plotting in 3D
    fig = plt.figure(figsize=(5.8, 5.2))
    ax1 = fig.add_subplot(111, projection='3d')
    
    for i in range(len(X_tsne)):
        s2 = ax1.scatter(X_tsne[i, 0], X_tsne[i, 1], X_tsne[i, 2], color='y', alpha=0.7)
    
    # ax1.view_init(azim=30)
    # ax1.set_title('Layer=16 in 3D PCA-reduced Space')
    # ax1.set_xlabel('PC1', fontdict={'fontsize': 10, 'fontweight': 'bold'})
    # ax1.set_ylabel('PC2', fontdict={'fontsize': 10, 'fontweight': 'bold'})
    # ax1.set_zlabel('PC3', fontdict={'fontsize': 10, 'fontweight': 'bold'})
    # ax1.set_xlim(-40, 20)
    # ax1.set_ylim(-40, 30)
    # ax1.set_zlim(-40, 20)
    ax1.tick_params(axis='x', labelsize=12)
    ax1.tick_params(axis='y', labelsize=12)
    ax1.tick_params(axis='z', labelsize=12)

    # ax1.legend((s1, s2), ("True", "False"), loc='best', prop={'size': 10, 'weight': 'bold'})
    plt.tight_layout()
    # plt.show()
    
    save_path = "data/embedding_reduced_space_32.png"
    plt.savefig(save_path, dpi=150, format='png')
        
    
    
    