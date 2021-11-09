X = np.array([[1, 1], [2, 1], [3, 1.2], [4, 1], [5, 0.8], [6, 1]])
n, m, k = X.shape + (2, ) # X の行数列数とコンポーネント数
W = np.random.uniform(size = (n, k))
H = np.random.uniform(size = (k, m))

# NMF
for i in range(100):
    # update B
    A = X.T.dot(W)
    B = W.T.dot(W)
    for j in range(k):
        H[j, :] = H[j, :] + A[:, j] - H.T.dot(B[:, j])

    # update A
    C = X.dot(H.T)
    D = H.dot(H.T)
    for j in range(k):
        W[:, j] = W[:, j] * D[j, j] + C[:, j] - W.dot(D[:, 0])
        W[:, j] = W[:, j] / np.linalg.norm(W[:, j])

print(X)
print(W.dot(H))
